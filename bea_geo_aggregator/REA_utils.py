# User notes
# - GDP: missing values for all reasons (e.g., "(D)", "(NA)", "(NM)" in the source data or "fischer" failures in calculations) will all be returned as (NA)

# Design notes:
# - Too hard to keep track separately of (D), NA, NM in the source data and then what calculations lead to NA in the results.
# - Convert TimePeriod to integer (since comparing), but keep Fips codes as strings.
# - The geo hierarchy is neatly represented as layers, where the components in a layer are disjoint and covering, 
#   so merge in as columns GeoAggXFips (X in [1,2]).
# - The industry hierarchy is not neatly represented such, so have separate data structure.
# - Currently don't have 'spans' for hierarchies. Possibly a good idea for geographic hierarchies, 
#   though more complicated for users.
# - If converting DataValue to an int, make sure to use int64 as some PI numbers exceed the int32 limit.

# Formatting: max line length 120. Do docstrings for public functions.

import pathlib
from typing import Optional, Union

import numpy as np
import pandas as pd
import tqdm
import beaapi


#### Constants
bea_regions = ["91", "92", "93", "94", "95", "96", "97", "98"]
metro_code = "998"
nonmetro_code = "999"
us_code = "00"
us_metro_code = us_code + metro_code
us_nonmetro_code = us_code + nonmetro_code
metro_name_part = "(Metropolitan Portion)"
nonmetro_name_part = "(Nonmetropolitan Portion)"
us_metro_name = f"United States {metro_name_part}"
us_nonmetro_name = f"United States {nonmetro_name_part}"
bundled_geo_levels = {"MSA":2, "PORT":3}  # value is how many aggregation level there are
#bundled_geo_levels = {"MSA":2, "MIC":1, "CSA":1, "PORT":3, "DIV":1} # current ones

# GDP constants
geo_has_additional_gdp_tables = ["MSA", "PORT"]  # available for tables CAGDP1, 8,9, 11
gdp_calced_names_primary = ['GDP_current_dollars', 'chain_type_q_index', 'real_GDP_chained_dollars', 'contrib_pct_change']
gdp_calced_names_unr = [name + '_unr' for name in gdp_calced_names_primary]
gdp_variable_table_mapping = {'GDP_current_dollars': 'CAGDP2', 'chain_type_q_index': 'CAGDP8', 'real_GDP_chained_dollars': 'CAGDP9', 
                              'contrib_pct_change': 'CAGDP11'} #Note also CAGDP1 (summary)
cagdp_start_year = 2001
# PI constants
CAINC_tables = ["CAINC1", "CAINC4", "CAINC5N", "CAINC6N", "CAINC30"]  # "CAINC91" BEA doesn't provide aggregates
pi_tables_div_lines = {"CAINC1": [(1,2,3, 1000)], #numerator, denominator, ratio, ratio multiple
                       "CAINC4" : [(10, 20, 30, 1000)],
                       "CAINC5N": [(10, 20, 30, 1000)],
                       "CAINC30": [(10,100,110, 1000), (45,100,120, 1000), (50,100,130, 1000), (60,100,140, 1000), 
                                (70,100,150, 1000), (80,100,160, 1000), (90,100,170, 1000)]}
cainc_start_year = 1969
old_CT_cnty_fips = ["09001", "09003", "09005", "09007", "09009", "09011", "09013", "09015"]
new_CT_cnty_fips = ["09110", "09120", "09130", "09140", "09150", "09160", "09170", "09180", "09190"]
CT_new_start = 2024



#### Functions

def integrate_dict_output(df_agg):
    dfList = [df_agg['CAGDP2'][['LineCode', 'TimePeriod', "GeoFips", "GeoName", "Description"]]]
    dfList += [(make_nullcomment_NA_if_available(df_agg[tablename])[['LineCode', 'TimePeriod', "GeoFips", 'DataValue']]
         .rename(columns={"DataValue": vname}))
         for vname, tablename in gdp_variable_table_mapping.items()]

    from functools import reduce
    return reduce(lambda x, y: pd.merge(x, y, how="left", on=['LineCode', 'TimePeriod', "GeoFips"]), dfList)


bea_geofile_base='https://apps.bea.gov'
def dl_bea_reg_geofile(geotype):
    import requests  
    import json

    url = rf'{bea_geofile_base}/regionalcore/data/regional/msaList?mtype={geotype}'
    response = requests.get(url)
    json_data = json.loads(response.content)
    df = pd.DataFrame(json_data["Geo"], dtype="string")
    df.drop(columns=['ValidFlag', 'Letter'])
    return df[['NameParent', 'CodeParent', 'NameChild', 'CodeChild']]

def get_geo_data_bea(geo_agg):
    assert geo_agg in bundled_geo_levels.keys(), f"geo_agg must be one of {list(bundled_geo_levels.keys())}"
    if geo_agg=="MSA":
        MetroSA = dl_bea_reg_geofile(5)[['NameChild', 'CodeChild', 'NameParent', 'CodeParent']]
        MetroSA = MetroSA.rename(columns={'NameChild':"County name", 'CodeChild':"County_GeoFips", 'NameParent':"Component name", 'CodeParent':"Component_GeoFips"})
        MetroSA['GeoAgg2_name'] = us_metro_name
        MetroSA['GeoAgg2Fips'] = us_metro_code
        MetroSA = MetroSA.rename(columns={"Component name":"GeoAgg1_name", "Component_GeoFips":"GeoAgg1Fips"})
        return MetroSA

    if geo_agg=="PORT":
        # 1. get county x (Non)Metro -> State x (Non)Metro
        Metro_portions = dl_bea_reg_geofile(10)[['NameChild', 'CodeChild', 'NameParent', 'CodeParent']]
        Nonmetro_portions = dl_bea_reg_geofile(11)[['NameChild', 'CodeChild', 'NameParent', 'CodeParent']]
        PORT_df = pd.concat([Metro_portions, Nonmetro_portions], axis=0, ignore_index=True).rename(columns={'NameChild':"County name", 'CodeChild':"County_GeoFips", 'NameParent':"GeoAgg1_name", 'CodeParent':"GeoAgg1Fips"})
        # 2. add -> regions x (Non)Metro
        PORT_df['state_fips'] = PORT_df['GeoAgg1Fips'].str.slice(0,2) + "000"
        PORT_df['metro_code'] = PORT_df['GeoAgg1Fips'].str.slice(2,5)
        bea_regions_df = dl_bea_reg_geofile(2)[['NameChild', 'CodeChild', 'NameParent', 'CodeParent']]
        PORT_df = PORT_df.merge(bea_regions_df.rename(columns={'NameChild':"state_name", 'CodeChild':"state_fips", 'NameParent':"region_name", 'CodeParent':"region_code"}), how='left', on='state_fips')
        metro_code2name_df = pd.DataFrame({'metro_code':[metro_code, nonmetro_code], 'metro_name_part':[metro_name_part, nonmetro_name_part]} ,dtype="string")
        PORT_df = PORT_df.merge(metro_code2name_df, how='left', on='metro_code')
        PORT_df['GeoAgg2_name'] = PORT_df['region_name'] + " " + PORT_df['metro_name_part']
        PORT_df['GeoAgg2Fips'] = PORT_df['region_code'].str.slice(0,2) + PORT_df['metro_code']
        # 3. add -> US x (Non)Metro
        PORT_df['GeoAgg3_name'] = "United States " + PORT_df['metro_name_part']
        PORT_df['GeoAgg3Fips'] = us_code + PORT_df['metro_code']
        PORT_df = PORT_df.drop(columns=["region_name", 'region_code', 'state_fips', "state_name", 'metro_code', 'metro_name_part'])
        return PORT_df


def gdp_combine_cnty_nominals_geoagg_measures(df_agg, df=None, geofips_filter=None, geo_agg="MSA", bea_key=None, verbosity:int=1):
    # Try to work with both (a) user from main (df_agg is dict and need to reget counties), and (b) they got df themselves (either source pull or histdata ext)
    if isinstance(df_agg, dict):
        df_agg = integrate_dict_output(df_agg)
    if 'child_count' in df_agg.columns: #(b)
        df_agg = df_agg.drop(columns=['child_count'])
    if "GeoFips" in df_agg.columns: #(a)
        df_agg = df_agg.rename(columns={"GeoFips":"GeoAggFips"})
    if geofips_filter is None: #can only handle 1 geoagg
       geofips_filter = df_agg.GeoAggFips.iloc[0]
    if df is not None: # (b)
        #df_prices = df[df.GeoFips==df.GeoFips.iloc[0]][['LineCode', 'TimePeriod', 'price']]
        if 'GeoAgg1Fips' in df.columns:
            df = df[df.GeoAgg1Fips==geofips_filter].copy()
        if 'GeoName' in df.columns:
            df['Name'] = df['GeoName'] + " GDP_current_dollars"
        else:
            df['Name'] = df['GeoFips'] + " GDP_current_dollars"
        if 'price' in df.columns:
            df = df.drop(columns=['price'])
    else: # have to download
        start_year, last_year = df_agg.TimePeriod.min(), df_agg.TimePeriod.max()
        nominals, prices, industries, ref_year = pull_source_data_gdp(start_year, last_year, bea_key, geofips_filter=geofips_filter,
                                                                  geo_agg=geo_agg, verbosity=verbosity)
        nominals = make_nullcomment_NA_if_available(nominals)
        nominals = nominals.drop(columns=['CL_UNIT', 'UNIT_MULT', 'NoteRef'])
        df, ind_rea_va_xw = combine_nominals_prices(nominals, prices, industries)
        df['LineCode'] = df['LineCode'].astype(int)
        
        df['Name'] = df['GeoName'] + " GDP_current_dollars"
        df = make_nullcomment_NA_if_available(df)
        df = df.rename(columns={'DataValue':'GDP_current_dollars'})[['LineCode','TimePeriod', 'Name', 'GDP_current_dollars']]

    df_s = df.pivot(index=['LineCode','TimePeriod'], columns='Name', values='GDP_current_dollars').reset_index()

    df_agg = df_agg[df_agg.GeoAggFips==geofips_filter].drop(columns=['GeoAggFips'])
    if 'GeoName' in df_agg.columns:
        agg_name = df_agg['GeoName'].iloc[0]
        df_agg = df_agg.drop(columns=['GeoName'])
    else:
        agg_name = geofips_filter
    df_ret = df_s.merge(df_agg, on=['LineCode', 'TimePeriod'], how="inner")
              
    for c in ['GDP_current_dollars', 'chain_type_q_index', 'real_GDP_chained_dollars', 'contrib_pct_change']:
       df_ret = df_ret.rename(columns={c:f"{agg_name} {c}"})

    initial_cols = ['LineCode'] + (['Description'] if 'Description' in df_ret.columns else []) +['TimePeriod']
    df_ret = df_ret[initial_cols + [c for c in df_ret.columns if c not in initial_cols]]
    return df_ret

def dict_ret_to_excel(gdp_dfs, gdp_outfname):
    with pd.ExcelWriter(gdp_outfname) as writer:
        for tablename, df_agg in gdp_dfs.items():
            df_agg.to_excel(writer, sheet_name=tablename, index=False)

def extract_regional_data_histdata(fname:Union[str,pathlib.Path], geofips_filter:Optional[str]=None, 
                                   ext_start_year:Optional[int]=None, last_year:Optional[int]=None, 
                                   unit_mult:Optional[int]=None, 
                                   geo_agg:Optional[Union[str,pathlib.Path,pd.DataFrame]]=None,
                                   drop_desc_linecode:bool=True, 
                                   return_line_descs:bool=False):
    # Extract dataframe and basic cleanup
    regional_table = (pd.read_csv(fname, encoding='latin-1', dtype='string')
                      .drop(columns=['Region', 'IndustryClassification'])
                      .rename(columns={'Unit':'CL_UNIT', 'GeoFIPS':'GeoFips'}))
    regional_table = regional_table[~regional_table[regional_table.columns[1]].isna()]
    regional_table['GeoFips'] = regional_table['GeoFips'].str.replace('"','').str.strip()

    # get industries while easy
    line_descs = regional_table[['LineCode', 'Description']].rename(columns={'LineCode':'Key', 'Description':'Desc'})
    line_descs['Desc'] = line_descs['Desc'].str.replace(' [0-9]/', '', regex=True).str.strip()
    line_descs = line_descs.drop_duplicates()
    #display(industries.head())
    
    #filter by component counties
    if geofips_filter is not None:
        if geo_agg is not None:
            geofips_c = get_counties_for_geoagg(geofips_filter, geo_agg)
            regional_table = regional_table[regional_table.GeoFips.isin(geofips_c)]
        else:
            regional_table = regional_table[regional_table.GeoFips==geofips_filter]
    
    #Make look like API returned table
    regional_table['Code'] = regional_table['TableName']+"-"+regional_table['LineCode']
    regional_table = regional_table.drop(columns=['TableName'])
    melt_ids = ['GeoFips', 'GeoName', 'Code', 'CL_UNIT']
    if drop_desc_linecode:
        regional_table = regional_table.drop(columns=['Description', 'LineCode'])
    else:
        melt_ids = melt_ids + ['Description', 'LineCode']
    
    
    df = (regional_table.melt(id_vars=melt_ids, var_name='TimePeriod', 
                              value_name='DataValue'))
    if unit_mult is not None:
        df['UNIT_MULT'] = unit_mult
    df['NoteRef'] = pd.Series("", index=df.index, dtype="string")
    is_non_numeric = pd.to_numeric(df['DataValue'], errors='coerce').isnull()
    df.loc[is_non_numeric, 'NoteRef'] = df['DataValue']
    df.loc[is_non_numeric, 'DataValue'] = '0'
    df['DataValue'] = pd.to_numeric(df['DataValue'])
    if df.dtypes['DataValue']=='Int64':
        df['DataValue'] = df['DataValue'].astype('int64')
    if df.dtypes['DataValue']=='Float64':
        df['DataValue'] = df['DataValue'].astype('float64')
    #filter by year
    if ext_start_year is not None:
        df = df[df['TimePeriod'].astype(int)>=ext_start_year]
    if last_year is not None:
        df = df[df['TimePeriod'].astype(int)<=last_year]

    if return_line_descs:
        return df, line_descs
    
    return df

def extract_GDPbyIndustry_data_histdata(workbook_fname, sheet_name, ext_start_year=None, last_year=None, skiprows=7):
    # get table and basic cleanup
    GDPbyIndustry_table = pd.read_excel(workbook_fname, sheet_name=sheet_name, skiprows=7).rename(columns={'Unnamed: 1': "IndustrYDescription"}).drop(columns=['Unnamed: 2'])
    GDPbyIndustry_table = GDPbyIndustry_table[~GDPbyIndustry_table[GDPbyIndustry_table.columns[1]].isna()]
    GDPbyIndustry_table = GDPbyIndustry_table[~GDPbyIndustry_table["Line"].isna()]
    GDPbyIndustry_table['IndustrYDescription'] = GDPbyIndustry_table['IndustrYDescription'].str.strip()
    
    # return the way API does
    GDPbyIndustry_table['IndustrYDescription'] = GDPbyIndustry_table['IndustrYDescription'].str.replace(r'.[0-9].','', regex=True) # remove "\#\"
    
    df = GDPbyIndustry_table.melt(id_vars=['Line', 'IndustrYDescription'], var_name='Year', value_name='DataValue')
    
    # Filter by year
    if ext_start_year is not None:
        df = df[df['Year'].astype(int)>=ext_start_year]
    if last_year is not None:
        df = df[df['Year'].astype(int)<=last_year]

    return df

def extract_source_data_histdata_gdp(Regional_CAGDP2_path:Union[str,pathlib.Path], 
                                     GDPbyIndustry_ValueAdded_path:Union[str,pathlib.Path], start_year: int, 
                                     last_year:int, ref_year:int, geofips_filter:Optional[str]=None, 
                                     geo_agg:Optional[str]=None):
    '''Extract data from spreadsheets (from zips) from apps.bea.gov/histdata/ to provide data that can be used by do_merging_and_calculations_gdp()'''
    ext_start_year = min(start_year-1, ref_year)
    ## Regional CAGDP2
    nominals, industries = extract_regional_data_histdata(Regional_CAGDP2_path, geofips_filter, 
                                                          ext_start_year=ext_start_year, last_year=last_year, 
                                                          unit_mult=3, geo_agg=geo_agg, return_line_descs=True)
    industries['Desc'] = "[CAGDP2] Gross Domestic Product (GDP): " + industries['Desc']
    #display(nominals.head())

    ##GDP by industry
    prices = extract_GDPbyIndustry_data_histdata(GDPbyIndustry_ValueAdded_path, "TVA104-A", ext_start_year, last_year, 
                                                 skiprows=7)
    #display(prices.head())
    
    
    return nominals, prices, industries

def display_df_auto(df):
    fmt_dict = {}
    if 'price' in df.columns:
        fmt_dict['price']='{:.3f}'
    if 'Code' in df.columns: #single table
        if df['Code'].iloc[0].startswith("CAGDP2") or df['Code'].iloc[0].startswith("CAGDP9"):
            fmt_dict['DataValue'] = '{:.0f}'
        if df['Code'].iloc[0].startswith("CAINC"):
            for c in df.columns:
                if c.endswith("DataValue"):
                    fmt_dict[c] = '{:.0f}'
    else: # integrated
        for c in df.columns:
            if c.endswith('GDP_current_dollars') or c.endswith('real_GDP_chained_dollars'):
                fmt_dict[c] = '{:.0f}'
    sty = df.style.hide(axis="index").format(fmt_dict)
    if 'Description' in df.columns:
        sty = sty.set_properties(subset=['Description'], **{'white-space': 'pre', 'text-align': 'left'})
    
    display(sty)

def display_notes(df):
    notes = pd.DataFrame(df.attrs.get('detail', {}).get('Notes',[]))
    if notes.shape[0]>0:
        print("Notes:")
        for row in notes.itertuples(index=False):
            print(f"- {row.NoteRef}: {row.NoteText}")
        #display(notes)
    else:
        print("No notes.")


def display_with_notes(df):
    """Used from within Jupyter notebooks. Display's dataframe and then notes (if available)

    Args:
        df (pd.DataFrame): Data
    """
    display_df_auto(df)
    display_notes(df)


def round_std(val, decimals=0):
    # Rounds half-up, rather than to even (bankers rounding). This is what central systems do.
    factor = 10 ** decimals
    return np.sign(val) * np.floor(np.abs(val) * factor + 0.5) / factor


# Build parent->leaves (ultimate descendants) dictionary
def build_descendants_map(dfrp):
    ind_parents = dfrp.Parent[dfrp.Parent!=""].unique()
    parent_leaves = {}
    parent_descendants = {}
    for ind_parent in ind_parents:
        mytree = set([ind_parent])
        while True:
            size_prev = len(mytree)
            mytree = mytree.union(set(list(dfrp.Child[dfrp.Parent.isin(mytree)].values)))
            if len(mytree)==size_prev:
                break
        parent_descendants[ind_parent] = list(mytree - set([ind_parent]))
        parent_leaves[ind_parent] = list(set(dfrp.loc[dfrp.leaf & dfrp.Child.isin(mytree),'Child']))
    
    return parent_leaves, parent_descendants



def pull_regional_data(tablename, geo_level, start_year, last_year, bea_key, verbosity=0):
    # Helper to pull all line-codes for a Regional table (can't do in one API call)
    # retains table notes
    assert start_year<=last_year, "start_year must not be greater than last_year"
    years = ",".join(str(l) for l in list(range(start_year, last_year+1)))

    # faster to pull by line-code
    line_codes = beaapi.get_parameter_values_filtered(bea_key, 'Regional', 'LineCode', TableName=tablename)
    dfs = []
    notes = []
    for row in tqdm.tqdm(line_codes.itertuples(index=False), total=line_codes.shape[0], disable=verbosity==0, 
                         desc="Pulling source data. Linecodes"):
        df_i = beaapi.get_data(bea_key, 'Regional', TableName=tablename, LineCode=row.Key, year=years, 
                               GeoFips=geo_level)
        notes.append(pd.DataFrame(df_i.attrs.get('detail', {}).get('Notes',[])))
        dfs.append(df_i)
    df = pd.concat(dfs, ignore_index=True)
    df.attrs['detail'] = {"Notes":pd.concat(notes, ignore_index=True).drop_duplicates().to_dict(orient='records')}
    return df, line_codes


def conv_REA_csv2api(df, tablename):
    # Converts an REA csv file from iTables to a table like the API returns and converted to long-format

     # remove header lines like "Addenda:"/"Income by place of residence"
    df = df[~((df.LineCode=="") | (df.LineCode.isnull()))]

    # Make long-format
    df = df.melt(id_vars=['GeoFips', 'GeoName', 'LineCode', 'Description'], var_name='TimePeriod', 
                 value_name='DataValue')
    
    # API has composite code rather than LineCode
    df['Code'] = f"{tablename}-" + df['LineCode']
    
    # API has numeric DataValue with non-numbers listed in new NoteRef column
    data_conv = pd.to_numeric(df['DataValue'], errors="coerce")
    df['NoteRef'] = pd.Series("", index=df.index, dtype="string")
    df.loc[data_conv.isna(),'NoteRef'] = df['DataValue']
    df['DataValue'] = data_conv.fillna(0)

    df = df[['Code', 'GeoFips', 'GeoName', 'TimePeriod', 'DataValue', 'NoteRef']]
    return df


# Used for pandas-groupby to propogate nan's.  to omit NAs use df_group_gb.sum(min_count=1).reset_index()
def sum_prop_na(g):
    # https://github.com/pandas-dev/pandas/issues/20824#issuecomment-705376621
    return np.sum(g.values)


def assert_contributions_check(df, parent_child_map=None, atol=1e-6, verbosity=0):
    if parent_child_map is None:
        dfrp = pd.read_csv(pathlib.Path(__file__).parent.joinpath("metadata/dfrp.csv"), dtype="string").drop(columns=['Label'])
        dfrp[['Child', 'Parent']] = dfrp[['Child', 'Parent']].astype(int)
        ind_parents = dfrp.Parent[dfrp.Parent!=""].unique()
        parent_child_map = {parent: list(set(dfrp.Child[dfrp.Parent==parent].values)) for parent in ind_parents}
    for ind_p, ind_children in parent_child_map.items():
        dfp = df[df['LineCode']==ind_p][['GeoFips', 'TimePeriod', 'DataValue']]
        dfp_child_agg = df[df['LineCode'].isin(ind_children)].groupby(['GeoFips', 'TimePeriod'], as_index=False)['DataValue'].agg(sum_prop_na)
        df_merge = dfp.merge(dfp_child_agg, how="left", on=['GeoFips', 'TimePeriod'], suffixes=("_p", "_child_agg"))
        df_merge['adiff'] = (df_merge['DataValue_p'] - df_merge['DataValue_child_agg']).abs()
        if verbosity>0:
            print(f"Checking parent {ind_p} against children {ind_children} (n={df_merge.shape[0]}, non-NA={df_merge['adiff'].notna().sum()})")
        ok_obs = (df_merge['adiff'].isna())  | (df_merge['adiff']<atol)
        if not ok_obs.all():
            print(df_merge[~ok_obs])
        assert ok_obs.all(), f"Difference between parent={ind_p} and child agg={ind_children} is too high:\n{df_merge[df_merge['adiff']>=atol]}"

def accept_sum_aresidual(aresid, count):
    # Returns whether the absolute residual between a sum of rounded subaggregates and a total is acceptable 
    # (just due to rounding).
    # - count is number of rounded subaggregates that are being summed (doesn't include the total term)
    return aresid<=np.floor(count/2)

def same_sums_rounded(total, sum_rounds, count):
    """Returns whether the absolute residual between a sum of rounded subaggregates and a total is acceptable 
    (just due to rounding).

    Args:
        total (float, int): Total
        sum_rounds (float, int): Sum of (rounded) subaggregates
        count (int): number of rounded subaggregates that are being summed (doesn't include the total term)

    Returns:
        bool: Wether difference is acceptable due to rounding
    """
    return accept_sum_aresidual((total-sum_rounds).abs(), count)

def make_nullcomment_NA_if_available(df):
    if 'NoteRef' in df.columns:
        df['DataValue'] = df['DataValue'].astype(float) # from int
        df.loc[df['NoteRef'].fillna("").str.find("(")>=0,'DataValue'] = np.nan
    return df

def has_suppression(df, treat_nm="display"):
    # treat_nm can be ignore, display, include
    d_mask = df['NoteRef'].fillna("").str.find("(D)")>=0 #"(D)" or "(D) *" (for BEA combined areas)
    if treat_nm!="ignore":
        nm_mask = df['NoteRef'].fillna("").str.find("(NM)")>=0
        if (treat_nm=="display") & (nm_mask.sum()>0):
            print("Warning: Data has (NM) notations.")
            print(df[nm_mask])
        if treat_nm=="include":
            return d_mask | nm_mask
    return d_mask

def has_E_estimate(df):
    return df['NoteRef'].fillna("").str.find("E")>=0

def has_other_null(df): # aside from suppression (catch NA NM)
    o_mask = df['NoteRef'].fillna("").str.find("(")>=0
    d_mask = df['NoteRef'].fillna("").str.find("(D)")>=0
    return o_mask & ~d_mask

def make_suppressed_NA(df, treat_nm="display"):
    # treat_nm can be ignore, display, include
    df['DataValue'] = df['DataValue'].astype(float) # from int
    df.loc[has_suppression(df, treat_nm=treat_nm),'DataValue'] = np.nan
    return df


def get_geo_file(geo_agg, append_pre2024_CT_defs=True):
    # check if a df already, path, or key to bundled file
    if not isinstance(geo_agg, pd.DataFrame):
        if isinstance(geo_agg, str) and geo_agg in bundled_geo_levels.keys(): #key to bundled file
            df = get_geo_data_bea(geo_agg)
            if append_pre2024_CT_defs:
                df = pd.concat([df, pd.read_csv(pathlib.Path(__file__).parent.joinpath(f"metadata/{geo_agg}_old_CT_GeoList.csv"), dtype="string")], axis=0, ignore_index=True)
        else:
            try:
                df = pd.read_csv(geo_agg, dtype="string")
            except Exception as e:
                raise AssertionError(f"Your geo_agg={geo_agg} was unrecognized (not MSA or PORT) and it's not the name of a file with such data.")
    else:
        df = geo_agg

    df = df.rename(columns={"County_GeoFips": "GeoFips"}) # in case
    req_cols = ["County name", 'GeoFips', "GeoAgg1_name", 'GeoAgg1Fips']
    assert set(req_cols).issubset(set(df.columns)), f"Missing required columns: {set(req_cols) - set(df.columns)}"
    assert df.duplicated(subset=['GeoFips']).sum()==0, "Error: duplicate 'GeoFips' in geo_agg table"

    return df


def get_fips_name_mapping(geo_file):
    # reshape to so it's two columns from fips to name
    n_geo_levels = int(geo_file.shape[1]/2)-1
    dfs = [geo_file[['GeoFips', 'County name']].rename(columns={'County name':'GeoName', "County_GeoFips":"GeoFips"})]
    for geo_level_i in range(1, n_geo_levels+1):
        dfs.append((geo_file[[f'GeoAgg{geo_level_i}Fips', f'GeoAgg{geo_level_i}_name']]
                    .rename(columns={f'GeoAgg{geo_level_i}_name':'GeoName', f'GeoAgg{geo_level_i}Fips':"GeoFips"})))
    return pd.concat(dfs, axis=0, ignore_index=True).drop_duplicates()

def get_current_ref_year(bea_key):
    # Get the current reference year for GDP chained dollars from BEA API
    # At cretion this is 2017, but may change
    import re
    df = beaapi.get_data(bea_key, "Regional", TableName="CAGDP1", GeoFips="00000", Year="ALL", LineCode="1")
    ref_year = int(re.match(".+ (....) dollars", df.attrs['detail']['UnitOfMeasure']).group(1))
    return ref_year


def gen_full_df_index_for_PORT(geo_file, time_periods, line_list):
    # Make a data frame with an index that is the full set of Metro/NonMetro by state, BEA region, US. 
    # In the data, not all of these combinations have non-zero data, so we have to add at the end.'''
    st_fips = geo_file['GeoAgg1Fips'].str.slice(0,2).unique()
    area_fips = np.concatenate((st_fips, np.array(bea_regions)))
    port_parts_fips = ["998", "999"]
    
    # cartesian product https://stackoverflow.com/questions/11144513/)
    cp = pd.DataFrame(np.dstack(np.meshgrid(area_fips, port_parts_fips)).reshape(-1, 2), columns=["st", "part"])
    cp['GeoFips'] = (cp['st'] + cp['part']).astype("string")
    mi = pd.MultiIndex.from_product([cp['GeoFips'], time_periods, line_list])
    df_all_blanks = (pd.DataFrame(index=mi).reset_index()
                     .rename(columns={'GeoFips':'GeoAggFips', 'level_1':'TimePeriod'}))
    return df_all_blanks

### GDP Functions

def _gen_basic_pqs(df, time_series_id = ['LineCode','GeoFips'], extra_group_vars=[]):
    df = df.sort_values(extra_group_vars + time_series_id +["TimePeriod"])

    df["fixed_dollars"] = df.GDP_current_dollars / df.price

    def gen_lag(df, vname):
        # Assumes it's sorted by unit-vars and then time.
        return df.groupby(extra_group_vars + time_series_id)[vname].shift(1)
    
    df["p1q1"] = df.price * df.fixed_dollars
    df["p1q0"] = df.price * gen_lag(df, "fixed_dollars")
    df["p0q1"] = gen_lag(df, 'price') * df.fixed_dollars
    df["p0q0"] = gen_lag(df, 'price') * gen_lag(df, "fixed_dollars")
    return df


def gen_has_zero_ref(df_agg, gb, ref_year):
    # has zero nominals in the reference year
    # gb: list of grouping vars
    return (df_agg.assign(zero_ref=lambda x: ~x.GDP_current_dollars.isnull() & (x.GDP_current_dollars==0) & (x.TimePeriod==ref_year))
            .groupby(gb)['zero_ref'].transform('max'))

def gen_has_na_ref(df_agg, gb, ref_year):
    # has NA nominals in the reference year
    # gb: list of grouping vars
    return (df_agg.assign(na_ref=lambda x: x.GDP_current_dollars.isnull() & (x.TimePeriod==ref_year))
            .groupby(gb)['na_ref'].transform('max'))

def calc_agg_metrics_gdp(df, ref_year, time_series_id = ['LineCode','GeoFips'], details=False, leaf_industries=False):
    # df needs vars: <time_series_id> vars, TimePeriod, agg_unit, GDP_current_dollars, price
    # details can be True (all intermediates), False (just aggs)/'contrib' (those needed for contrib_pct_change)
    assert df.price.isnull().sum()==0, "Error: missing data for prices"
    assert (~df.GDP_current_dollars.isnull() & df.GDP_current_dollars<0).sum()==0, "Error: negative nominal GDP data"
    assert (df.price<0).sum()==0, "Error: negative prices"
    assert sum([df[v].isnull().sum() for v in time_series_id])==0, "Error: missing data in timeseries_id var"

    years = list(np.sort(df.TimePeriod.unique()))

    df_agg = (df.drop(columns=['price'] + time_series_id).groupby(["agg_unit", "TimePeriod"], as_index=True)
              .agg(sum_prop_na).reset_index())
    # If counties change for a geo agg, then aggregate P0Q1, P1Q0, P0Q0 will be null in the first period of the change.
    # Possibly could work around by (a) bringing fixed_dollar (Q1) into this function (have it at calling site), 
    # (b) calculate P0Q0 as lag(P1Q1) and Q0 as lag(fixed_dollar),
    # (c) calculate P1Q1 as P1Q1/Q1*Q0 and P0Q1 as P0Q1 as P0Q0/Q0*Q1
    # df_agg['price'] = df_agg['GDP_current_dollars'] / df_agg["fixed_dollars"] # Don't need

    # Fisher is geometric mean of paasche and laspeyres
    df_agg["fisher_q_rel"] = np.sqrt((df_agg.p1q1/df_agg.p1q0) * (df_agg.p0q1/df_agg.p0q0))
    df_agg = df_agg.sort_values(["agg_unit", "TimePeriod"])

    
    def genL(df, vname): #Lag
        # Assumes it's sorted by agg_unit and then time.
        return df.groupby('agg_unit')[vname].shift(1)
    def genF(df, vname): #Forward
        # Assumes it's sorted by agg_unit and then time.
        return df.groupby('agg_unit')[vname].shift(-1)

    # Desired reference period is set to 100 and the relatives are stitched on.
    # Please note that when on left of reference period, division is involved and on the right it is multiplication.
    df_agg['has_zero_ref'] = gen_has_zero_ref(df_agg, ["agg_unit"], ref_year)
    df_agg['has_na_ref'] = gen_has_na_ref(df_agg, ["agg_unit"], ref_year)
    df_agg['chain_type_q_index'] = np.nan
    df_agg.loc[(df_agg.TimePeriod==ref_year) & ~df_agg.GDP_current_dollars.isnull(), 'chain_type_q_index'] = 100
    for year in years:
        if year > ref_year:
            df_agg.loc[df_agg.TimePeriod==year, 'chain_type_q_index'] = \
                df_agg["fisher_q_rel"] * genL(df_agg, 'chain_type_q_index')
    import warnings
    with warnings.catch_warnings(): # suppress warnings here from dividing by 0. I manually verify results below
        warnings.simplefilter("ignore", RuntimeWarning)
        for year in years[::-1]:  # go backwards
            if year < ref_year:
                df_agg.loc[(df_agg.TimePeriod==year), 'chain_type_q_index'] = \
                    (genF(df_agg, 'chain_type_q_index') /  genF(df_agg, 'fisher_q_rel'))
    # Ensure if ref_year nominals=0, chain_type_q_index will always be missing.
    df_agg.loc[df_agg.has_zero_ref, 'chain_type_q_index'] = np.nan
    
    # Chained dollars are nothing but rescaled quantity index
    # You scaled the index to be 100 in reference period and then multiply by the nominal value of the 
    # aggregate in the reference period.
    ref_level_df = (df_agg.loc[df_agg.TimePeriod == ref_year,["agg_unit", "p1q1"]]
                    .rename(columns={"p1q1": "ref_level"}))
    df_agg = df_agg.merge(ref_level_df, on=["agg_unit"], how="left").sort_values(['agg_unit', 'TimePeriod'])
    df_agg['real_GDP_chained_dollars'] = df_agg['chain_type_q_index']/100*df_agg["ref_level"]
    if leaf_industries: # For leaf industries, we can always calculate chained dollars directly from nominals and prices regardless of quantity indexes.
        df_agg = df_agg.merge((df[["agg_unit", "TimePeriod", 'price']]
                               .groupby(["agg_unit", "TimePeriod"], as_index=True).agg("first").reset_index()), 
                               how="left")
        df_agg.loc[df_agg['real_GDP_chained_dollars'].isna(), 'real_GDP_chained_dollars'] = df_agg['GDP_current_dollars']/df_agg['price']
        df_agg = df_agg.drop(columns=['price'])

    if details=='contrib':
        contrib_detail_vars = ['p1q1', 'p1q0', 'p0q1', 'p0q0', 'fisher_q_rel']
        return df_agg[["agg_unit", "TimePeriod"] + ['GDP_current_dollars', 'chain_type_q_index', 'real_GDP_chained_dollars'] + 
                      contrib_detail_vars]
    
    if details==True:
        return df_agg

    df_agg = df_agg[["agg_unit", "TimePeriod"] + ['GDP_current_dollars', 'chain_type_q_index', 'real_GDP_chained_dollars']]
    return df_agg


def calc_lowest_contrib_pct_change(df, df_agg, time_series_id = ['LineCode','GeoFips'], details=False):
    # Required df vars: <time_series_id>, nominals, prices, TimePeriod, toward_agg_unit
    # Required df_agg vars: toward_agg_unit, TimeSeries, and the detail from calc_agg_metrics (detail='contrib')

    assert df.price.isnull().sum()==0, "Error: missing data for prices"

    # For contributions there is a concept of an "Anchor" (aggregate series whose growth you are decomposing) and 
    # a "Descendant" for the pieces that underly the aggregate. 
    df2 = df.merge((df_agg[["toward_agg_unit", "TimePeriod", "p1q1", "p1q0", "p0q1", "p0q0", "fisher_q_rel"]]
                    .rename(columns={"p1q1": "p1q1_agg", "p1q0": "p1q0_agg", "p0q1": "p0q1_agg", 
                                     "p0q0": "p0q0_agg", "fisher_q_rel": "fisher_q_rel_agg"})), 
                on=["toward_agg_unit", "TimePeriod"], how="left")
    df2['contrib_pct_change'] = ((df2.p1q1_agg/df2.p0q0_agg*(df2.p0q1-df2.p0q0) + \
                                  df2.fisher_q_rel_agg*(df2.p1q1-df2.p1q0))/
                                (df2.p1q1_agg + df2.fisher_q_rel_agg*df2.p1q0_agg) *100)
    #equivalent from BEA NIPA methodology ch4
    #df2['contrib_pct_change2'] = (((df2.p1q1-df2.p1q0) + df2.fisher_q_rel_agg*(df2.p0q1-df2.p0q0))/
    #                              (df2.p1q0_agg + df2.fisher_q_rel_agg*df2.p0q0_agg) *100) 
    
    if details:
        return df2

    df2 = df2[["toward_agg_unit"] + time_series_id + ["TimePeriod", "contrib_pct_change", 'used_for']]
    return df2


def round_gdp_vars(df_agg, industry_vname, top_industry):
    df_agg = df_agg.rename(columns={'GDP_current_dollars':'nominal_unr', 'chain_type_q_index':'chain_type_q_index_unr', 
                                    'real_GDP_chained_dollars':'chained_dollars_unr', 
                                    'contrib_pct_change':'contrib_pct_change_unr'})
    df_agg['GDP_current_dollars'] = df_agg['nominal_unr'] # integers stay integers
    df_agg['chain_type_q_index'] = round_std(df_agg['chain_type_q_index_unr'], decimals=3)
    df_agg['real_GDP_chained_dollars'] = round_std(df_agg['chained_dollars_unr'], decimals=0)
    df_agg['contrib_pct_change'] = np.where(df_agg[industry_vname]==top_industry, 
                                            round_std(df_agg['contrib_pct_change_unr'], decimals=1),
                                            round_std(df_agg['contrib_pct_change_unr'], decimals=2))
    return df_agg

def calc_used_for(df, ind_parent_leaves_map, parent_child_map, ind_leaves, parent_descendants, geo_vname, industry_vname, time_series_id, verbosity=1):
    # For each time-period x county, identify nodes that are parents and all children are non-missing in this and previous period
    if verbosity: print("Calculating county-year non-missing data to use.")
    if len(ind_parent_leaves_map)>0:
        # A Get a list of parents, where descendants are always earlier than ancestors.
        def subtree_cmp_core(a,b):
            if set(ind_parent_leaves_map[a])==set(ind_parent_leaves_map[b]):
                return 0
            if set(ind_parent_leaves_map[a]).issubset(set(ind_parent_leaves_map[b])):
                return -1
            if set(ind_parent_leaves_map[b]).issubset(set(ind_parent_leaves_map[a])):
                return 1
            return None
        
        #simple and full sort, sorted(sorted_parents, key=cmp_to_key(subtree_cmp)) and simple sorting doesn't work since I just have a partial ordering
        sorted_parents = list(ind_parent_leaves_map.keys())
        need_to_recheck=True
        while need_to_recheck:
            need_to_recheck=False
            for i in range(len(sorted_parents)-1):
                for j in range(i+1, len(sorted_parents)):
                    a = sorted_parents[i]
                    b = sorted_parents[j]
                    if subtree_cmp_core(a,b)==1: # swap
                        #print(f"Swapping {a} and {b} in sorted_parents")
                        sorted_parents[i], sorted_parents[j] = b, a
                        need_to_recheck=True
                        break

        if verbosity>1: print(("sorted_parents", sorted_parents))
        
        # B For each parent_ind, identify for each geo x Timeperiod if that node has all fine children industries
        markings = []
        for parent_ind in sorted_parents:
            child_inds = parent_child_map[parent_ind]
            df_p_child = df[df[industry_vname].isin(child_inds)].copy()
            df_p_child['L_GDP_current_dollars'] = df.groupby(time_series_id)['GDP_current_dollars'].shift(1)
            df_c_agg = (df_p_child.groupby([geo_vname, 'TimePeriod'])[[geo_vname, 'TimePeriod', 'GDP_current_dollars', 'L_GDP_current_dollars']]
                        .apply(lambda x: x['GDP_current_dollars'].isna().sum() + x['L_GDP_current_dollars'].isna().sum()==0))
            df_c_agg = df_c_agg.reset_index(name='parent_and_fine_children')
            df_c_agg[industry_vname] = parent_ind
            markings.append(df_c_agg)
        marking = pd.concat(markings, ignore_index=True)
        for v in time_series_id:
            marking[v] = marking[v].astype(df[v].dtype) # make sure to match types for merge
        df = df.merge(marking, on=time_series_id + ['TimePeriod'], how='left')
        with pd.option_context("future.no_silent_downcasting", True):
            df['parent_and_fine_children'] = df['parent_and_fine_children'].fillna(False).astype(bool)
        df = df.rename(columns={'parent_and_fine_children':'use_children'})

        # Also can't use nodes that don't have prices
        missing_prices = df[df.price.isna()]['LineCode'].drop_duplicates().tolist() #Assume this is static across time
        df.loc[df.LineCode.isin(missing_prices), 'use_children'] = True
    else:
        sorted_parents = []
        df['use_children'] = False
    if verbosity>1: print("df:\n", df.sort_values(['GeoFips','LineCode']))
    
    # C define 'used_for' for main real gdp calculations
    # - if targetting leaf or parent with some missing children: use that industry directly (sometimes that'll be missing)
    # - else (is use_children): concatenate what to use for the children 
    # do this by building up from the bottom
    df['used_for'] = pd.Series(",", index=df.index, dtype="string") # make sure to put "," before/after each industry so when we search we can find "1" w/o getting "11"
    for ind in ind_leaves:
        df.loc[df[industry_vname]==ind, 'used_for'] = f",{ind},"
    for parent_ind in sorted_parents:
        mask = ~df['use_children'] & (df[industry_vname]==parent_ind)
        df.loc[mask, 'used_for'] = f",{parent_ind},"
    df['children_to_use'] = df['used_for'] # so far these are the same
    #now deal with the use_children==True
    def comb_str_lists(x):
        inds = set([a_i for a_i in ''.join(x).split(',') if len(a_i)>0]) #len() check is to remove ''
        comb_str = ''.join([f"{x_i}," for x_i in inds])
        if len(comb_str)>0:
            comb_str = "," + comb_str # bookend with comma so we can search for ",1," w/o getting ",11,"
        return comb_str
    if verbosity>1: print("df:\n", df.sort_values(['GeoFips','LineCode']))
    for parent_ind in sorted_parents:
        p_mask = df['use_children'] & (df[industry_vname]==parent_ind)
        if p_mask.sum()==0:
            continue

        # calculate node's children_to_use from the children
        df_p = df[p_mask]
        df_p_child = (df[df[industry_vname].isin(parent_child_map[parent_ind])]
                .merge(df_p[[geo_vname, 'TimePeriod']], how="inner", on=[geo_vname, 'TimePeriod'])) #filter to those in df_p
        df_c_agg = (df_p_child.groupby([geo_vname, 'TimePeriod'])['children_to_use']
                    .apply(lambda x: comb_str_lists(x)).reset_index()) #agg over linecodes
        df_c_agg[industry_vname] = pd.Series(parent_ind, index=df_c_agg.index, dtype="string")
        df = df.merge(df_c_agg, how='left', on=[industry_vname, geo_vname, 'TimePeriod'], suffixes=("", "_2")) 
        df.loc[p_mask, 'children_to_use'] = df['children_to_use_2']
        df = df.drop(columns=['children_to_use_2'])
        
        #add parent_ind to used_for variable for those listed in children_to_use
        df_p_desc = (df[df[industry_vname].isin(parent_descendants[parent_ind])]
                .merge(df[p_mask][[geo_vname, 'TimePeriod', 'children_to_use']].rename(columns={'children_to_use':'p_children_to_use'}), how="inner", on=[geo_vname, 'TimePeriod'])) #filter to those in df_p
        df_p_desc.loc[df_p_desc.apply(lambda x: ("," + x[industry_vname]+",") in x['p_children_to_use'], axis=1), 'used_for'] += f"{parent_ind},"
        df = df.merge(df_p_desc[[geo_vname, 'TimePeriod', industry_vname, 'used_for']], 
                      how="left", on=[geo_vname, 'TimePeriod', industry_vname], suffixes=("", "_2"))
        df.loc[~df['used_for_2'].isna(), 'used_for'] = df['used_for_2']
        if verbosity>1: print(f"df (after parent_ind={parent_ind}):\n", df.sort_values(['GeoFips','LineCode']))
        df = df.drop(columns=['used_for_2'])
    df.loc[df['used_for']==",",'used_for'] = "" # if nothing was used, make it blank instead of ","
    if verbosity>1: print("df:\n", df.sort_values(['GeoFips','LineCode']))
    df = df.drop(columns=['children_to_use', 'use_children'])
    return df

def do_main_calcs_gdp(df, dfrp, ref_year, n_geo_levels=1, 
                      geo_vname='GeoFips', industry_vname="LineCode", top_industry="1", 
                      rounding="none", real_inds_used="lowest_available", verbosity=0):
    # Do the main calculations

    # Args:
    #     rounding: Options: "none", "round", "both" (extra variable names ending in "_unr")
    #     top_industry: The industry aggregate that we calculate contributions toward.
    # contrib_pct_change when TimePeriod is min (assuming this will be filtered out by caller)

    # Returns:
    #     _type_: _description_
    nominal_ind_used="highest" # only option currently supported. more expected accurate than leaf as less rounding error.
    if real_inds_used not in ['leaf', 'lowest_available']:
        raise NotImplementedError("do_main_calcs_gdp currently only supports real_inds_used='leaf' or 'lowest_available'")
    if isinstance(top_industry, int):
        top_industry = str(top_industry)
    time_series_id = [geo_vname, industry_vname]
    
    # Get other views of the industry hierarchy
    if dfrp is None or dfrp.shape[0]==0:
        ind_parent_leaves_map={}
        ind_leaves = df[industry_vname].unique().tolist()
        parent_child_map = {}
        parent_descendants = {}
    else:
        dfrp['leaf'] = ~dfrp.Child.isin(dfrp.Parent)
        ind_parent_leaves_map, parent_descendants = build_descendants_map(dfrp)
        ind_leaves = ind_parent_leaves_map[top_industry]
        assert set(dfrp.Child[dfrp.leaf].unique()) == set(ind_parent_leaves_map[top_industry]), "industry categorization isnt' consistent"
        parent_child_map = {parent: list(set(dfrp.Child[dfrp.Parent==parent].values)) 
                            for parent in ind_parent_leaves_map.keys()}
        if verbosity>1:
            print(("parent_child_map:", parent_child_map))
            print(("ind_parent_leaves_map: ", ind_parent_leaves_map))

    # 1. Get some basic calculations used commonly
    df = _gen_basic_pqs(df, time_series_id=time_series_id) # this sorts df
    
    #if verbosity:
    #    print("Calculating main aggregates")
    
    # 2. Calculate aggregate nominals, quantity index for real GDP, and real GDP in chained dollars
    # 2.1 Define of potential aggregations
    # Assume components are (counties, leaf industries)
    # You can aggregate along 2 dimensions: 
    #  - industries (2 options): keep the same, or agg leaves to agg industry)
    #  - geographies (1-3 types): GeoAgg  (or for some hierarchies GeoAgg2, GeoAgg3)
    # We don't need to calculate anything where counties stay (and just aggregate industries) as 
    #   those are already calculated

    geo_level_vars = [f"GeoAgg{l+1}" for l in range(n_geo_levels)]
    calc_sets = [] # Each element will be (parent_ind, geo agg var, ind_leaves)
    for geo_level_i in range(n_geo_levels):
        # Aggregate (Industries: keep each the same (sep. groups), Geography: to geo_level_i) for leaf industries
        calc_sets += [("-", f"GeoAgg{geo_level_i+1}", ind_leaves)]

        # For each agg industry: Aggregate (Industries: to agg industry, Geography: to geo_level_i) for 
        # agg industries leaves
        calc_sets += [(parent, f"GeoAgg{geo_level_i+1}", leaves) for parent, leaves in ind_parent_leaves_map.items()]

    # 2.2 For each parent industry, define for each county-year appropriate industry leaves
    df = calc_used_for(df, ind_parent_leaves_map, parent_child_map, ind_leaves, parent_descendants, geo_vname, industry_vname, time_series_id,verbosity=verbosity)
    
    # 2.3 do the calculations
    if verbosity>1:
        print("Main calculations")
    details_for_contrib = {}
    initial_aggregations = []
    for parent_ind, geo_agg_var, component_ind in tqdm.tqdm(calc_sets, disable=(verbosity!=1), 
                                                            desc="Main calculations. Parent industries"):
        if verbosity>1:
            print(f"parent industries={parent_ind}")
        # First calculate reals (and nominals) using the real-level and overwrite with nominal level if different
        if real_inds_used=="leaf" or parent_ind=="-":
            df_group_components = df[df[industry_vname].isin(component_ind)].copy()
        else:
            df_group_components = df[df['used_for'].str.contains(f",{parent_ind},")].copy()
        if verbosity>1: print("Main calcs components 1:\n", df_group_components)

        agg_ind = (df_group_components[industry_vname] if parent_ind=="-" else parent_ind)
        df_group_components['agg_unit'] = df_group_components[geo_agg_var+'Fips'] + ":" + agg_ind
        df_group_components = df_group_components[["agg_unit", "TimePeriod"] + time_series_id + 
                                                  ['GDP_current_dollars', 'price', "p1q1", "p1q0", "p0q1", "p0q0"]]
        # if the top industry, get extra details for contributions
        is_top = (parent_ind==top_industry) or (len(ind_parent_leaves_map)==0)  #latter means "-" is top_industry
        do_details = 'contrib' if is_top else False
        df_agg = calc_agg_metrics_gdp(df_group_components, ref_year, time_series_id=time_series_id, 
                                      details=do_details, leaf_industries=(parent_ind=="-"))
        # calc_agg_metrics_gdp incidentally calculates aggregate nominal GDP, but we might want to do that a different way.
        if parent_ind!="-" and nominal_ind_used=="highest": 
            df_group_highest = df[df[industry_vname]==parent_ind].copy()
            if verbosity>1: print("Main calcs group components 1a:\n", df_group_highest)
            df_group_highest['agg_unit'] = df_group_highest[geo_agg_var+'Fips'] + ":" + agg_ind
            
            df_agg_highest = (df_group_highest.groupby(["agg_unit", "TimePeriod"])[['GDP_current_dollars']]
                                        .agg(sum_prop_na).reset_index())
            df_agg = df_agg.merge(df_agg_highest, on=['agg_unit', 'TimePeriod'], how='left', suffixes=("", "_2"))
            df_agg.loc[~df_agg[f"GDP_current_dollars_2"].isna(), 'GDP_current_dollars'] = df_agg['GDP_current_dollars_2']
            df_agg = df_agg.drop(columns=['GDP_current_dollars_2'])

        if is_top:
            details_for_contrib[geo_agg_var] = df_agg.copy()
            df_agg = df_agg.drop(columns=["p1q1", "p1q0", "p0q1", "p0q0", "fisher_q_rel"])
        df_agg[['GeoAggFips',industry_vname]] = df_agg['agg_unit'].str.split(":", expand=True)
        df_agg = df_agg.merge((df_group_components.groupby(["agg_unit", "TimePeriod"])[['price']]
                            .agg('count')
                            .reset_index().rename(columns={'price':'child_count'})), 
                            how='left', on=['agg_unit', 'TimePeriod'])

        initial_aggregations.append(df_agg.drop(columns=['agg_unit']))
    df_agg0 = pd.concat(initial_aggregations , ignore_index=True)    

    # 3. calculate contributions: For GeoAgg to each GeoAgg total, (and for GeoAgg2 up to the GeoAgg2 total), ...
    if verbosity>1:
        print("Calculating contributions")
    calc_sets_contrib = ind_parent_leaves_map.copy()
    calc_sets_contrib["-"] = ind_leaves #Geo aggregate of the leaf industries
    agg_contrib_lists = []
    for geo_agg_var in tqdm.tqdm(geo_level_vars, desc="Calculating contributions. Levels", disable=(verbosity!=1)):
        if verbosity>1:
            print(f"Contributions: Aggregation level={geo_agg_var[-1]}")
        # A. calculate lowest level contributions (county by leaf industry)
        # We filter later, so could do all, but might as well filter somewhat
        if real_inds_used=="leaf":
            df_components = df[df[industry_vname].isin(ind_leaves)].copy()
        else:
            df_components = df[df['used_for'].str.contains(",")].copy()
        if verbosity>1: print("Contributions group components:\n", df_components)
        # toward_agg_unit here is for calc_lowest_contrib_pct_change, 
        # but we're not summing like we are below, so call slightly differently
        df_components['toward_agg_unit'] = df[geo_agg_var+'Fips']  + ":" + top_industry
        details_agg = details_for_contrib[geo_agg_var].rename(columns={'agg_unit':'toward_agg_unit'})
        df_components = df_components[["toward_agg_unit", 'TimePeriod'] + time_series_id + ['GDP_current_dollars', 
                                         'price', "p1q1", "p1q0", "p0q1", "p0q0"] + ['used_for']]
        #if verbosity>1: print("Contributions df_components:\n", df_components)
        df_contrib_lowest = calc_lowest_contrib_pct_change(df_components, details_agg, time_series_id=time_series_id)
        df_contrib_lowest['GeoAggFips'] = df_contrib_lowest['toward_agg_unit'].str.split(":", expand=True)[0]  # 1=industry_vname
        if verbosity>1:  print("Contributions df_contrib_lowest:\n", df_contrib_lowest)
        
        # B. Aggregate
        for parent_ind, component_ind in calc_sets_contrib.items():
            if verbosity>1: print(f"Contributions: geo_agg_var={geo_agg_var}, parent={parent_ind}")

            if real_inds_used=="leaf" or parent_ind=="-":
                df_group_components = df_contrib_lowest[df_contrib_lowest[industry_vname].isin(component_ind)].copy()
            else:
                df_group_components = (df_contrib_lowest[df_contrib_lowest['used_for'].str.contains(f",{parent_ind},")]
                                       .copy())
            if verbosity>1: print("Contributions df_group_components:\n", df_group_components)

            if parent_ind!="-":
                df_group_components[industry_vname] = pd.Series(parent_ind, index=df_group_components.index, dtype='string') #make it a string dtype rather than object
            agg_contribs = (df_group_components.groupby([industry_vname, "GeoAggFips", 'TimePeriod'])
                            [["contrib_pct_change"]]
                            .agg(sum_prop_na).reset_index())
            if verbosity>1: print("Contributions agg_contribs:\n", agg_contribs)
            agg_contrib_lists.append(agg_contribs)
    df_agg_contrib = pd.concat(agg_contrib_lists , ignore_index=True)

    df_agg = df_agg0.merge(df_agg_contrib, how='left', indicator=True, on=[industry_vname, 'GeoAggFips', 'TimePeriod'])
    assert (df_agg['_merge']=="right_only").sum()==0, "Error: bad merge with contributions estimates."
    df_agg = df_agg.drop(columns=['_merge'])

        
    # 4. Round
    if rounding in ["round", "both"]:
        df_agg = round_gdp_vars(df_agg, industry_vname, top_industry)
        if rounding=="round":
            df_agg = df_agg.drop(columns=gdp_calced_names_unr)
    
    initial_cols = ['GeoAggFips', industry_vname, 'TimePeriod']
    df_agg = df_agg[initial_cols + [c for c in df_agg.columns if c not in initial_cols]].sort_values(initial_cols)
    return df_agg


def combine_nominals_prices(nominals, prices, industries, verbosity=0):
        
    nominals["LineCode"] = nominals['Code'].str.slice(len("CAGDP2")+1)

    # 2. Combine data
    # 2.1 Bring in bundled industry meta-data
    pkg_path = pathlib.Path(__file__).parent
    ind_rea_va_xw = pd.read_csv(pkg_path.joinpath("metadata/ind_rea_va_xw.csv"), dtype="string")

    ind_code_name_xw = (nominals[['LineCode']].drop_duplicates()
                        .merge(industries.rename(columns={"Key": "LineCode", "Desc": "REA Description"}), 
                               on="LineCode", how="left"))
    ind_code_name_xw['REA Description'] = (ind_code_name_xw['REA Description']
                                           .str.slice(39) # remove inital prefix about the table
                                           .str.replace(r" \([0-9].+\)", "", regex=True)) # remove ending footnotes

    ind_xw = ind_code_name_xw.merge(ind_rea_va_xw[['REA Description', 'VA Description']], 
                                    on="REA Description", how="left")

    df = nominals.merge(ind_xw, on="LineCode", how="inner")
    if nominals.shape[0]!=df.shape[0]:
        temp = nominals.merge(ind_xw, on="LineCode", how="outer", indicator=True)
        print("bad industry merge. left, right: ", temp[temp._merge!="both"].drop_duplicates())
        assert nominals.shape[0]==df.shape[0], "Error: Bad merge when getting industries"

    # 2.2 Bring in Prices
    # cleanup industry practice of footnotes in names
    prices['IndustrYDescription'] = prices.IndustrYDescription.str.replace(r"<sup>.*</sup>", "", regex=True) 
    df = (df.rename(columns={"DataValue": 'GDP_current_dollars'})
           .merge(prices[["IndustrYDescription", "Year", "DataValue"]]
                  .rename(columns={"IndustrYDescription": "VA Description", 
                                   "Year": "TimePeriod", "DataValue": 'price'}), 
                  on=["TimePeriod", "VA Description"], how="left"))
    if verbosity>1:
        print("Noting that we don't have prices for:")
        print(df[df.price.isna()][['LineCode', 'VA Description', 'TimePeriod']].drop_duplicates())
    # Note some of the addendum lines will be missing in prices
    df['TimePeriod'] = df['TimePeriod'].astype(int)

    return df, ind_rea_va_xw

def do_merging_and_calculations_gdp(nominals:pd.DataFrame, prices:pd.DataFrame, industries:pd.DataFrame, 
                                    geo_agg: Union[str,pathlib.Path,pd.DataFrame], ref_year:int, start_year:int,
                                    source_has_all_counties:bool=True, custom_df_is_port:bool=False, 
                                    rounding="none", ret_na_0withnote:bool=False, limit_to_start_year:bool=True, 
                                    append_pre2024_CT_defs:bool=True,
                                    verbosity:bool=False) -> dict[str, pd.DataFrame]:
    """Once we have the source data, do data manipulation and calculations to get final GDP data.

    Args:
        nominals (pd.DataFrame): nominal GDP data frame (from pull_source_data_gdp())
        prices (pd.DataFrame): industry price indexes GDP data frame (from pull_source_data_gdp())
        industries (pd.DataFrame): industry metadata  (from pull_source_data_gdp())
        geo_agg (Union[str,pathlib.Path,pd.DataFrame]): key to bundled file, Path/str to csv, or dataframe
        ref_year (int): Reference year for index series (from pull_source_data_gdp())
        start_year (int): First year for desired results data
        source_has_all_counties (bool, optional): Does the source data contain all counties. If not, only do 1 level of aggregation.
        custom_df_is_port (bool, optional): If passing in a custom geo_agg data frame, treat as PORT 
          (return all possible product of portions). Defaults to False.
        rounding (str, optional): Options: "none", "round" (to BEA standard for published tables), 
          "both" (extra variable names ending in "_unr"). Defaults to "none".
        ret_na_0withnote (bool, optional): Whether to return a version of the data with NAs replaced by 0s and a note added about this. Defaults to False.
        limit_to_start_year (bool, optional): Limits data to starting at start_year. Defaults to True.
        append_pre2024_CT_defs (bool, optional): If True, append bundled pre-2024 CT definitions to geo_agg if MSA or PORT. Defaults to True.
        verbosity (bool, optional): Level of verbosity (high is more). Defaults to False.

    Returns:
        dict[str -> pd.DataFrame]: Mapping of GDP table names to data frames.
    """    
    assert nominals['DataValue'].isnull().sum()==0, "Source data (nominals) has nulls in DataValue"
    assert (nominals.GeoFips=="").sum()==0, "Error: There are counties without GeoFips"
    # Suppression approach: Tracking has_suppressed through calculations is too hard. 
    # So do calculations twice, once filling NAs with 0s, and once with the original NAs and then 
    # seeing where the NAs end up.
    import json
    pkg_path = pathlib.Path(__file__).parent
    
    nominals_meta_data = nominals.groupby(['Code','TimePeriod'])[['CL_UNIT', 'UNIT_MULT']].first().reset_index()
    nominals_meta_data['TimePeriod'] = nominals_meta_data['TimePeriod'].astype(int)
    nominals_meta_data["LineCode"] = nominals_meta_data['Code'].str.slice(len("CAGDP2")+1).astype(int)
    nominals_meta_data = nominals_meta_data.drop(columns=['Code'])
    df_notes = nominals.attrs.get('detail', {}).get('Notes',[])
    
    # 1. basic source data cleanup
    nominals['has_suppressed'] =  has_suppression(nominals)
    nominals['has_NA'] = has_other_null(nominals) # lump all as NA
    nominals = nominals.drop(columns=['CL_UNIT', 'UNIT_MULT', 'NoteRef'])

    # 2. Integrate data
    df, ind_rea_va_xw = combine_nominals_prices(nominals, prices, industries, verbosity=verbosity)
    
    dfrp = pd.read_csv(pkg_path.joinpath("metadata/dfrp.csv"), dtype='string').drop(columns=['Label'])
    
    # Bring in Geo file to get GeoAgg
    geo_file = get_geo_file(geo_agg, append_pre2024_CT_defs)
    if not source_has_all_counties:
        geo_file = geo_file[["County name", 'GeoFips', "GeoAgg1_name", 'GeoAgg1Fips']]
    n_geo_levels = int(geo_file.shape[1]/2)-1
    #n_obs_prev = df.shape[0]
    df = df.merge(geo_file.drop(columns=["GeoAgg1_name", "County name"]), on="GeoFips", how="inner")
    assert pd.isnull(df['GeoAgg1Fips']).sum()==0, "Error: There are counties without GeoAgg1Fips"
    #print(f"Going from {n_obs_prev} to {df.shape[0]} rows after merging with geo file")
    
    # 3. Calculate
    # 3.1 Run main calculation
    # Always run once with propagating suppressions
    #if verbosity:
    #    print("Running main calculations")
    df.loc[df['has_suppressed'] | df['has_NA'], 'GDP_current_dollars'] = np.nan
    df_agg = do_main_calcs_gdp(df, dfrp, ref_year, n_geo_levels, 
                                 rounding=rounding, verbosity=verbosity)
    gdp_calced_names = gdp_calced_names_primary + (gdp_calced_names_unr if rounding=="both" else [])


    # 3.2 For PORT, ensure all combinations of State/Region x PORT parts exist
    if (isinstance(geo_agg, str) and geo_agg=="PORT" and source_has_all_counties) or custom_df_is_port:
        # PORT return 0s for empty state/Region x PORT geographies
        df_all_blanks = gen_full_df_index_for_PORT(geo_file, df['TimePeriod'].unique(), 
                                                   industries.rename(columns={'Key':'LineCode'})['LineCode'])
        df_agg = df_all_blanks.merge(df_agg, on=['GeoAggFips', 'LineCode', 'TimePeriod'], how='outer', indicator=True)

        df_agg.loc[df_agg['_merge']=='left_only', gdp_calced_names + ["child_count"]] = 0
        df_agg["child_count"] = df_agg["child_count"].astype(int)
        df_agg = df_agg.drop(columns=['_merge'])

    df_agg['LineCode'] = df_agg['LineCode'].astype(int) # so we can sort

    #Bring in GeoName
    df_agg = df_agg.rename(columns={"GeoAggFips":"GeoFips"}).merge(get_fips_name_mapping(geo_file), 
                                                                   on='GeoFips', how='left')

    # Add indented industry label
    ind_rea_va_xw['LineCode'] = ind_rea_va_xw['REA LineCode'].astype(int)
    df_agg = (df_agg.merge(ind_rea_va_xw[['LineCode', 'REA Description indent']]
                           .rename(columns={'REA Description indent':'Description'}), 
                           how='left', on='LineCode'))
    
    index_v = ['GeoFips', "GeoName", 'LineCode', 'Description', 'TimePeriod']
    df_agg = df_agg.sort_values(index_v)
        
    # Remove additional years acquired necessary for previous calculations
    if limit_to_start_year:
        df_agg = df_agg[df_agg.TimePeriod>=start_year].reset_index(drop=True)
    
    # 4. Separate the combined dataframe into multiple dataframes for each Regional table
    ret_dfs = {}
    for gdp_calced_name, tablename in gdp_variable_table_mapping.items():
        extra_calc_v = [f'{gdp_calced_name}_unr'] if rounding=="both" else []
        cols_to_select = index_v + [gdp_calced_name] + extra_calc_v # skip 'child_count'
        cols_rename = {gdp_calced_name:'DataValue'}
        df_g = df_agg[cols_to_select].rename(columns=cols_rename)
        if rounding=="both":
            df_g = df_g.rename(columns={f'{gdp_calced_name}_unr':'DataValue_unr'})

        if gdp_calced_name=='GDP_current_dollars':
            df_g = df_g.merge(nominals_meta_data, on=['LineCode', 'TimePeriod'], how='left')
            # "Thousands of dollars", 3
        elif gdp_calced_name=='chain_type_q_index':
            df_g['CL_UNIT'] = "Quantity index"
            df_g['UNIT_MULT'] = 0
        elif gdp_calced_name=='real_GDP_chained_dollars':
            df_g['CL_UNIT'] = nominals_meta_data.loc[0, 'CL_UNIT']
            df_g['UNIT_MULT'] = nominals_meta_data.loc[0, 'UNIT_MULT']
        else: #"contrib_pct_change"
            df_g['CL_UNIT'] = "Percent change"
            df_g['UNIT_MULT'] = 0

        if ret_na_0withnote:
            # Convert missing to 0 but with an (NA) noteref
            df_g['NoteRef'] = pd.Series("", index=df_g.index, dtype='string') #make it a string dtype rather than object
            df_g.loc[df_g['DataValue'].isnull(), 'NoteRef'] = "(NA)"
            df_g['DataValue'] = df_g['DataValue'].fillna(0)
            if gdp_calced_name in ['GDP_current_dollars', 'real_GDP_chained_dollars']:
                df_g['DataValue'] = df_g['DataValue'].astype(np.int64)
        
        df_g['Code'] = f'{tablename}-' + df_g['LineCode'].astype(str)

        df_g.attrs['detail'] = {'Notes': df_notes}

        ret_dfs[tablename] = df_g
    # CAGDP1 is special: it combines 3 lines from other tables
    df_g = pd.concat([ret_dfs['CAGDP9'][ret_dfs['CAGDP9'].LineCode==1].assign(LineCode=1), 
                                   ret_dfs['CAGDP8'][ret_dfs['CAGDP8'].LineCode==1].assign(LineCode=2),
                                   ret_dfs['CAGDP2'][ret_dfs['CAGDP2'].LineCode==1].assign(LineCode=3)], 
                                   axis=0, ignore_index=True)
    df_g['Code'] = 'CAGDP1-' + df_g['LineCode'].astype(str)
    df_g.attrs['detail'] = {'Notes': df_notes}
    ret_dfs['CAGDP1'] = df_g

    return ret_dfs

def get_counties_for_geoagg(geofips_filter, geo_agg, append_pre2024_CT_defs=True):
    geo_df = get_geo_file(geo_agg, append_pre2024_CT_defs)
    mask = geo_df.GeoAgg1Fips==geofips_filter
    assert mask.sum()>0, f"Error: {geofips_filter} not found in provided type of geographic aggregate."
    return geo_df[mask]['GeoFips'].unique().tolist()


def pull_source_data_gdp(start_year: int, last_year:int, bea_key:str, geofips_filter:Optional[str]=None, 
                         geo_agg:Optional[Union[str,pathlib.Path,pd.DataFrame]]=None, append_pre2024_CT_defs:bool=True, 
                         verbosity:int=1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Pull the necessary source data from the BEA API for calculating GDP aggregates.
    It will remove new CT counties from pre-2024 and old CT counties from 2024 onward since those are not consistent across years.

    Args:
        start_year (int): First year for desired results data
        last_year (int): Last year for returned data
        bea_key (str): BEA API key
        geofips_filter (Optional[str]): a fips code of a geographic aggregate to filter data pulled. Must be a first-level aggregate (MSA or PORT).
        geo_agg (Optional[Union[str,pathlib.Path,pd.DataFrame]]): defaults to None. If using geofips_filter, must be a key to bundled file, Path/str to csv, or dataframe.
        append_pre2024_CT_defs (bool, optional): If True, append bundled pre-2024 CT definitions to geo_agg if MSA or PORT. Defaults to True.
        verbosity (int, optional): Level of verbosity (high is more). Defaults to 1 (show progress).

    Returns:
        tuple: nominals (county level), chained_dollrs (county level), prices, industries, ref_year
    """    
    ref_year=get_current_ref_year(bea_key)
    if start_year>cagdp_start_year:
        pull_start_year = min(start_year-1, ref_year)
    else:
        pull_start_year = start_year
    if pull_start_year<cagdp_start_year: #prices data starts in 1997
        raise ValueError(f"Data prior to {cagdp_start_year} not supported for GDP calculations.")
    if geofips_filter is None:
        geofips = 'COUNTY'
    else:
        geofips = ','.join(get_counties_for_geoagg(geofips_filter, geo_agg, append_pre2024_CT_defs=append_pre2024_CT_defs))

    nominals, industries = pull_regional_data("CAGDP2", geofips, pull_start_year, last_year, bea_key, 
                                              verbosity=verbosity)
    nominals_tp = nominals['TimePeriod'].astype(int)
    nominals = nominals[((nominals_tp<CT_new_start) & ~nominals['GeoFips'].isin(new_CT_cnty_fips)) | ((nominals_tp>=CT_new_start) & ~nominals['GeoFips'].isin(old_CT_cnty_fips))]

    years = ",".join(str(l) for l in list(range(pull_start_year, last_year+1)))
    # Get Chain-Type Price Indexes for Value Added by Industry (A) (Q)
    prices = beaapi.get_data(bea_key, 'GDPbyIndustry', TableID="11", Frequency='A', year=years, Industry="ALL") 

    return nominals, prices, industries, ref_year

def pull_merge_calc_gdp(start_year: int, last_year: int, geo_agg: Union[str,pathlib.Path,pd.DataFrame], 
                        bea_key: str, geofips_filter:Optional[str]=None, custom_df_is_port:bool=False, 
                        rounding:str="none", append_pre2024_CT_defs:bool=True, verbosity:int=1) -> dict[str, pd.DataFrame]:
    """Pulls source data from BEA API, merges, and does calculations to calculate GDP aggregate estimates.

    Args:
        start_year (int): First year for returned data
        last_year (int): Last year for returned data
        geo_agg (Union[str,pathlib.Path,pd.DataFrame]): key (MSA or PORT) to bundled file, Path/str to csv, or dataframe
        bea_key (str): BEA API key
        geofips_filter (Optional[str]): a fips code of a geographic aggregate to filter data pulled. Must be a first-level aggregate (MSA or PORT).
        custom_df_is_port (bool, option): If passing in a custom geo_agg data frame, treat as PORT 
          (return all possible product of portions). Defaults to False.
        rounding (str, optional): Options: "none", "round" (to BEA standard for published tables), 
          "both" (extra variable names ending in "_unr"). Defaults to "none".
        append_pre2024_CT_defs (bool, optional): If True, append bundled pre-2024 CT definitions to geo_agg if MSA or PORT. Defaults to True.
        verbosity (int, optional): Level of verbosity (high is more). Defaults to 1 (show progress).

    Returns:
        dict: dictionary of talename to dataframe
    """    
    nominals, prices, industries, ref_year = pull_source_data_gdp(start_year, last_year, bea_key, 
                                                                  geofips_filter=geofips_filter,
                                                                  geo_agg=geo_agg, 
                                                                  append_pre2024_CT_defs=append_pre2024_CT_defs,verbosity=verbosity)
    ret_dfs = do_merging_and_calculations_gdp(nominals, prices, industries, geo_agg, ref_year, start_year, 
                                              source_has_all_counties=(geofips_filter is None),
                                              custom_df_is_port=custom_df_is_port, rounding=rounding, 
                                              append_pre2024_CT_defs=append_pre2024_CT_defs, verbosity=verbosity)
    
    return ret_dfs


### PI functions

def pi_combine_cnty_geoagg_calc(df_agg, df=None, geofips_filter=None, geo_agg="MSA", bea_key=None, verbosity:int=1):
    # Try to work with both (a) user from main reget counties, and (b) they got df themselves (either source pull or histdata ext)
    if 'child_count' in df_agg.columns: #(b)
        df_agg = df_agg.drop(columns=['child_count'])
    if "GeoFips" in df_agg.columns: #(a)
        df_agg = df_agg.rename(columns={"GeoFips":"GeoAggFips"})
    if geofips_filter is None: #can only handle 1 geoagg
       geofips_filter = df_agg.GeoAggFips.iloc[0]
    if df is not None: # (b)
        if 'GeoAgg1Fips' in df.columns:
            df = df[df.GeoAgg1Fips==geofips_filter].copy()
        if 'GeoName' in df.columns:
            df['Name'] = df['GeoName'] + " DataValue"
        else:
            df['Name'] = df['GeoFips'] + " DataValue"
    else: # have to download
        start_year, last_year = df_agg.TimePeriod.min(), df_agg.TimePeriod.max()
        tablename = df_agg['Code'].iloc[0].split("-")[0]
        df, line_codes = pull_source_data_pi(start_year, last_year, bea_key, tablename, geofips_filter=geofips_filter,
                                                                  geo_agg=geo_agg, verbosity=verbosity)
        df = make_nullcomment_NA_if_available(df)
        df = df.drop(columns=['CL_UNIT', 'UNIT_MULT', 'NoteRef'])
        df["LineCode"] = df['Code'].str.slice(len(tablename)+1)
        df['LineCode'] = df['LineCode'].astype(int)
        df['TimePeriod'] = df['TimePeriod'].astype(int)
        
        df['Name'] = df['GeoName'] + " DataValue"
        df = df[['LineCode','TimePeriod', 'Name', 'DataValue']]

    df_s = df.pivot(index=['LineCode','TimePeriod'], columns='Name', values='DataValue').reset_index()

    df_agg = df_agg[df_agg.GeoAggFips==geofips_filter].drop(columns=['GeoAggFips'])
    if 'GeoName' in df_agg.columns:
        agg_name = df_agg['GeoName'].iloc[0]
        df_agg = df_agg.drop(columns=['GeoName'])
    else:
        agg_name = geofips_filter
    df_ret = df_s.merge(df_agg, on=['LineCode', 'TimePeriod'], how="inner")
              
    for c in ['DataValue']:
       df_ret = df_ret.rename(columns={c:f"{agg_name} {c}"})

    initial_cols = ['LineCode'] + (['Description'] if 'Description' in df_ret.columns else []) +['TimePeriod']
    df_ret = df_ret[initial_cols + [c for c in df_ret.columns if c not in initial_cols]]
    return df_ret

def do_main_calcs_pi(df, tablename, n_geo_levels, make0_div_0denominators=False):
    # Mostly Just sum
    df_aggs = []
    for geo_level_i in range(1, n_geo_levels+1):
        df_agg_i = (df.groupby(['TimePeriod', 'Code', f"GeoAgg{geo_level_i}Fips"])
                    .agg({'DataValue':sum_prop_na, "GeoName": "count", 'has_suppressed': "max", 'has_NA': "max"})
                    .rename(columns={'GeoName':"child_count"})
                    .reset_index())
        
        df_aggs.append(df_agg_i.rename(columns={f"GeoAgg{geo_level_i}Fips": "GeoAggFips"}))
    df_agg = pd.concat(df_aggs, axis=0, ignore_index=True)
    df_agg['DataValue'] = df_agg['DataValue'].astype('float64')

    # re-calculate per-capita lines
    if tablename in pi_tables_div_lines.keys():
        # If dividing rows, set a meaningful index so things are ensured to line up if non-rectangular
        df_agg_idx = df_agg.set_index(['GeoAggFips', 'Code', 'TimePeriod']).sort_index()
        for numerator_line, denominator_line, ratio_line, ratio_mult in pi_tables_div_lines.get(tablename, []):
            numerator = (df_agg[df_agg.Code==f"{tablename}-{numerator_line}"]
                         .set_index(['GeoAggFips', 'TimePeriod'])[["DataValue", 'has_suppressed', 'has_NA']])
            denominator = (df_agg[df_agg.Code==f"{tablename}-{denominator_line}"]
                           .set_index(['GeoAggFips', 'TimePeriod'])[["DataValue", 'has_suppressed', 'has_NA']])
            ratio = (numerator[["DataValue"]]/denominator[["DataValue"]])*ratio_mult
            if make0_div_0denominators:
                ratio[denominator["DataValue"]==0] = 0
                ratio = ratio.astype(np.int64)
            ratio['Code'] = f"{tablename}-{ratio_line}"
            ratio = ratio.reset_index().set_index(['GeoAggFips', 'Code', 'TimePeriod'])
            df_agg_idx.loc[(slice(None), f"{tablename}-{ratio_line}", slice(None)),"DataValue"] = ratio 
            
            for h in ["has_suppressed", "has_NA"]:
                ratio_has = numerator[[h]] | denominator[[h]]
                if h=="has_NA" and make0_div_0denominators:
                    ratio_has[denominator["DataValue"]==0] = True
                ratio_has['Code'] = f"{tablename}-{ratio_line}"
                ratio_has = ratio_has.reset_index().set_index(['GeoAggFips', 'Code', 'TimePeriod'])
                df_agg_idx.loc[(slice(None), f"{tablename}-{ratio_line}", slice(None)),h] = ratio_has 

        df_agg = df_agg_idx.reset_index()
    
    # Round
    # Most linecodes are just summing integers, so stays the same, but ratio lines will be decimals.
    df_agg['DataValue'] = round_std(df_agg['DataValue'])
    
    return df_agg

def do_merging_and_calculations_pi(df:pd.DataFrame, line_descs:pd.DataFrame, 
                                   geo_agg: Union[str,pathlib.Path,pd.DataFrame], tablename: str, 
                                   source_has_all_counties:bool=True, custom_df_is_port: bool=False, 
                                   ret_na_0withnote:bool=False, append_pre2024_CT_defs:bool=True, verbosity: int=1) -> pd.DataFrame:
    """Merges and does calculations to calculate Personal Income aggregate estimates.

    Args:
        df (pd.DataFrame): source table (from pull_source_data_pi())
        line_descs (pd.DataFrame): line code descriptions  (from pull_source_data_pi())
        geo_agg (Union[str,pathlib.Path,pd.DataFrame]): key to bundled file, Path/str to csv, or dataframe
        tablename (str): Table name (e.g., "CAINC1")
        source_has_all_counties (bool, optional): Does the source data contain all counties. If not, only do 1 level of aggregation.
        custom_df_is_port (bool, optional): If passing in a custom geo_agg data frame, treat as PORT 
          (return all possible product of portions). Defaults to False.
        append_pre2024_CT_defs (bool, optional): If True, append bundled pre-2024 CT definitions to geo_agg if MSA or PORT. Defaults to True.
        verbosity (int, optional): Level of verbosity (high is more). Defaults to 1 (show progress).

    Returns:
        pd.DataFrame: Dataframe at the aggregated level(s)
    """    
    assert df['DataValue'].isnull().sum()==0, "Error: Source data has nulls in DataValue" # See assume no missing values
    # Suppression approach: Track has_suppressed through calculations
    df_notes = df.attrs.get('detail', {}).get('Notes',[])

    # 1. basic source data cleanup
    df['TimePeriod'] = df['TimePeriod'].astype(int)
    meta_data = df.groupby(['Code','TimePeriod'])[['CL_UNIT', 'UNIT_MULT']].first().reset_index()
    df['has_suppressed'] = has_suppression(df)
    df['has_NA'] = has_other_null(df) # lump all as NA
    df = df.drop(columns=['CL_UNIT', 'UNIT_MULT', 'NoteRef'])
        
    # 2. Combine data
    # Bring in Geo file to get GeoAgg
    geo_file = get_geo_file(geo_agg, append_pre2024_CT_defs=append_pre2024_CT_defs)
    if not source_has_all_counties:
        geo_file = geo_file[["County name", 'GeoFips', "GeoAgg1_name", 'GeoAgg1Fips']]
    n_geo_levels = int(geo_file.shape[1]/2)-1
    # n_obs_prev = df.shape[0]
    df = df.merge(geo_file.drop(columns=["GeoAgg1_name", "County name"]), on="GeoFips", how="inner")
    assert pd.isnull(df['GeoAgg1Fips']).sum()==0, "Error: There are counties without GeoAgg1Fips"
    # print(f"Going from {n_obs_prev} to {df.shape[0]} rows after merging with geo file")

    # 3. Calculate
    if verbosity: print("Main calculations")
    df_agg = do_main_calcs_pi(df, tablename, n_geo_levels)

    if (isinstance(geo_agg, str) and geo_agg=="PORT" and source_has_all_counties) or custom_df_is_port:
        # PORT return 0s for empty state/Region x PORT geographies. 
        df_all_blanks = gen_full_df_index_for_PORT(geo_file, df['TimePeriod'].unique(), 
                                                   df_agg['Code'].sort_values().drop_duplicates())
        df_agg = df_all_blanks.merge(df_agg, on=['GeoAggFips', 'Code', 'TimePeriod'], how='outer', indicator=True)

        df_agg.loc[df_agg['_merge']=='left_only', ["DataValue", "child_count"]] = 0
        df_agg["DataValue"] = df_agg["DataValue"].astype(np.int64)
        df_agg["child_count"] = df_agg["child_count"].astype(int)

        df_agg.loc[df_agg['_merge']=='left_only', ["has_suppressed", "has_NA"]] = False
        df_agg[["has_suppressed", "has_NA"]] = df_agg[["has_suppressed", "has_NA"]].astype(bool)

        df_agg = df_agg.drop(columns=['_merge'])
    
    df_agg = df_agg.merge(meta_data, on=['Code','TimePeriod'], how='left', indicator=False)
    df_agg["LineCode"] = df_agg['Code'].str.slice(len(tablename)+1).astype(int)
    #df_agg = df_agg.drop(columns=['Code'])
    df_agg = df_agg.sort_values(['GeoAggFips', 'LineCode', 'TimePeriod']).reset_index(drop=True)

    # 5. Cleanup for presentation
    if ret_na_0withnote:
        df_agg['NoteRef'] = pd.Series("", index=df_agg.index, dtype='string') #make it a string dtype rather than object
        df_agg.loc[df_agg['has_suppressed'], 'NoteRef'] = "(D)"
        df_agg.loc[df_agg['has_NA'], 'NoteRef'] += "(NA)"        
        for h in ["has_suppressed", "has_NA"]:
            df_agg.loc[df_agg[h], 'DataValue'] = 0
        df_agg['DataValue'] = df_agg['DataValue'].astype(np.int64)
    else:
        # convert the zeros to missing
        for h in ["has_suppressed", "has_NA"]:
            df_agg.loc[df_agg[h], 'DataValue'] = np.nan        
    df_agg = df_agg.drop(columns=['has_suppressed', 'has_NA'])

    #Bring in GeoName
    line_descs = line_descs.rename(columns={"Key": "LineCode", "Desc": "Description"})
    line_descs['LineCode'] = line_descs['LineCode'].astype(int)
    line_descs['Description'] = line_descs['Description'].str.slice(len(f"[{tablename}] "))
    df_agg = (df_agg.rename(columns={"GeoAggFips":"GeoFips"})
              .merge(get_fips_name_mapping(geo_file), on='GeoFips', how='left')
              .merge(line_descs, on="LineCode", how="left"))
    

    ind_df = pd.read_csv(pathlib.Path(__file__).parent.joinpath(f"metadata/{tablename}_LineCode_indent.csv"), dtype=int)
    df_agg = df_agg.merge(ind_df, how="left", on="LineCode")
    df_agg['Description'] = df_agg['indent'].apply(lambda k: " " * k) + df_agg['Description'] # add indent spaces
    df_agg = df_agg.drop(columns=['indent'])

    index_vars = ['GeoFips', "GeoName", 'LineCode', 'Description', 'TimePeriod', 'Code']
    df_agg = df_agg[index_vars + [c for c in df_agg.columns if c not in index_vars]].drop(columns=['child_count'])
    
    df_agg.attrs['detail'] = {'Notes':df_notes}

    return df_agg


def pull_source_data_pi(start_year: int, last_year: int, bea_key: str, tablename: str, 
                        geofips_filter:Optional[str]=None, geo_agg:Optional[Union[str,pathlib.Path,pd.DataFrame]]=None, 
                        append_pre2024_CT_defs:bool=True, verbosity: int=1) -> pd.DataFrame:
    """Pull the necessary source data from BEA API to be used to calculate Personal Income aggregate estimates.
    It will remove new CT counties from pre-2024 and old CT counties from 2024 onward since those are not consistent across years.

    Args:
        start_year (int): First year for desired results data
        last_year (int): Last year for returned data
        bea_key (str): BEA API key
        tablename (str): Table name
        geofips_filter (Optional[str]): a fips code of a geographic aggregate to filter data pulled. Must be a first-level aggregate (MSA or PORT).
        geo_agg (Optional[Union[str,pathlib.Path,pd.DataFrame]]): defaults to None. If using geofips_filter, must be a key to bundled file, Path/str to csv, or dataframe.
        append_pre2024_CT_defs (bool, optional): If True, append bundled pre-2024 CT definitions to geo_agg if MSA or PORT. Defaults to True.
        verbosity (int, optional): Level of verbosity (high is more). Defaults to 1 (show progress).

    Returns:
        pd.DataFrame: source data (county-level). Does remove new CT counties from pre-2024 and old CT counties from 2024 onward.
    """    
    if tablename in ["CAINC5S", "CAINC6S"]:
        raise ValueError("Utilities not built for historical SIC-based tables, use NAICS-based tables (-N).")
    if tablename in ["CAINC5", "CAINC6"]:
        tablename = tablename+"N"
    assert tablename in CAINC_tables, f"Error: Unknown tableID={tablename}."
    if start_year<cainc_start_year: 
        raise ValueError(f"Data prior to {cainc_start_year} not supported for GDP calculations.")

    if geofips_filter is None:
        geofips = 'COUNTY'
    else:
        geofips = ','.join(get_counties_for_geoagg(geofips_filter, geo_agg, append_pre2024_CT_defs))
    df, line_codes = pull_regional_data(tablename, geofips, start_year, last_year, bea_key, verbosity=verbosity)
    df_tp = df['TimePeriod'].astype(int)
    df = df[((df_tp<CT_new_start) & ~df['GeoFips'].isin(new_CT_cnty_fips)) | ((df_tp>=CT_new_start) & ~df['GeoFips'].isin(old_CT_cnty_fips))]

    # Now that we filter the CT counties the below isn't always true
    #assert df.shape[0]==len(df["GeoFips"].unique())*line_codes.shape[0]*(last_year-start_year+1), "Error: Pulled non-rectangular data"
    
    return df, line_codes

def pull_merge_calc_pi(tablename: str, start_year: int, last_year: int, geo_agg: Union[str,pathlib.Path,pd.DataFrame],
                       bea_key: str,  geofips_filter:Optional[str]=None, custom_df_is_port:bool=False, 
                       append_pre2024_CT_defs:bool=True, verbosity: int=1) -> pd.DataFrame:
    """Pulls source data from BEA API, merges, and does calculations to calculate Personal Income aggregate estimates.

    Args:
        tablename (str): Table name (e.g., "CAINC1")
        start_year (int): First year for returned data
        last_year (int): Last year for returned data
        geo_agg (Union[str,pathlib.Path,pd.DataFrame]): key (MSA or PORT) to bundled file, Path/str to csv, or dataframe
        bea_key (str): BEA API key
        geofips_filter (Optional[str]): a fips code of a geographic aggregate to filter data pulled. Must be a first-level aggregate (MSA or PORT).
        custom_df_is_port (bool, option): If passing in a custom geo_agg data frame, treat as PORT 
          (return all possible product of portions). Defaults to False.
        append_pre2024_CT_defs (bool, optional): If True, append bundled pre-2024 CT definitions to geo_agg if MSA or PORT. Defaults to True.
        verbosity (int, int): Level of verbosity (high is more). Defaults to 1 (show progress).

    Returns:
        pd.DataFrame: Dataframe at the aggregated level(s)
    """    
    df, line_codes = pull_source_data_pi(start_year, last_year, bea_key, tablename, geofips_filter=geofips_filter, 
                                         geo_agg=geo_agg, append_pre2024_CT_defs=append_pre2024_CT_defs,verbosity=verbosity)
    df_agg = do_merging_and_calculations_pi(df, line_codes, geo_agg, tablename, 
                                            source_has_all_counties=(geofips_filter is None), 
                                            custom_df_is_port=custom_df_is_port, 
                                            append_pre2024_CT_defs=append_pre2024_CT_defs, verbosity=verbosity)
    return df_agg
