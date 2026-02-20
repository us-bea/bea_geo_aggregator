# bea_geo_aggregator
BEA first [announced](https://www.bea.gov/information-updates-national-regional-economic-accounts) on its public website in 2025 and subsequently in its February 5, 2026, news release “Gross Domestic Product by County and Personal Income by County, 2024” that it will no longer produce gross domestic product (GDP) and personal income estimates for Metropolitan Statistical Areas (MSAs), Micropolitan Statistical Areas (MICs), Metropolitan Divisions (DIVs), Combined Statistical Areas (CSAs), and metropolitan and nonmetropolitan portions (PORTs). For more information, see the [FAQ](https://www.bea.gov/help/faq/1481).

These "geographic aggregates" may be useful for some users. We present here a way to derive, in some cases, approximations using the still available county estimates and national industry-level price estimates. These estimates will be of lesser quality than those produced internally at BEA for several reasons: the published industry price and GDP data are less detailed than is used internally, the public county data for current dollar GDP contains suppressions, we don't handle cases where non-standard price indexing fails, and using published data introduces rounding at an intermediate stage of calculation. If one only needs statistics prior to 2024, then the BEA’s [historical archive](https://apps.bea.gov/histdata/) is likely superior (though note that it will not receive revisions to national prices by industry). We do not recommend mixing estimates from the historical archive and this tool as that could create discontinuities at the ‘seam’ between the two sources. Finally, we note that this tool can be used for (a) user-defined geographic aggregates, and (b) estimates for some pre-defined geographic aggregates that had not been available for some BEA tables (not all tables included all aggregates). 

This repository provides Python-based tools to calculate these estimates. There is also an accompanying [technical document](https://www.bea.gov/sites/default/files/2026-02/geo-aggregator-technical-document.pdf) that details the methodology.

## How to run
The tools are comprised two python-based Jupyter notebook interfaces depending on your need: `Geo_Aggregator_simple` for straight-foward cases of looking at single geographic aggregations, and `Geo_Aggregator_full` that shows a few more options. We provide variants that can used in cloud-based notebook environments (e.g., Google Colab) that alleviate the need for a local installation of Python, as well at variants designed to be run locally.

Regardless of the way the notebooks are run, all users will need a [BEA API key](https://apps.bea.gov/api/signup/). The BEA API has a limit on the rate of API requests. Exceeding this will cause you to be temporarily blocked from the API for an hour. The provided `beaapi` python package will automatically try to throttle requests to not exceed this limit (though with heavy usage this still may happen in rare occasions). `beaapi` can not effectively throttle the requests if they happen from separate processes (e.g., multiple windows in Google Colab running simultaneously) making it more likely to go over the limit, so we discourage simultaneous usage.

### Running in cloud (e.g., Google Colab)
Here are example results for Google Colab, though other cloud notebook services may work as well.
1. Go to [colab.research.google.com](http://colab.research.google.com) in a browser and login. You will need a Google account
2. When the dialog appears for opening a notebook, select the `GitHub` tab on the left, enter the repo as `us-bea/bea_geo_aggregator`, click `enter`, and then click the `Geo_Aggregator_simple-cloud.ipynb` (or `Geo_Aggregator_full-cloud.ipynb`) file.
3. Follow the instructions at the top of the notebook.


### Running locally
1. Your Python environment will need the following packages: `pandas`, `numpy`, `jupyter`, `tqdm`, and `beaapi`.
2. Clone the repo and open `Geo_Aggregator_simple.ipynb` (or `Geo_Aggregator_full.ipynb`).
3. Follow the instructions at the top of the notebook.


## Tool Documentation
Geographic options: The can calculate for "MSA" or "PORT" definitions provided by BEA. If you would like to create your own geographic aggregation structure then you can create aggregation tables in the same format. Currently, they are restricted to allow counties to only comprise a single aggregate (i.e., a county can only be part of a single MSA code).

Notebook variants:
* The `_simple` version of the tools will quickly create results for single geographic aggregate (e.g., MSA)
* The `_full` version will take longer but create results for full country. It can also accommodate custom geographic aggregations.

Why are some estimates for geographic aggregates (e.g., MSA) missing?
- GDP: 
    - Current dollar GDP: There is a missing value in the current dollar GDP for that same industry in one of the component counties for that "estimate" year.
    - Quantity index for real GDP: There is a missing or zero value in the current dollar GDP for that same industry for a component county for a year in the span from that "estimate" year to 2017. Missing values can occur if counties definitions change in this span (e.g., CT county redefinitions).
    - Real GDP in chained dollars: The quantity index is missing for that geographic aggregate for that industry for that year.
    - Contributions to Pct Change in real GDP: Real GDP is missing for the All Industry line for the geo aggregate or a component county has a missing value for current dollar GDP in that period or adjacent period.
- Personal Income:
    - Non-ratio lines: That same Line Code is missing in one of the component counties
    - ratio lines: either the numerator is missing or the denominator is either missing or zero.

## How to get help
For methodological question, please first consult the [technical document](https://www.bea.gov/sites/default/files/2026-02/geo-aggregator-technical-document.pdf). If that is not sufficient, please contact reis@bea.gov.

## Quick Links
* [CODE_OF_CONDUCT.md](https://github.com/us-bea/.github/blob/main/CODE_OF_CONDUCT.md)
* [SECURITY](https://github.com/us-bea/.github/blob/main/Security.md)
* [LICENSE](LICENSE)
* [CONTRIBUTING.md](CONTRIBUTING.md)
* [SUPPORT.md](SUPPORT.md)