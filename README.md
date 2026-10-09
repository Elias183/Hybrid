Hybrid Hydrological Modeling

Model Description
This project implements a Hybrid Data-Driven and Process-Based Modeling framework for streamflow forecasting. It integrates the physically-based QSWATPLUS model with advanced machine learning algorithms including Random Forest (RF), XGBoost, LightGBM, and LSTM.

The integration is achieved via a Stacking Ensemble (Meta-learner) and Bayesian Model Averaging (BMA), which effectively bridge the gap between physical process understanding and the pattern-recognition capabilities of ML.

Hybrid QSWAT+–machine-learning streamflow forecasting

This repository contains the notebook, Python workflow, and generated results for a hybrid streamflow forecasting study in the Upper Ganale River Basin. The workflow combines daily QSWAT+ discharge with observed streamflow and climate inputs, then trains Random Forest (RF), XGBoost, LightGBM, and Long Short-Term Memory (LSTM) models. Ridge regression stacks out-of-fold base-model predictions; a Bayesian Model Averaging (BMA) mixture provides a second ensemble estimate.

The executed workflow evaluates the overlapping historical record chronologically: calibration/tuning uses 1988–2002, and independent validation uses 2003–2011. Future projections are generated for SSP2-4.5 and SSP5-8.5 using the supplied QSWAT+ scenario runs and CMIP6 climate data.

## Contents

- `hybrid_qswat_cmip6_workflow.ipynb` — executed notebook with method description, results, and figures.
- `hybrid_qswat_cmip6_workflow.py` — script version of the workflow.
- `hybrid_outputs/model_performance_table.csv` — Table X metrics for QSWAT+, each ML model, stacking, and BMA.
- `hybrid_outputs/cross_validation_results.csv` — three-fold expanding-window CV summaries within calibration.
- `hybrid_outputs/hyperparameter_search_results.csv` — selected RF, XGBoost, and LightGBM hyperparameters.
- `hybrid_outputs/cmip6_precipitation_model_selection.csv` — six precipitation-model names and their Rainfall CRI scores.
- `hybrid_outputs/bma_weights.csv` — fitted BMA weights and residual variances.
- `hybrid_outputs/model_validation_predictions.csv` — daily observed and predicted flows for the independent period.
- `hybrid_outputs/chronological_calibration_validation_predictions.csv` — stacked predictions for calibration CV and validation.
- `hybrid_outputs/future_cmip6_hybrid_streamflow.csv` — combined daily projections for both scenarios.
- `hybrid_outputs/future_cmip6_hybrid_streamflow_ssp245.csv` and `..._ssp585.csv` — scenario-specific projection tables.
- `hybrid_outputs/calibration_validation_scatter.png` and `future_cmip6_streamflow.png` — diagnostic and projection figures.
- `hybrid_outputs/hybrid_models.joblib` and `lstm_model.keras` — fitted model artifacts.

## Method summary

### Historical inputs and target

The historical daily modeling table is formed by joining observed discharge, QSWAT+ simulated discharge, and daily climate summaries on date. The target is observed streamflow. QSWAT+ discharge is retained as a process-model predictor and separately scored as a baseline. Climate inputs include precipitation, maximum and minimum temperature, solar radiation, wind, and humidity.

The merged historical input and model-ready data cover 1988–2011 after date matching and feature warm-up: 5,472 calibration rows (1988–2002) and 3,287 validation rows (2003–2011). The 7-day lag/rolling features remove the initial warm-up days. The calibration-validation boundary is chronological; validation data are not used for hyperparameter tuning, feature scaling, or ensemble fitting.

### Features and models

The workflow uses 19 predictors: QSWAT+ flow; daily climate; one-day lagged QSWAT+ flow; one- and seven-day lagged precipitation; seven-day rolling precipitation; one- and seven-day lagged observed flow; seven-day rolling observed flow; and calendar fields (year, month, day of year, weekday, and ISO week). LSTM features are standardized using a scaler fit on calibration data only. Tree models use the unscaled features.

RF, XGBoost, and LightGBM hyperparameters are selected with randomized search (four sampled configurations per model) scored by R² on three expanding-window `TimeSeriesSplit` folds. LSTM uses a one-step sequence architecture with early stopping. Base-model calibration predictions used by Ridge stacking and BMA are out-of-fold predictions. Stacking uses Ridge regression with alpha selected by internal time-series CV; its reported calibration CV is nested and chronological. BMA weights and component variances are estimated from calibration-only out-of-fold predictions using an iterative Gaussian-mixture fit.

### Future scenario inputs

Future temperature is taken from CanESM5. Daily precipitation is averaged across the six highest-ranked models in the supplied ranking CSV (`Rainfall_CRI`, descending): **INM-CM4-8, CanESM5, KACE-1-0-G, INM-CM5-0, FGOALS-g3, and GISS-E2-1-G**. This resolves the apparent five-versus-six discrepancy: the supplementary model table lists all six; the shorter narrative omitted INM-CM4-8. Archive or deposit the ranking CSV with the code and data release so this selection can be reproduced.

QSWAT+ outlet flow is extracted from each scenario SQLite output using `cha001` / `gis_id=1`, which the project owner confirmed as the outlet. Near-term runs provide 2015–2044; mid-term runs are restricted to 2045–2074 to avoid the overlap between the run periods. The saved daily projections span 2015–2074.

## Reproduce the workflow

### Requirements

The executed environment used Python 3.12 and the following packages:

| Package | Version |
|---|---:|
| pandas | 3.0.5 |
| NumPy | 2.5.2 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.4.1 |
| LightGBM | 4.7.0 |
| TensorFlow | 2.21.0 |
| Matplotlib | 3.11.2 |
| joblib | 1.6.0 |

Install compatible builds of these packages in a Python environment with TensorFlow support. Exact numerical reproduction can vary by operating system, hardware, library build, and TensorFlow execution backend.

### Required input files

The script no longer contains the original user's absolute Windows input paths. By default, place inputs in this project-relative layout:

```text
data/
├── Q_observation_historical.csv
├── Q_simulation_historical.csv
├── combined_climate_with_station_names.csv
├── combined_cri_scores.csv
├── CMIP6/
│   └── <Model_Name>/<Model_Name>_daily_data.csv
└── qswat_scenarios/
    ├── SSPN245/Results/swatplus_output.sqlite
    ├── SSPM245/Results/swatplus_output.sqlite
    ├── SSPN585/Results/swatplus_output.sqlite
    └── SSPM585/Results/swatplus_output.sqlite
```

Override paths with `HYBRID_PROJECT_DIR` (project/output directory), `HYBRID_DATA_DIR` (input root), `Q_OBSERVATION_FILE`, `Q_SIMULATION_FILE`, `STATION_CLIMATE_FILE`, `CMIP6_DATA_DIR`, `CMIP6_RANKING_FILE`, or `QSWAT_FUTURE_DIR`. Individual file/folder variables override the matching `HYBRID_DATA_DIR` defaults. Outputs are written under `hybrid_outputs/` in `HYBRID_PROJECT_DIR`.

Required inputs are:

1. `Q_observation_historical.csv` — daily observed flow with a date and observed-flow column.
2. `Q_simulation_historical.csv` — daily historical QSWAT+ flow with a date and simulated-flow column.
3. `combined_climate_with_station_names.csv` — daily station data with `date`, `pcp`, `tmax`, `tmin`, `slr`, `wnd`, and `hmd`; the workflow averages stations by date.
4. CMIP6 daily data files under one folder per model, named `<Model_Name>_daily_data.csv`. They need dates, scenario labels, precipitation, and CanESM5 maximum/minimum temperature columns. The ranking CSV must contain `Model_Name` and `Rainfall_CRI` columns and include at least six models.
5. QSWAT+ future scenario SQLite outputs at the paths above, or corresponding paths configured with `QSWAT_FUTURE_DIR`.

The CSVs and SQLite project inputs are not redistributed in this repository. Obtain permission and cite their source and processing history in any publication or public data release.

### Run

From the project directory, after placing inputs in the layout above and installing compatible package versions:

```powershell
python hybrid_qswat_cmip6_workflow.py
```

The script trains models, writes historical evaluation files, extracts the scenario outlet flows, computes CMIP6 hybrid projections, and saves results under `hybrid_outputs/`. Set the environment variables above before starting the notebook or script when using other input locations; the notebook passes its environment to the script.

## Executed evaluation results

Independent-period results (flow errors in m³/s):

| Model | R² | NSE | RMSE | MAE |
|---|---:|---:|---:|---:|
| QSWAT+ | 0.6238 | 0.6238 | 53.5453 | 38.6216 |
| Random Forest | 0.9567 | 0.9567 | 18.1655 | 8.4263 |
| XGBoost | 0.9576 | 0.9576 | 17.9835 | 8.4481 |
| LightGBM | 0.9580 | 0.9580 | 17.8984 | 8.5556 |
| LSTM | 0.9052 | 0.9052 | 26.8733 | 18.3223 |
| Stacking ensemble | 0.9593 | 0.9593 | 17.6196 | 8.2042 |
| BMA | 0.9590 | 0.9590 | 17.6771 | 8.3968 |

Full training and three-fold time-series CV results are in `model_performance_table.csv` and `cross_validation_results.csv`. LSTM fold performance was variable (mean CV R² 0.4661, SD 0.5950); report this uncertainty with the other model results.

## Interpretation and limitations

- R² and NSE are numerically equal in the reported tables because each is computed on the same observed values and NSE uses the observed mean as its reference.
- Historical validation is a daily one-step-ahead evaluation: lagged observed streamflow features use preceding observed days, as would be available for a daily update. The future simulation instead recursively feeds the model's own predicted flows into its lagged-flow features.
- The LSTM receives a sequence length of one day per sample. Temporal context is supplied by the engineered lag and rolling predictors; the LSTM itself does not process a multi-day recurrent sequence.
- The 2015 projection initialization uses the final historical flow/climate values from 2011 because no 2012–2014 observed bridge is available in the supplied inputs. That gap should be disclosed and, ideally, replaced with an explicit spin-up or bridge period before interpreting early projection years.
- Future solar radiation, wind, and humidity are held at historical means; only precipitation and minimum/maximum temperature vary with the supplied CMIP6 scenarios. Future projections are conditional on the supplied QSWAT+ scenario simulations and climate-model inputs, not independent observations.
- The six precipitation models are selected by descending CRI from the ranking CSV; the six selected names are listed above. Include the ranking data with the public release so the selection is auditable.
- Results describe the provided datasets, period, outlet, and model configuration. They should not be generalized to other basins or interpreted as a causal climate attribution without additional analysis.

## Citation and data availability

### Data and software references

- **CMIP6 downscaled daily climate:** Thrasher, B. et al. (2022). “NASA Global Daily Downscaled Projections, CMIP6.” *Scientific Data*, 9, 262. [Article/DOI](https://doi.org/10.1038/s41597-022-01393-4) · [NASA NCCS dataset access](https://www.nccs.nasa.gov/data-collections/nex-gddp-cmip6/). Follow the [NEX-GDDP-CMIP6 technical note](https://www.nccs.nasa.gov/sites/default/files/NEX-GDDP-CMIP6-v2-Tech_Note.pdf) for acknowledgement and terms; record dataset version and access date.
- **CHIRPS precipitation reference:** Funk, C. et al. (2015). “The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes.” *Scientific Data*, 2, 150066. [Article/DOI](https://doi.org/10.1038/sdata.2015.66) · [CHIRPS data page](https://www.chc.ucsb.edu/data/chirps). Confirm the version used and record access date.
- **CHIRTS temperature reference:** Verdin, A. et al. (2020). “Development and validation of the CHIRTS-daily quasi-global high-resolution daily temperature data set.” *Scientific Data*, 7, 303. [Article/DOI](https://doi.org/10.1038/s41597-020-00643-7) · [CHIRTS-daily data page](https://chc.ucsb.edu/data/chirtsdaily).
- **SWAT+:** cite the exact version used to produce the supplied simulations. The [official citation and license page](https://docs.swat.tamu.edu/introduction/citing-and-license/) provides versioned release DOIs; [SWAT+ source code](https://github.com/swat-model/swatplus). The project used [QSWAT+](https://github.com/swat-model/QSWATPlus) to build the model. Record SWAT+, QSWAT+, QGIS, and SWAT+ Toolbox versions.
- **Model methods:** Breiman (2001), Random Forests, [DOI](https://doi.org/10.1023/A:1010933404324); Chen & Guestrin (2016), XGBoost, [DOI](https://doi.org/10.1145/2939672.2939785); Ke et al. (2017), LightGBM, [NeurIPS paper](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html); Hochreiter & Schmidhuber (1997), LSTM, [DOI](https://doi.org/10.1162/neco.1997.9.8.1735); Wolpert (1992), stacked generalization, [DOI](https://doi.org/10.1016/S0893-6080(05)80023-1); and Ajami et al. (2007), Bayesian multimodel combination for hydrologic prediction, [DOI](https://doi.org/10.1029/2005WR004745). Describe this implementation's calibration-only OOF weight estimation when citing BMA.
- **Nash–Sutcliffe efficiency:** Nash & Sutcliffe (1970), [DOI](https://doi.org/10.1016/0022-1694(70)90255-6).
- **Observed meteorology:** Ethiopian Meteorological Institute (EMI; previously Ethiopian National Meteorological Agency), as stated in the supplementary methods. [Official institute and data services](https://www.ethiomet.gov.et/). Cite the station data under EMI terms and report station names, coverage, and access date.
- **Observed streamflow:** Ministry of Water and Energy (MoWE), Chenemsa gauge, 1988–2011, as stated in the supplementary methods. [Official ministry](https://mowe.gov.et/). No public download URL for this study-specific gauge series was supplied. Report station identifier, access date, and data-use permissions.
- **Soil:** FAO/IIASA/ISRIC/ISS-CAS/JRC, Harmonized World Soil Database v1.2. [FAO v1.2 page and access](https://www.fao.org/soils-portal/soil-survey/soil-maps-and-databases%20/harmonized-world-soil-database-v12/en/). Confirm the project used v1.2 rather than a later release.
- **Elevation:** Shuttle Radar Topography Mission (SRTM), as identified in the supplementary material. [SRTM project and data access](https://srtm.csi.cgiar.org/). Add the exact product, resolution, and access date from the model project.
- **Land cover:** the supplementary material identifies “Copernicus Global Land Cover 2009.” [Copernicus land-cover catalogue](https://www.copernicus.eu/en/access-data/copernicus-services-catalogue/land-cover-classification-gridded-maps-1992-present). Confirm the actual product name, collection, resolution, and year from raster metadata before citing; the supplied description does not identify a unique release.

### Project-specific citation and data release

The supplementary material mentions an Ali (2026) Zenodo deposit for climate-model evaluation scripts, but does not give its DOI or record URL. Add the verified record if that workflow is used to reproduce the CRI ranking. No public link/DOI was supplied for the meteorological station data, gauge series, QSWAT+ project, model-ranking table, or processed CMIP6 files. Before public release, deposit the code and shareable derived data in a stable repository, archive the exact ranking and processing configuration, and mint a versioned DOI. State which source data can be requested from custodians and do not redistribute restricted data without permission.

Recommendations
•	Extend the framework with CA Markov LULC forecasts to simulate land use change impacts on hydrology.
•	Deploy results in a dashboard/API for stakeholder decision support.
•	Apply the hybrid approach to other Ethiopian basins for comparative analysis.
•	Integrate socio economic indicators for holistic climate smart water management.

References
•	Abbaspour, K. C., Vaghefi, S. A., & Srinivasan, R. (2018). A guideline for successful calibration and validation of hydrological models. Water, 10(6), 728. https://doi.org/10.3390/w10060728
•	Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. https://doi.org/10.1145/2939672.2939785
•	Hochreiter, S., & Schmidhuber, J. (1997). Long short term memory. Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735 
•	Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., … Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3146–3154.
•	Solanki, A., et al. (2025). Improving streamflow prediction using multiple hydrological models and machine learning. Water Resources Research, 61(2), e2024WR036789. https://doi.org/10.1029/2024WR036789 
•	Vaghefi, S. A., Abbaspour, K. C., & Srinivasan, R. (2019). Integrating SWAT and machine learning models for improved hydrological predictions. Hydrology, 6(3), 45. https://doi.org/10.3390/hydrology6030045

