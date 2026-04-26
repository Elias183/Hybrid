Hybrid Hydrological Modeling

Model Description
This project implements a Hybrid Data-Driven and Process-Based Modeling framework for streamflow forecasting. It integrates the physically-based QSWATPLUS model with advanced machine learning algorithms including Random Forest (RF), XGBoost, LightGBM, and LSTM.

The integration is achieved via a Stacking Ensemble (Meta-learner) and Bayesian Model Averaging (BMA), which effectively bridge the gap between physical process understanding and the pattern-recognition capabilities of ML.

Key Features
Feature Engineering: Incorporation of lagged flow, rolling precipitation statistics, and seasonal time-based features.
Uncertainty Analysis: Normality testing of residuals and RAPS analysis to identify persistent changes.
Climate Change Projections: Forecasting under CMIP6 scenarios (SSP245 and SSP585) for near-term and mid-century periods.
Results Summary
Superior Accuracy: The Hybrid Model (Stacking Ensemble) achieved the best performance (R² ≈ 0.968, NSE ≈ 0.968) compared to individual models.
Trend Analysis: Mann-Kendall tests revealed a significant increasing trend in daily and seasonal flow for most models.
Robustness: ML models significantly improved upon the baseline physically-based model by capturing non-linear hydrological dynamics.

Hybrid Hydrological Modeling
Model Description
This project implements a Hybrid Data Driven and Process Based Modeling framework for streamflow forecasting in the Ganale Dawa River Basin. The framework integrates the physically based QSWAT+ hydrological model with advanced machine learning algorithms — Random Forest (RF), XGBoost, LightGBM, and LSTM.
The integration is achieved through a Stacking Ensemble (meta learner) and Bayesian Model Averaging (BMA), bridging the gap between physical process understanding and the pattern recognition capabilities of ML.
Requirements
•	Software & Tools: 
o	QSWAT+ (for hydrological simulation)
o	Python (Colab/Jupyter environment)
o	Libraries: scikit-learn, xgboost, lightgbm, tensorflow/keras, pandas, numpy, matplotlib
•	Data Inputs: 
o	Climate data (precipitation, temperature, ET, humidity, wind, radiation)
o	Remote sensing data (soil moisture, vegetation indices, DEM, land use)
o	Observed streamflow records (1985–2014)
•	Evaluation Metrics: RMSE, MAE, NSE, R²
Methodology
1.	Data Preprocessing
o	Bias correction and downscaling of climate series.
o	HRU delineation in QSWAT+.
o	Feature engineering: lagged flows, rolling precipitation statistics, seasonal indicators.
2.	Physical Simulation (QSWAT+)
o	Hydrological process simulation at HRU, sub basin, and reach scales.
o	Baseline runoff and water balance outputs.
3.	Machine Learning Models
o	RF: nonlinear feature importance.
o	XGBoost: efficient gradient boosting.
o	LightGBM: scalable boosting for large datasets.
o	LSTM: temporal sequence learning.
4.	Hybrid Integration
o	Stacking ensemble with meta learner.
o	Bayesian Model Averaging for uncertainty quantification.
5.	Validation & Projection
o	Historical validation against observed streamflow.
o	Climate change projections under CMIP6 scenarios (SSP2 4.5 and SSP5 8.5).
Results & Discussion
•	Superior Accuracy: 
The hybrid ensemble achieved R² ≈ 0.968, NSE ≈ 0.968, outperforming individual models and the baseline QSWAT+.
•	Trend Analysis: 
Mann Kendall tests revealed significant increasing trends in daily and seasonal flows.
•	Uncertainty Analysis: 
Residual normality testing and RAPS analysis confirmed robustness and identified persistent changes.
•	Climate Change Projections: 
Under SSP5 8.5, mid century flows show stronger variability, highlighting vulnerability of water resources.
Main Findings
•	Hybrid modeling significantly improves streamflow prediction accuracy.
•	ML models capture nonlinear hydrological dynamics missed by purely process based models.
•	Ensemble fusion provides explicit uncertainty handling.
•	Climate scenarios indicate increasing flow trends with potential implications for irrigation and flood risk.
Conclusion
The QSWAT+–ML hybrid framework demonstrates the value of combining physical realism with data driven learning. It provides robust, accurate, and uncertainty aware streamflow forecasts for the Ganale Dawa River Basin.
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

