# Food-insecurity-GP-forecasting

This is a repository for Food-security forecasting using Gaussian process regression

### Repository structure
```
├── Code
│   ├── Stan
│   │   ├── GP_helpers.stan *Collection of user-specified functions*
|   |   ├── GPst_est.stan *For estimating model parameters in spatio-temporal model with SE kernels*
|   |   ├── GPst_pred.stan *Same as above, but for prediction*
│   │   ├── GPst_est_sqcenfBM.stan *For estimating model parameters in spatio-temporal model with fBM kernels*
|   |   └── GPst_pred_sqcenfBM.stan *Same as above, bt for prediction*
│   ├── R
|   |   └── GP_spatio_temporal.R *running the Stan files for spatio-temporal models*
│   └──Python
│       ├── data_check.ipynb *Checking missing values, correlations etc*
│       ├── data_wangling.ipynb *Transforming the data to long format, etc*
│       ├── geocoord_conversion_f.py *helper functions to transform coordinate systems*
|       └── GP-spatiotemporal.ipynb *running the Stan files for spatio-temporal models*
└── Data
    └── Foini2023 *Data copied and edited from the paper cited below
         ├── output_data
         └── Missing-data-info
```
s
### Resources
The base data is
* discussed at [Foini et al. 2023](https://www.nature.com/articles/s41598-023-29700-y#Sec22) and
* available at [Github repo](https://github.com/pietro-foini/ISI-WFP)
