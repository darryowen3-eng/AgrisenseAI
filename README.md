# AgriSenseAI

## Intelligent Crop Yield Prediction and Agricultural Decision Support System for Zambia

AgriSenseAI is a Zambia-focused agricultural intelligence system for **Maize and Groundnuts**. It combines real Zambia agricultural data, soil, weather, environmental and topographic information with machine-learning yield prediction, model-driven recommendations, risk tracking, GPS location detection, and CNN-based crop-leaf diagnosis.

## Architecture

```text
Farmer
  ↓
Streamlit Frontend
  ↓
FastAPI Backend
  ├── Yield Feature Engineering → ExtraTrees Model
  └── Leaf Image → MobileNetV2 CNN
  ↓
Predictions + Harvest Estimate + Recommendations + Risks
  ↓
Farmer
```

## Main Objectives

- Predict crop yield in MT/ha.
- Estimate expected harvest from farm size.
- Use real Zambia agricultural data.
- Incorporate soil and environmental information.
- Detect farmer location through browser GPS when permission is available.
- Provide model-driven recommendations.
- Highlight important risks.
- Diagnose crop-leaf images during the growing season.
- Keep the interface farmer-friendly.
- Provide a foundation for a future time-aware forecasting system.

## Technology Stack

### Frontend
- Python
- Streamlit
- streamlit-js-eval
- Pandas
- Image upload
- Map visualization

### Backend
- FastAPI
- Uvicorn
- Pydantic

### Machine Learning
- Python
- Pandas
- NumPy
- scikit-learn
- ExtraTreesRegressor
- joblib

### Computer Vision
- TensorFlow
- Keras
- MobileNetV2

### Data Sources
- Zambia Statistics Agency Crop Forecast Survey
- NASA POWER
- CHIRPS
- iSDAsoil
- SRTM
- Zambia location data

## Project Structure

```text
AgriSenseAI/
├── api/
│   ├── main.py
│   ├── schema.py
│   └── feature_engineering.py
├── frontend/
│   └── app.py
├── data/
│   ├── agrisense_master.csv
│   ├── agrisense_training_ready.csv
│   ├── nasa_power_daily.csv
│   ├── zambia_locations.csv
│   ├── zambia_environment_features.csv
│   ├── zambia_seasonal_environment_features.csv
│   ├── zambia_soil_features.csv
│   ├── zambia_soil_features_clean.csv
│   ├── zambia_topography.csv
│   └── zamstats/
│       └── Crop SM Final.sav
├── models/
│   └── agrisense/
│       ├── baseline/
│       │   └── agrisense_baseline_model.joblib
│       ├── management/
│       │   └── agrisense_management_model.joblib
│       ├── agrisense_production_model.joblib
│       └── vision/
│           ├── maize/
│           └── groundnuts/
├── scripts/
├── train_agrisense_baseline.py
├── train_agrisense_management.py
├── train_agrisense_production.py
├── requirements.txt
├── README.md
└── CHANGELOG.md
```

## Agricultural Dataset

Primary source:

```text
data/zamstats/Crop SM Final.sav
```

Approximate source size:

```text
32,031 rows
124 columns
```

Target crops:

| Crop | Code | Records |
|---|---:|---:|
| Maize | 101 | 17,035 |
| Groundnuts | 503 | 3,467 |

Final target dataset:

```text
20,502 records
```

## Location Matching

The project matches agricultural records using province, district, ward, latitude and longitude.

Approximately:

```text
686 raw location rows
108 unique valid coordinate locations
18,893 / 20,502 agricultural records matched
≈ 92.15% match rate
```

Unmatched records are not assigned fabricated coordinates.

## Soil Data

Cleaned iSDAsoil data includes:

```text
soil_ph
soil_organic_carbon
soil_total_nitrogen
soil_phosphorus
soil_potassium
soil_cec
soil_bulk_density
soil_clay
soil_sand
soil_silt
soil_stone
soil_depth_to_bedrock
```

Depths include:

```text
0–20 cm
20–50 cm
```

where available.

## Environmental Data

### NASA POWER

Daily weather coverage:

```text
2010–2025
```

Variables:

```text
T2M
T2M_MAX
T2M_MIN
RH2M
ALLSKY_SFC_SW_DWN
WS2M
PRECTOTCORR
```

Processed dataset:

```text
≈631,152 rows
```

### CHIRPS

Rainfall features include:

```text
chirps_mean_pentad
chirps_total_2010_2025
season_total_rainfall
season_mean_daily_rainfall
```

### NDVI

```text
ndvi_max_2010_2025
ndvi_mean_2010_2025
```

### Topography

```text
elevation_m
slope_degrees
```

## Seasonal Environmental Data

```text
Establishment: 2024-10-01 → 2024-11-30
Early growth:  2024-12-01 → 2025-01-31
Mid growth:    2025-02-01 → 2025-03-31
Late growth:   2025-04-01 → 2025-05-31
```

## Master Dataset

```text
data/agrisense_master.csv
```

Shape:

```text
20,502 rows × 365 columns
```

It combines crop, management, location, soil, weather, environmental, topographic, seasonal and yield information.

## Yield Target

Target:

```text
Yield
```

corresponding to:

```text
YIELDMT
```

The supplied yield target is used directly rather than reconstructing yield from production and harvested area.

Overall:

```text
Mean:   1.4811 MT/ha
Std:    1.3345
Median: 1.1500 MT/ha
Max:    5.9630 MT/ha
```

### Maize

```text
n = 17,035
Mean ≈ 1.6501 MT/ha
Median ≈ 1.3800 MT/ha
Maximum ≈ 5.9630 MT/ha
```

### Groundnuts

```text
n = 3,467
Mean ≈ 0.6508 MT/ha
Median ≈ 0.5333 MT/ha
Maximum ≈ 2.3457 MT/ha
```

## Baseline ML Model

Model:

```text
ExtraTreesRegressor
```

Configuration:

```text
n_estimators = 500
max_features = 0.8
min_samples_leaf = 3
random_state = 42
n_jobs = -1
```

Saved model:

```text
models/agrisense/baseline/agrisense_baseline_model.joblib
```

The baseline model uses 39 final features: 31 numeric and 8 categorical.

### Numeric features

```text
AREA_PLANTED
CR05_AREA
PROV
REGION
latitude
longitude
soil_ph_0_20
soil_ph_20_50
soil_organic_carbon_gkg_0_20
soil_organic_carbon_gkg_20_50
soil_total_nitrogen_gkg_0_20
soil_total_nitrogen_gkg_20_50
soil_phosphorus_ppm_0_20
soil_phosphorus_ppm_20_50
soil_potassium_ppm_0_20
soil_potassium_ppm_20_50
soil_clay_pct_0_20
soil_sand_pct_0_20
soil_silt_pct_0_20
soil_bulk_density_gcm3_0_20
soil_cec_cmolkg_0_20
soil_stone_pct_0_20
elevation_m
slope_degrees
chirps_mean_pentad
chirps_total_2010_2025
ndvi_max_2010_2025
ndvi_mean_2010_2025
season_total_rainfall
season_mean_daily_rainfall
TCP05
```

### Categorical features

```text
crop
DIST
CONST
WARD
province
district_name
district_key
ward_environment
```

### Baseline performance

Five-fold cross-validation:

```text
MAE  = 0.8663 MT/ha
RMSE = 1.1598 MT/ha
R²   = 0.2411
```

## Production Model

The larger production model uses:

```text
297 predictors
281 numeric
16 categorical
```

Five-fold out-of-fold performance:

```text
MAE  = 0.7350 MT/ha
RMSE = 1.0325 MT/ha
R²   = 0.4014
```

Saved model:

```text
models/agrisense/agrisense_production_model.joblib
```

It uses later seasonal environmental information, so it is not currently a strict pre-planting forecasting model.

## Management Model

Current validation:

```text
MAE  = 0.8848 MT/ha
RMSE = 1.1761 MT/ha
R²   = 0.2194
```

Saved model:

```text
models/agrisense/management/agrisense_management_model.joblib
```

## FastAPI Backend

Main files:

```text
api/main.py
api/schema.py
api/feature_engineering.py
```

Responsibilities:

- receive farmer inputs
- construct model features
- retrieve environmental/soil information
- run yield prediction
- calculate harvest
- generate recommendations
- generate risks
- run CNN diagnosis
- return structured responses

### Endpoints

```http
GET /health
POST /predict
POST /predict-midseason
```

## Frontend

Main file:

```text
frontend/app.py
```

The farmer interface contains:

- GPS detection
- farm map
- crop selection
- farm size
- location
- soil information
- planting information
- fertilizer information
- previous crop
- water source
- yield prediction
- recommendations
- risks
- leaf image upload
- CNN diagnosis

## GPS Auto-Detection

The frontend uses browser geolocation through `streamlit-js-eval`.

It retrieves:

```text
latitude
longitude
```

The farmer must grant browser location permission.

Manual coordinates can be used when GPS is unavailable.

## Recommendation Engine

Recommendations use counterfactual/model-sensitivity predictions.

The backend can compare scenarios such as:

```text
Current scenario
    ↓
Alternative planting scenarios
Alternative crop scenario
Alternative pH scenarios
    ↓
Compare predictions
    ↓
Generate farmer-facing recommendation
```

These represent model sensitivity, not causal proof.

## Risk Engine

The system checks for:

- missing soil information
- low predicted yield
- prediction below the overall median
- planting-timing sensitivity
- environmental-data limitations

## CNN Leaf Diagnosis

The vision model uses:

```text
MobileNetV2
224 × 224 RGB
```

### Maize classes

```text
common_rust
corn_leaf_blight
gray_leaf_spot
healthy
insect_damage
```

### Groundnut classes

```text
early_leaf_spot
healthy
late_leaf_spot
nutrition_deficiency
rust
```

The CNN returns:

```text
Diagnosis
Confidence
Probabilities
Action
```

It does not use arbitrary disease-to-yield multipliers.

## Important Source Mapping

Validated ZamStats metadata includes:

```text
TCP05 = planting timing
```

TCP05 is **not** soil pH.

Also:

```text
TCP12_N
```

is not fertilizer amount. Fertilizer-related source groups include:

```text
TCP01
TCP02
TCP03
```

## Current Model Limitations

The current baseline does not directly learn all farmer-interface fields:

```text
crop variety
previous crop
water source
fertilizer type
fertilizer amount
farmer-selected soil type
```

These should not be presented as learned quantitative effects.

The current model also contains seasonal environmental information that may become available later in the growing season. Therefore it is not yet a fully leakage-free real-time pre-planting model.

## Installation

```bash
cd ~/AgriSenseAI
pip install -r requirements.txt
```

## Run FastAPI

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API:

```text
http://127.0.0.1:8000
```

Docs:

```text
http://127.0.0.1:8000/docs
```

## Run Frontend

In another terminal:

```bash
cd ~/AgriSenseAI
streamlit run frontend/app.py
```

## Test Backend

```bash
curl http://127.0.0.1:8000/health
```

## Current Status

- [x] Zambia agricultural dataset
- [x] Maize data
- [x] Groundnut data
- [x] Location matching
- [x] Soil integration
- [x] Environmental integration
- [x] Weather integration
- [x] Topography
- [x] Master dataset
- [x] Baseline yield model
- [x] Production model
- [x] Management model
- [x] FastAPI backend
- [x] Streamlit frontend
- [x] GPS detection
- [x] Farm map
- [x] Model-driven recommendations
- [x] Risk engine
- [x] Leaf image upload
- [x] CNN diagnosis
- [x] CNN confidence
- [x] CNN probability display

## Future Development

- [ ] True pre-planting model
- [ ] Time-aware mid-season model
- [ ] Better uncertainty estimation
- [ ] Strict soil matching threshold
- [ ] Complete fertilizer metadata mapping
- [ ] Better management-response modeling
- [ ] Farmer history
- [ ] Farm-season tracking
- [ ] Model monitoring
- [ ] Improved CNN validation
- [ ] Zambia field-image dataset
- [ ] Database integration
- [ ] Authentication
- [ ] Deployment
- [ ] Low-connectivity/offline functionality

## Long-Term Architecture

```text
                         FARMER
                            │
                            ▼
                     FARMER APP
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
          ▼                                   ▼
    PRE-PLANTING                          MID-SEASON
      ENGINE                                ENGINE
          │                                   │
          ▼                                   ▼
   Early-season ML                       CNN + ML
          │                                   │
          └────────────────┬──────────────────┘
                           ▼
                    DECISION ENGINE
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
           YIELD      RECOMMENDATIONS   RISKS
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                         FARMER
```

## Project Principles

### Real data
Use real agricultural and environmental information whenever possible.

### No fabricated data
Missing information should remain missing.

### Model-driven recommendations
Recommendations should be connected to model behavior or explicitly identified agronomic rules.

### Farmer-first design
The farmer should not need to understand machine learning to use the system.

### Separate models
Yield prediction and leaf diagnosis are different tasks.

### Transparent limitations
A prediction is an estimate, not a guarantee of actual harvest.

## Summary

```text
REAL ZAMBIA DATA
       +
SOIL
       +
WEATHER
       +
ENVIRONMENT
       +
TOPOGRAPHY
       +
MACHINE LEARNING
       +
COMPUTER VISION
       +
GPS
       +
RECOMMENDATIONS
       +
RISK ANALYSIS
       ↓
AGRICULTURAL DECISION SUPPORT
```

The next major technical milestone is a strictly time-aware forecasting architecture that separates pre-planting, mid-season, and later-season information.
