# AgriSenseAI Changelog

All notable development milestones, changes, fixes, and architectural decisions for AgriSenseAI are documented here.

The project is currently under active development.

---

# [Unreleased]

## Added — Farmer-Oriented Frontend

Added farmer-facing inputs for:

- farm location
- crop
- crop variety
- previous crop
- water source
- farm size
- soil type
- soil pH
- planting month
- fertilizer type
- fertilizer amount
- province
- district
- ward

## Added — GPS Auto Detection

Added browser-based GPS detection using `streamlit-js-eval`.

The application can obtain latitude and longitude after browser permission is granted.

Manual coordinates remain available when GPS cannot be obtained.

## Added — Farm Map

Added farm location visualization.

## Added — Model-Driven Recommendations

Added model-sensitivity/counterfactual recommendations using alternative scenarios such as:

- planting month
- crop
- pH

The system compares model predictions before generating recommendations.

## Added — Risk Engine

Added risk tracking for:

- missing soil information
- low predicted yield
- prediction below median
- planting timing sensitivity
- environmental-data limitations

## Added — CNN Image Display

The frontend now displays the uploaded crop-leaf image.

## Added — CNN Diagnosis Display

The mid-season workflow displays:

- uploaded image
- diagnosis
- confidence
- class probabilities
- recommended action

## Added — Mid-Season API

Added:

```http
POST /predict-midseason
```

---

# [Milestone] — AgriSenseAI v2 Farmer System

## Added

Integrated:

```text
api/main.py
api/schema.py
api/feature_engineering.py
frontend/app.py
requirements.txt
README_V2.md
```

The v2 architecture combines FastAPI, Streamlit, the ExtraTrees yield model, MobileNetV2 CNN, GPS, environmental/soil lookup, recommendations and risks.

---

# [Milestone] — Baseline Yield Model

## Added

Created:

```text
train_agrisense_baseline.py
```

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

Saved:

```text
models/agrisense/baseline/agrisense_baseline_model.joblib
```

Five-fold validation:

```text
MAE  = 0.8663 MT/ha
RMSE = 1.1598 MT/ha
R²   = 0.2411
```

---

# [Milestone] — Production Yield Model

## Added

Created:

```text
train_agrisense_production.py
```

Model size:

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

Saved:

```text
models/agrisense/agrisense_production_model.joblib
```

The production model uses later seasonal environmental information and is therefore not treated as a strict pre-planting model.

---

# [Milestone] — Management Model

## Added

Created:

```text
train_agrisense_management.py
```

Validation:

```text
MAE  = 0.8848 MT/ha
RMSE = 1.1761 MT/ha
R²   = 0.2194
```

Saved:

```text
models/agrisense/management/agrisense_management_model.joblib
```

---

# [Milestone] — Zambia Master Dataset

## Added

Created:

```text
data/agrisense_master.csv
```

Shape:

```text
20,502 rows × 365 columns
```

Also created:

```text
data/agrisense_training_ready.csv
```

The master dataset combines agricultural management, crop, location, soil, weather, environmental, topographic, seasonal and yield information.

---

# [Milestone] — Zambia Agricultural Data Integration

## Added

Integrated:

```text
data/zamstats/Crop SM Final.sav
```

Approximate source size:

```text
32,031 rows
124 columns
```

Target crops:

```text
101 = Maize
503 = Groundnuts
```

Target records:

```text
17,035 Maize
3,467 Groundnuts
20,502 total
```

---

# [Milestone] — Location Matching

## Added

Created Zambia agricultural location matching.

Results:

```text
686 raw location rows
≈108 unique valid coordinate locations
18,893 / 20,502 records matched
≈92.15% match rate
```

Unmatched records were not assigned fabricated coordinates.

---

# [Milestone] — Soil Integration

## Added

Integrated iSDAsoil information.

Created:

```text
data/zambia_soil_features.csv
data/zambia_soil_features_clean.csv
```

Variables include:

- pH
- organic carbon
- total nitrogen
- phosphorus
- potassium
- CEC
- bulk density
- clay
- sand
- silt
- stone
- bedrock depth

Depths include 0–20 cm and 20–50 cm where available.

---

# [Milestone] — Environmental Integration

## Added — NASA POWER

Integrated daily weather data covering 2010–2025.

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

## Added — CHIRPS

Integrated rainfall features:

```text
chirps_mean_pentad
chirps_total_2010_2025
season_total_rainfall
season_mean_daily_rainfall
```

## Added — NDVI

Integrated:

```text
ndvi_max_2010_2025
ndvi_mean_2010_2025
```

## Added — Topography

Integrated:

```text
elevation_m
slope_degrees
```

---

# [Milestone] — Seasonal Environment Dataset

## Added

Created:

```text
data/zambia_seasonal_environment_features.csv
```

Season windows:

```text
Establishment:
2024-10-01 → 2024-11-30

Early growth:
2024-12-01 → 2025-01-31

Mid growth:
2025-02-01 → 2025-03-31

Late growth:
2025-04-01 → 2025-05-31
```

---

# [Milestone] — Yield Target Validation

## Changed

Standardized the target to:

```text
Yield
```

corresponding to:

```text
YIELDMT
```

Stopped reconstructing yield from production and harvested area because very small harvested-area values produced unrealistic calculated yields.

Recorded overall target statistics:

```text
Mean   = 1.4811 MT/ha
Std    = 1.3345
Median = 1.1500 MT/ha
Max    = 5.9630 MT/ha
```

---

# [Milestone] — ZamStats Metadata Validation

## Added

Validated:

```text
REGION
1 = Rural
2 = Urban
```

Crop codes:

```text
101 = Maize
503 = Groundnuts
```

`TCP05` was confirmed as planting timing.

Observed planting months include:

- October 2024
- November 2024
- December 2024
- January 2025
- February 2025
- March 2025

---

# [Milestone] — Corrected Feature Interpretation

## Fixed

An earlier implementation incorrectly treated:

```text
TCP05
```

as soil pH.

Correct:

```text
TCP05 = planting timing
```

## Fixed

An earlier implementation treated:

```text
TCP12_N
```

as fertilizer amount.

This was corrected.

`TCP12_N` is associated with seed retained for future planting.

Fertilizer-related source groups include:

```text
TCP01
TCP02
TCP03
```

---

# [Milestone] — Initial Streamlit Prototype

## Added

The first Streamlit prototype included:

- crop selection
- farm size
- GPS/location
- map
- planting information
- water source
- fertilizer
- soil pH
- yield prediction
- mid-season image upload

---

# [Milestone] — CNN Vision System

## Added

Implemented crop-leaf classification using:

```text
MobileNetV2
224 × 224 RGB
```

Maize classes:

```text
common_rust
corn_leaf_blight
gray_leaf_spot
healthy
insect_damage
```

Groundnut classes:

```text
early_leaf_spot
healthy
late_leaf_spot
nutrition_deficiency
rust
```

---

# [Milestone] — CNN Architecture Correction

## Changed

Removed arbitrary disease-to-yield multipliers such as:

```text
healthy = 1.00
disease = 0.70
other = 0.85
```

The vision system now keeps diagnosis separate from yield prediction.

The CNN returns:

```text
Diagnosis
Confidence
Probabilities
Action
```

---

# [Milestone] — Farmer Decision Support

## Changed

The project evolved from:

```text
Input → Prediction
```

to:

```text
Input
  ↓
Prediction
  ↓
Harvest estimate
  ↓
Recommendations
  ↓
Risks
```

This marks the transition from a model demonstration to an agricultural decision-support prototype.

---

# Known Issues

## Temporal model limitation

The current feature set contains seasonal environmental information that may become available only after planting.

Future models should only use information available at prediction time.

## Soil matching

Not every location has a direct soil observation.

The system should not fabricate soil data.

A strict geographic matching threshold should be enforced in a future version.

## Management data

Exact labels for some fertilizer/management categories still need to be mapped directly from source metadata.

## Farmer inputs

The frontend collects variety, previous crop, water source, fertilizer type, fertilizer amount and soil type, but the baseline model does not currently learn all of these variables.

## Recommendation causality

Current recommendations are based on model sensitivity/counterfactual scenarios and should not be interpreted as controlled agricultural experiments.

## CNN validation

The CNN requires additional validation using representative Zambia field images.

---

# Future Development

- [ ] True pre-planting model
- [ ] Time-aware mid-season model
- [ ] Yield uncertainty intervals
- [ ] Better soil matching
- [ ] Fertilizer metadata mapping
- [ ] Management-response model
- [ ] Farmer history
- [ ] Seasonal farm tracking
- [ ] Model monitoring
- [ ] Improved CNN validation
- [ ] Zambia field-image dataset
- [ ] Database
- [ ] Authentication
- [ ] Deployment
- [ ] Offline/low-connectivity support

---

# Project Evolution

```text
DATA COLLECTION
      ↓
DATA CLEANING
      ↓
ZAMBIA LOCATION MATCHING
      ↓
SOIL INTEGRATION
      ↓
WEATHER INTEGRATION
      ↓
ENVIRONMENTAL INTEGRATION
      ↓
MASTER DATASET
      ↓
YIELD MODEL
      ↓
CNN MODEL
      ↓
FASTAPI BACKEND
      ↓
STREAMLIT FRONTEND
      ↓
GPS + MAP
      ↓
RECOMMENDATIONS
      ↓
RISK ENGINE
      ↓
INTEGRATED FARMER SYSTEM
      ↓
TIME-AWARE AGRICULTURAL INTELLIGENCE
```

---

# Current Release State

AgriSenseAI is an end-to-end Zambia agricultural decision-support prototype for Maize and Groundnuts.

The project connects:

```text
Real agricultural data
        +
Environmental data
        +
Soil data
        +
Machine learning
        +
Computer vision
        +
GPS
        +
FastAPI
        +
Farmer interface
        +
Recommendations
        +
Risk tracking
```

The next major milestone is a strictly time-aware forecasting architecture separating pre-planting, mid-season, and later-season information.
