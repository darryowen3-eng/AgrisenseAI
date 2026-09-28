from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent

DATA = BASE / "data" / "agrisense_master.csv"

MODEL_DIR = BASE / "models" / "agrisense" / "baseline"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA)

print("=" * 70)
print("AGRISENSEAI BASELINE MODEL")
print("=" * 70)

print("Dataset:", df.shape)


# ============================================================
# TARGET
# ============================================================

TARGET_CANDIDATES = [
    "Yield",
    "YIELDMT",
    "yield_tonnes_per_ha",
]

target = None

for col in TARGET_CANDIDATES:

    if col in df.columns:
        target = col
        break


if target is None:

    raise ValueError(
        "Could not find yield target."
    )


print("Target:", target)


# ============================================================
# ONLY FARMER-AVAILABLE / PRE-PLANTING VARIABLES
# ============================================================

candidate_features = [

    # Crop
    "crop",

    # Farm size
    "AREA_PLANTED",
    "CR05_AREA",

    # Location
    "PROV",
    "DIST",
    "CONST",
    "WARD",
    "REGION",

    "province",
    "district_name",
    "district_key",
    "ward_environment",

    "latitude",
    "longitude",

    # Soil
    "soil_ph_0_20",
    "soil_ph_20_50",

    "soil_organic_carbon_gkg_0_20",
    "soil_organic_carbon_gkg_20_50",

    "soil_total_nitrogen_gkg_0_20",
    "soil_total_nitrogen_gkg_20_50",

    "soil_phosphorus_ppm_0_20",
    "soil_phosphorus_ppm_20_50",

    "soil_potassium_ppm_0_20",
    "soil_potassium_ppm_20_50",

    "soil_clay_pct_0_20",
    "soil_sand_pct_0_20",
    "soil_silt_pct_0_20",

    "soil_bulk_density_gcm3_0_20",

    "soil_cec_cmolkg_0_20",

    "soil_stone_pct_0_20",

    # Topography
    "elevation_m",
    "slope_degrees",

    # Historical/environmental information
    "chirps_mean_pentad",
    "chirps_total_2010_2025",

    "ndvi_max_2010_2025",
    "ndvi_mean_2010_2025",

    # Pre-plant seasonal weather
    "establishment_mean_temp",
    "establishment_total_rainfall",
    "establishment_mean_daily_rainfall",

    "season_total_rainfall",
    "season_mean_daily_rainfall",

    # Planting timing if available
    "TCP05",
]


features = [
    c for c in candidate_features
    if c in df.columns
]


print("\nFeatures found:")
for f in features:
    print(" ", f)

print("\nFeature count:", len(features))


# ============================================================
# REMOVE TARGET / LEAKAGE
# ============================================================

LEAKAGE = {

    target,

    "YIELDFLAG",
    "ProductionMT",
    "Production_Kg",

    "AREA_HARVESTED",
    "TCP_AREA_HARVESTED_HA",

    "TCP09_QTY",
    "TCP09_UNIT",
    "TCP09_UNIT_OTHER",

    "TCP08__1",
    "TCP08__2",
    "TCP08__3",
    "TCP08__4",
    "TCP08__5",
    "TCP08__6",
    "TCP08__7",
    "TCP08__8",
    "TCP08__9",
    "TCP08__10",
    "TCP08__11",
    "TCP08__12",

    "TCP10a",
    "TCP10_UNIT",
    "TCP10b",
    "TCP11",

    "TCP07",

    "REPLANTED_KG",
    "REPLANTED_MT",
}


features = [
    c for c in features
    if c not in LEAKAGE
]


# ============================================================
# DATA
# ============================================================

work = df[
    features + [target]
].copy()


work[target] = pd.to_numeric(
    work[target],
    errors="coerce"
)


work = work[
    work[target].notna()
].copy()


work = work[
    np.isfinite(
        work[target]
    )
].copy()


print(
    "\nTraining rows:",
    len(work)
)


# ============================================================
# X / y
# ============================================================

X = work[features]

y = work[target]


# ============================================================
# GROUPS
# ============================================================

group_candidates = [
    "district_key",
    "DIST",
    "district_name",
    "WARD",
    "latitude",
]


group_col = None

for col in group_candidates:

    if col in X.columns:

        group_col = col
        break


if group_col is None:

    groups = np.arange(len(X))

else:

    groups = (
        X[group_col]
        .fillna("UNKNOWN")
        .astype(str)
    )


# ============================================================
# NUMERIC / CATEGORICAL
# ============================================================

numeric_features = [
    c for c in features
    if pd.api.types.is_numeric_dtype(
        X[c]
    )
]


categorical_features = [
    c for c in features
    if c not in numeric_features
]


print(
    "\nNumeric:",
    len(numeric_features)
)

print(
    "Categorical:",
    len(categorical_features)
)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
        ),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# ============================================================
# MODEL
# ============================================================

regressor = ExtraTreesRegressor(
    n_estimators=500,
    min_samples_leaf=3,
    max_features=0.8,
    random_state=42,
    n_jobs=-1,
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            regressor,
        ),
    ]
)


# ============================================================
# GROUP CROSS VALIDATION
# ============================================================

gkf = GroupKFold(
    n_splits=5
)


fold_results = []


for fold, (
    train_idx,
    test_idx
) in enumerate(
    gkf.split(
        X,
        y,
        groups
    ),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    fold_results.append(
        {
            "fold": fold,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }
    )

    print(
        f"\nFold {fold}"
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R2   : {r2:.4f}"
    )


# ============================================================
# FINAL MODEL
# ============================================================

pipeline.fit(
    X,
    y
)


# ============================================================
# SAVE
# ============================================================

model_path = (
    MODEL_DIR /
    "agrisense_baseline_model.joblib"
)


joblib.dump(
    pipeline,
    model_path
)


features_path = (
    MODEL_DIR /
    "agrisense_baseline_features.json"
)


with open(
    features_path,
    "w"
) as f:

    json.dump(
        {
            "target": target,
            "features": features,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "group_column": group_col,
        },
        f,
        indent=2,
    )


metrics = {

    "model":
        "ExtraTreesRegressor",

    "target":
        target,

    "rows":
        len(work),

    "features":
        len(features),

    "folds":
        fold_results,

    "mean_mae":
        float(
            np.mean(
                [
                    x["MAE"]
                    for x in fold_results
                ]
            )
        ),

    "mean_rmse":
        float(
            np.mean(
                [
                    x["RMSE"]
                    for x in fold_results
                ]
            )
        ),

    "mean_r2":
        float(
            np.mean(
                [
                    x["R2"]
                    for x in fold_results
                ]
            )
        ),
}


with open(
    MODEL_DIR /
    "agrisense_baseline_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )


print("\n" + "=" * 70)

print(
    "BASELINE MODEL SAVED"
)

print(
    model_path
)

print(
    "\nMean MAE:",
    metrics["mean_mae"]
)

print(
    "Mean RMSE:",
    metrics["mean_rmse"]
)

print(
    "Mean R2:",
    metrics["mean_r2"]
)

print("=" * 70)
