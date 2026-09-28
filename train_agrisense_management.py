import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/home/trojan-horse/AgriSenseAI"

DATA = os.path.join(
    BASE_DIR,
    "data",
    "csvs",
    "agrisense_master.csv"
)

# Fallback if your master is directly under data/
if not os.path.exists(DATA):
    DATA = os.path.join(
        BASE_DIR,
        "data",
        "agrisense_master.csv"
    )

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models",
    "agrisense",
    "management"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("AGRISENSEAI MANAGEMENT MODEL")
print("=" * 70)

print(f"Dataset: {DATA}")

df = pd.read_csv(
    DATA,
    low_memory=False
)

print(f"Dataset shape: {df.shape}")


# ============================================================
# TARGET
# ============================================================

TARGET = "Yield"

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found."
    )

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

df = df.dropna(
    subset=[TARGET]
).copy()


# ============================================================
# BASELINE FEATURES
# ============================================================

BASELINE_FEATURES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "agrisense",
    "baseline",
    "agrisense_baseline_features.json"
)

with open(BASELINE_FEATURES_PATH, "r") as f:
    baseline_features = json.load(f)


# ============================================================
# MANAGEMENT FEATURES
# ============================================================

management_features = []


# Organic fertilizer
for col in df.columns:

    if col.startswith("TCP01__"):
        management_features.append(col)


# Inorganic basal fertilizer
for col in df.columns:

    if col.startswith("TCP02__"):
        management_features.append(col)


# Top dressing
for col in df.columns:

    if col.startswith("TCP03__"):
        management_features.append(col)


# Pesticides
for col in df.columns:

    if col.startswith("TCP04__"):
        management_features.append(col)


# Planting timing
if "TCP05" in df.columns:
    management_features.append("TCP05")


# Replanting
for col in [
    "replanted",
    "REPLANTED_KG",
    "REPLANTED_MT",
    "RETENTIONFLAG"
]:

    if col in df.columns:
        management_features.append(col)


# Existing fertilizer indicator
if "fertilizer_any" in df.columns:
    management_features.append(
        "fertilizer_any"
    )


# Remove duplicates
management_features = list(
    dict.fromkeys(management_features)
)


print("\nManagement features:")

for feature in management_features:
    print(" ", feature)

print(
    f"\nManagement feature count: "
    f"{len(management_features)}"
)


# ============================================================
# COMBINE
# ============================================================

features = []

for feature in baseline_features:

    if feature in df.columns:
        features.append(feature)


for feature in management_features:

    if feature in df.columns:
        if feature not in features:
            features.append(feature)


# Remove obvious target / harvest leakage
LEAKAGE = {

    "Yield",
    "YIELDMT",
    "ProductionMT",
    "Production_Kg",

    "AREA_HARVESTED",
    "TCP_AREA_HARVESTED_HA",

    "TCP09_QTY",
    "TCP09_UNIT",

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
    "TCP10b",
    "TCP11",

    "ProductionMT",
    "Production_Kg",

    "REPLANTED_KG",
    "REPLANTED_MT"
}


features = [
    f for f in features
    if f not in LEAKAGE
]


print(
    f"\nFinal feature count: "
    f"{len(features)}"
)


# ============================================================
# GROUPS
# ============================================================

if "district_key" in df.columns:

    groups = (
        df["district_key"]
        .fillna("unknown")
        .astype(str)
    )

elif "DIST" in df.columns:

    groups = (
        df["DIST"]
        .fillna("unknown")
        .astype(str)
    )

else:

    groups = pd.Series(
        np.arange(len(df))
    )


# ============================================================
# X / y
# ============================================================

X = df[features].copy()
y = df[TARGET].astype(float)


numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    c for c in X.columns
    if c not in numeric_features
]


print(f"Training rows: {len(X)}")
print(f"Numeric: {len(numeric_features)}")
print(f"Categorical: {len(categorical_features)}")


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# MODEL
# ============================================================

model = ExtraTreesRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=3,
    max_features=0.8
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# CROSS VALIDATION
# ============================================================

n_groups = groups.nunique()

n_splits = min(
    5,
    n_groups
)

cv = GroupKFold(
    n_splits=n_splits
)


fold_results = []

print("\n" + "=" * 70)
print("CROSS VALIDATION")
print("=" * 70)


for fold, (train_idx, val_idx) in enumerate(
    cv.split(
        X,
        y,
        groups=groups
    ),
    start=1
):

    print(f"\nFold {fold}")

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_val
    )

    predictions = np.maximum(
        predictions,
        0
    )

    mae = mean_absolute_error(
        y_val,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            predictions
        )
    )

    r2 = r2_score(
        y_val,
        predictions
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

    fold_results.append(
        {
            "fold": fold,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        }
    )


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MANAGEMENT MODEL")
print("=" * 70)

pipeline.fit(
    X,
    y
)


# ============================================================
# METRICS
# ============================================================

mean_mae = np.mean([
    x["mae"]
    for x in fold_results
])

mean_rmse = np.mean([
    x["rmse"]
    for x in fold_results
])

mean_r2 = np.mean([
    x["r2"]
    for x in fold_results
])


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "agrisense_management_model.joblib"
)

FEATURES_PATH = os.path.join(
    OUTPUT_DIR,
    "agrisense_management_features.json"
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "agrisense_management_metrics.json"
)


import joblib

joblib.dump(
    pipeline,
    MODEL_PATH
)


with open(
    FEATURES_PATH,
    "w"
) as f:

    json.dump(
        features,
        f,
        indent=2
    )


metrics = {

    "model": "ExtraTreesRegressor",

    "training_rows":
        int(len(X)),

    "feature_count":
        int(len(features)),

    "management_feature_count":
        int(len(management_features)),

    "mean_mae":
        float(mean_mae),

    "mean_rmse":
        float(mean_rmse),

    "mean_r2":
        float(mean_r2),

    "folds":
        fold_results
}


with open(
    METRICS_PATH,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("MANAGEMENT MODEL SAVED")
print("=" * 70)

print(MODEL_PATH)

print(
    f"\nMean MAE  : {mean_mae:.4f}"
)

print(
    f"Mean RMSE : {mean_rmse:.4f}"
)

print(
    f"Mean R2   : {mean_r2:.4f}"
)

print("=" * 70)
