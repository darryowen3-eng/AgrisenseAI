from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "agrisense"
    / "baseline"
    / "agrisense_baseline_model.joblib"
)

FEATURES_PATH = (
    BASE_DIR
    / "models"
    / "agrisense"
    / "baseline"
    / "agrisense_baseline_features.json"
)


# =========================================================
# LOAD
# =========================================================

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    feature_info = json.load(f)


if isinstance(feature_info, dict):
    FEATURES = feature_info.get("features", [])
else:
    FEATURES = feature_info


if not FEATURES:
    FEATURES = list(model.feature_names_in_)


# =========================================================
# GET ACTUAL FEATURES FROM TRAINED PIPELINE
# =========================================================

preprocessor = model.named_steps["preprocessor"]

NUMERIC_FEATURES = []
CATEGORICAL_FEATURES = []

for name, transformer, columns in preprocessor.transformers_:

    if name == "numeric":
        NUMERIC_FEATURES = list(columns)

    elif name == "categorical":
        CATEGORICAL_FEATURES = list(columns)


print("=" * 70)
print("AGRISENSE BASELINE MODEL")
print("=" * 70)
print("Total features:", len(FEATURES))
print("Numeric:", len(NUMERIC_FEATURES))
print("Categorical:", len(CATEGORICAL_FEATURES))
print("=" * 70)


# =========================================================
# PREPARE ROW
# =========================================================

def prepare_row(feature_dict):

    row = {}

    for feature in FEATURES:

        value = feature_dict.get(feature)

        # ---------------------------------------------
        # NUMERIC
        # ---------------------------------------------

        if feature in NUMERIC_FEATURES:

            if value is None or value == "":
                row[feature] = np.nan

            else:
                try:
                    row[feature] = float(value)
                except (ValueError, TypeError):

                    raise ValueError(
                        f"Feature '{feature}' must be numeric, "
                        f"but received: {value!r}"
                    )

        # ---------------------------------------------
        # CATEGORICAL
        # ---------------------------------------------

        elif feature in CATEGORICAL_FEATURES:

            if value is None or str(value).strip() == "":
                row[feature] = "Unknown"
            else:
                row[feature] = str(value).strip()

        # ---------------------------------------------
        # FALLBACK
        # ---------------------------------------------

        else:

            row[feature] = value

    return pd.DataFrame(
        [row],
        columns=FEATURES
    )


# =========================================================
# PREDICT
# =========================================================

def predict_yield(feature_dict):

    input_df = prepare_row(feature_dict)

    print("\n" + "=" * 70)
    print("AGRISENSE MODEL INPUT")
    print("=" * 70)

    print(input_df.T.to_string())

    print("=" * 70)

    prediction = model.predict(input_df)[0]

    prediction = float(prediction)

    return max(0.0, prediction)
