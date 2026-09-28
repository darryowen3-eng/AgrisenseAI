# api/scenarios.py

import json
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(
    __file__
).resolve().parent.parent


MODEL_PATH = (
    BASE_DIR
    / "models"
    / "agrisense"
    / "management"
    / "agrisense_management_model.joblib"
)

FEATURES_PATH = (
    BASE_DIR
    / "models"
    / "agrisense"
    / "management"
    / "agrisense_management_features.json"
)


management_model = joblib.load(
    MODEL_PATH
)


with open(
    FEATURES_PATH,
    "r"
) as f:

    MANAGEMENT_FEATURES = json.load(f)


print(
    f"Management features: "
    f"{len(MANAGEMENT_FEATURES)}"
)


def prepare_management_row(
    features
):

    row = {}

    for feature in MANAGEMENT_FEATURES:

        row[feature] = features.get(
            feature,
            None
        )

    return pd.DataFrame(
        [row],
        columns=MANAGEMENT_FEATURES
    )


def predict_management(
    features
):

    df = prepare_management_row(
        features
    )

    prediction = (
        management_model
        .predict(df)[0]
    )

    return max(
        0.0,
        float(prediction)
    )


# ============================================================
# MANAGEMENT FEATURE RESET
# ============================================================

def clear_management_features(
    features
):

    result = features.copy()

    for feature in MANAGEMENT_FEATURES:

        if (
            feature.startswith("TCP01__")
            or feature.startswith("TCP02__")
            or feature.startswith("TCP03__")
            or feature.startswith("TCP04__")
        ):

            result[feature] = 0

    if "fertilizer_any" in result:

        result["fertilizer_any"] = 0

    return result


# ============================================================
# FIND FERTILIZER FEATURE
# ============================================================

def find_feature(
    prefix,
    keywords
):

    if isinstance(
        keywords,
        str
    ):

        keywords = [keywords]

    keywords = [
        str(k).lower()
        for k in keywords
    ]

    for feature in MANAGEMENT_FEATURES:

        if not feature.startswith(prefix):
            continue

        lower = feature.lower()

        for keyword in keywords:

            if keyword in lower:
                return feature

    return None


# ============================================================
# CREATE SCENARIO
# ============================================================

def create_scenario(
    base_features,
    fertilizer_type
):

    features = (
        clear_management_features(
            base_features
        )
    )

    fertilizer = (
        str(fertilizer_type)
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # None
    # --------------------------------------------------------

    if fertilizer == "none":

        return features

    # --------------------------------------------------------
    # Organic
    # --------------------------------------------------------

    if fertilizer == "organic compost":

        feature = find_feature(
            "TCP01__",
            [
                "organic"
            ]
        )

        if feature:

            features[feature] = 1

        if "fertilizer_any" in features:

            features[
                "fertilizer_any"
            ] = 1

        return features

    # --------------------------------------------------------
    # D compound
    # --------------------------------------------------------

    if fertilizer in [
        "d-compound",
        "compound d"
    ]:

        feature = find_feature(
            "TCP02__",
            [
                "compound d"
            ]
        )

        if feature:

            features[feature] = 1

        if "fertilizer_any" in features:

            features[
                "fertilizer_any"
            ] = 1

        return features

    # --------------------------------------------------------
    # Compound B
    # --------------------------------------------------------

    if fertilizer == "compound b":

        feature = find_feature(
            "TCP02__",
            [
                "compound b"
            ]
        )

        if feature:

            features[feature] = 1

        if "fertilizer_any" in features:

            features[
                "fertilizer_any"
            ] = 1

        return features

    # --------------------------------------------------------
    # Generic inorganic
    # --------------------------------------------------------

    if fertilizer == "inorganic":

        feature = find_feature(
            "TCP02__",
            [
                "compound"
            ]
        )

        if feature:

            features[feature] = 1

        if "fertilizer_any" in features:

            features[
                "fertilizer_any"
            ] = 1

        return features

    return features


# ============================================================
# COMPARE
# ============================================================

def compare_scenarios(
    base_features
):

    scenarios = [

        "None",

        "Organic Compost",

        "D-Compound",

        "Compound B",

        "Inorganic"
    ]

    results = []

    for scenario in scenarios:

        scenario_features = (
            create_scenario(
                base_features,
                scenario
            )
        )

        prediction = predict_management(
            scenario_features
        )

        results.append(
            {
                "scenario":
                    scenario,

                "predicted_yield_t_ha":
                    round(
                        prediction,
                        3
                    )
            }
        )

    return sorted(
        results,
        key=lambda x:
            x[
                "predicted_yield_t_ha"
            ],
        reverse=True
    )
