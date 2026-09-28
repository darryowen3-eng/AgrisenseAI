from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL = joblib.load(BASE_DIR / "models/agrisense/baseline/agrisense_baseline_model.joblib")

NUMERIC_FEATURES = [
"AREA_PLANTED","CR05_AREA","PROV","REGION","latitude","longitude",
"soil_ph_0_20","soil_ph_20_50","soil_organic_carbon_gkg_0_20","soil_organic_carbon_gkg_20_50",
"soil_total_nitrogen_gkg_0_20","soil_total_nitrogen_gkg_20_50",
"soil_phosphorus_ppm_0_20","soil_phosphorus_ppm_20_50",
"soil_potassium_ppm_0_20","soil_potassium_ppm_20_50",
"soil_clay_pct_0_20","soil_sand_pct_0_20","soil_silt_pct_0_20",
"soil_bulk_density_gcm3_0_20","soil_cec_cmolkg_0_20","soil_stone_pct_0_20",
"elevation_m","slope_degrees","chirps_mean_pentad","chirps_total_2010_2025",
"ndvi_max_2010_2025","ndvi_mean_2010_2025","season_total_rainfall",
"season_mean_daily_rainfall","TCP05"
]
CATEGORICAL_FEATURES = [
"crop","DIST","CONST","WARD","province","district_name","district_key","ward_environment"
]
MODEL_INPUT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

PROVINCE_TO_CODE = {
"Central":1,"Copperbelt":2,"Eastern":3,"Luapula":4,"Lusaka":5,
"Muchinga":6,"Northern":7,"North Western":8,"Southern":9,"Western":10
}
REGION_TO_CODE = {"Rural":1,"Urban":2}
CROP_NAMES = {"maize":"Maize","groundnut":"Groundnuts","groundnuts":"Groundnuts"}
PLANTING_MONTH_TO_CODE = {
"October 2024":1,"November 2024":2,"December 2024":3,
"January 2025":4,"February 2025":5,"March 2025":6
}

FILES = {
"environment": BASE_DIR/"data/zambia_environment_features.csv",
"seasonal": BASE_DIR/"data/zambia_seasonal_environment_features.csv",
"soil": BASE_DIR/"data/zambia_soil_features_clean.csv",
"topography": BASE_DIR/"data/zambia_topography.csv"
}

def _csv(name):
    p = FILES[name]
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def _nearest(df, lat, lon):
    if df.empty or not {"latitude","longitude"}.issubset(df.columns):
        return None
    d = df.copy()
    d["latitude"] = pd.to_numeric(d["latitude"], errors="coerce")
    d["longitude"] = pd.to_numeric(d["longitude"], errors="coerce")
    d = d.dropna(subset=["latitude","longitude"])
    if d.empty:
        return None
    dist = (d.latitude-lat)**2 + (d.longitude-lon)**2
    return d.loc[dist.idxmin()]

def _float(v):
    if v is None:
        return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan

def build_model_features(
    latitude, longitude, crop, province, region, planting_month,
    farm_size_ha, soil_ph=None, fertilizer_amount_kg_ha=None,
    district=None, ward=None
):
    crop_key = str(crop).strip().lower()
    if crop_key not in CROP_NAMES:
        raise ValueError("crop must be Maize or Groundnuts")
    if province not in PROVINCE_TO_CODE:
        raise ValueError("Invalid Zambia province")
    if region not in REGION_TO_CODE:
        raise ValueError("region must be Rural or Urban")
    if planting_month not in PLANTING_MONTH_TO_CODE:
        raise ValueError("Invalid planting month")

    lat, lon = float(latitude), float(longitude)
    ref = {}
    for name in ["environment","seasonal","soil","topography"]:
        row = _nearest(_csv(name), lat, lon)
        if row is not None:
            ref.update(row.to_dict())

    ref.update({
        "latitude": lat,
        "longitude": lon,
        "AREA_PLANTED": float(farm_size_ha),
        "CR05_AREA": float(farm_size_ha),
        "PROV": float(PROVINCE_TO_CODE[province]),
        "REGION": float(REGION_TO_CODE[region]),
        "TCP05": float(PLANTING_MONTH_TO_CODE[planting_month]),
        "crop": CROP_NAMES[crop_key],
        "province": province,
    })
    if soil_ph is not None:
        ref["soil_ph_0_20"] = float(soil_ph)
        ref["soil_ph_20_50"] = float(soil_ph)
    if district:
        ref["DIST"] = district
        ref["district_name"] = district
        ref["district_key"] = district
    if ward:
        ref["WARD"] = ward

    row = {}
    for c in NUMERIC_FEATURES:
        row[c] = _float(ref.get(c))
    for c in CATEGORICAL_FEATURES:
        v = ref.get(c)
        row[c] = None if v is None or (isinstance(v,float) and math.isnan(v)) else v
    return pd.DataFrame([row], columns=MODEL_INPUT_FEATURES)

def missing_model_features(df):
    return [c for c in MODEL_INPUT_FEATURES if c not in df or pd.isna(df.iloc[0][c])]
