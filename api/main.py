from pathlib import Path
from functools import lru_cache
import io, json
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from .feature_engineering import MODEL, build_model_features, missing_model_features
from .schema import PredictRequest, PredictionResponse

BASE_DIR = Path(__file__).resolve().parents[1]
app = FastAPI(title="AgriSenseAI", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

Q1, MEDIAN, Q3 = 0.448, 1.150, 2.271605
MONTHS = ["October 2024","November 2024","December 2024",
          "January 2025","February 2025","March 2025"]

@app.get("/")
def home():
    return {"name":"AgriSenseAI","status":"running","version":"2.0"}

@app.get("/health")
def health():
    return {"status":"ok","yield_model_loaded":MODEL is not None}

def band(y):
    if y < Q1: return "Low model estimate"
    if y <= Q3: return "Middle training-data range"
    return "High model estimate"

def predict_for(r, crop=None, month=None, ph=None):
    f = build_model_features(
        r.latitude, r.longitude, crop or r.crop, r.province, r.region,
        month or r.planting_month, r.farm_size_ha,
        ph if ph is not None else r.soil_ph,
        r.fertilizer_amount_kg_ha, r.district, r.ward
    )
    return f, max(0.0, float(MODEL.predict(f)[0]))

def recommendations(r, base):
    recs, scenarios = [], []
    for m in MONTHS:
        if m == r.planting_month: continue
        try:
            _, y = predict_for(r, month=m)
            scenarios.append({"type":"planting_month","option":m,"yield":y,"delta":y-base})
        except Exception:
            pass

    better = sorted([x for x in scenarios if x["delta"] > 0],
                    key=lambda x:x["delta"], reverse=True)
    if better:
        x = better[0]
        recs.append({
            "title":"📅 Model-tested planting window",
            "message":f"For the same supplied farm conditions, the model estimates {x['option']} at {x['yield']:.2f} MT/ha versus {base:.2f} MT/ha for the selected month.",
            "type":"model_scenario"
        })
    else:
        recs.append({
            "title":"📅 Planting window",
            "message":"Among the other planting months represented in the model, none produced a higher model estimate for this scenario.",
            "type":"model_scenario"
        })

    other = "Groundnuts" if r.crop == "Maize" else "Maize"
    try:
        _, y = predict_for(r, crop=other)
        recs.append({
            "title":"🌱 Crop comparison",
            "message":f"Keeping the location and other supplied inputs the same, the model estimates {other} at {y:.2f} MT/ha. This is a model comparison, not a guarantee.",
            "type":"model_scenario"
        })
    except Exception:
        pass

    if r.soil_ph is not None:
        tests = []
        for ph in sorted({max(4.5,min(7.5,r.soil_ph-0.5)), max(4.5,min(7.5,r.soil_ph+0.5))}):
            try:
                _, y = predict_for(r, ph=ph)
                tests.append((ph,y))
            except Exception:
                pass
        if tests:
            ph,y = max(tests,key=lambda x:x[1])
            recs.append({
                "title":"🧪 Soil-pH sensitivity",
                "message":f"A model sensitivity test at pH {ph:.1f} gives {y:.2f} MT/ha. This is NOT a lime/fertilizer prescription; confirm soil-management decisions locally.",
                "type":"model_scenario"
            })

    recs.append({
        "title":"🧺 Fertilizer record",
        "message":"Fertilizer type and amount are recorded for the farm plan, but the current baseline yield model was not trained with those fields, so they do not artificially change the prediction.",
        "type":"model_coverage"
    })
    return recs, scenarios

def risks(r, f, y, scenarios):
    out, level = [], "LOW"
    missing = missing_model_features(f)
    soil = {"soil_ph_0_20","soil_organic_carbon_gkg_0_20",
            "soil_total_nitrogen_gkg_0_20","soil_phosphorus_ppm_0_20",
            "soil_potassium_ppm_0_20"}
    if soil.intersection(missing):
        level = "HIGH"
        out.append({"title":"🧪 Soil-data gap","severity":"high",
                    "message":"Some soil variables are unavailable at this location, so the model pipeline is imputing them."})
    if y < Q1:
        level = "HIGH"
        out.append({"title":"📉 Low model estimate","severity":"high",
                    "message":f"The estimate ({y:.2f} MT/ha) is below the training-data first quartile ({Q1:.3f})."})
    elif y < MEDIAN:
        level = "MEDIUM"
        out.append({"title":"📊 Yield pressure","severity":"medium",
                    "message":f"The estimate ({y:.2f} MT/ha) is below the training-data median ({MEDIAN:.2f})."})
    months = [x["yield"] for x in scenarios if x["type"]=="planting_month"]
    if months and max(months)-min(months) >= 0.5:
        level = "MEDIUM" if level=="LOW" else level
        out.append({"title":"📅 Timing sensitivity","severity":"medium",
                    "message":"The model changes noticeably across the tested planting months for this scenario."})
    out.append({"title":"🌦️ Weather-data limitation","severity":"info",
                "message":"This baseline model uses the environmental features available in its training pipeline; it is not a live weather forecast."})
    return level, out

@app.post("/predict", response_model=PredictionResponse)
def predict(r: PredictRequest):
    try:
        f,y = predict_for(r)
        recs,scenarios = recommendations(r,y)
        level,rs = risks(r,f,y,scenarios)
        return PredictionResponse(
            predicted_yield_mt_per_ha=y,
            predicted_total_harvest_mt=y*r.farm_size_ha,
            yield_band=band(y),
            risk_level=level,
            risks=rs,
            recommendations=recs,
            model_features_missing=missing_model_features(f),
            context={"crop":r.crop,"latitude":r.latitude,"longitude":r.longitude,
                     "farm_size_ha":r.farm_size_ha,"soil_ph":r.soil_ph}
        )
    except Exception as e:
        raise HTTPException(400,str(e))

@lru_cache(maxsize=4)
def load_cnn(crop):
    import tensorflow as tf
    candidates = [
        BASE_DIR/f"models/agrisense/vision/{crop}/best_model.keras",
        BASE_DIR/f"models/agrisense/vision/{crop}/{crop}_vision_model.keras",
        BASE_DIR/f"models/vision/{crop}/best_model.keras",
        BASE_DIR/f"models/vision/{crop}/{crop}_vision_model.keras",
        BASE_DIR/f"models/agrisense/{crop}/best_model.keras",
    ]
    p = next((x for x in candidates if x.exists()), None)
    if not p:
        raise FileNotFoundError("CNN model not found. Checked: " + ", ".join(map(str,candidates)))
    model = tf.keras.models.load_model(p)
    classes = None
    for m in [p.parent/"class_names.json", p.parent/"model_metadata.json"]:
        if m.exists():
            meta=json.loads(m.read_text())
            classes = meta if isinstance(meta,list) else meta.get("class_names") or meta.get("classes")
            if classes: break
    if not classes:
        raise FileNotFoundError(f"No class_names.json/model_metadata.json beside {p}")
    return model,list(classes),p

def leaf_action(d):
    s=d.lower().replace("_"," ")
    if s=="healthy": return "🌿 The classifier sees a healthy class. Keep monitoring new growth."
    if "nutrition" in s: return "🧪 Possible nutrition deficiency. Check the field and soil/fertilizer record before applying an input."
    if "insect" in s: return "🐛 Possible insect damage. Inspect the underside of leaves and nearby plants before choosing control."
    if "rust" in s: return "🍂 Possible rust. Check nearby plants for spread and confirm locally before treatment."
    if "blight" in s or "spot" in s: return "🍃 Possible leaf-spot/blight condition. Inspect surrounding plants and confirm locally before treatment."
    return "🔎 Non-healthy class detected. Confirm the field diagnosis before treatment."

@app.post("/predict-midseason")
async def predict_midseason(crop: str, image: UploadFile = File(...)):
    key = crop.lower().replace("groundnuts","groundnut")
    if key not in {"maize","groundnut"}:
        raise HTTPException(400,"crop must be Maize or Groundnuts")
    try:
        model,classes,path=load_cnn(key)
        from PIL import Image
        raw=await image.read()
        img=Image.open(io.BytesIO(raw)).convert("RGB")
        x=np.asarray(img.resize((224,224)),dtype=np.float32)
        probs=np.asarray(model.predict(np.expand_dims(x,0),verbose=0)[0],dtype=float)
        probs=probs/probs.sum() if probs.sum() else probs
        i=int(np.argmax(probs))
        return {
            "crop":"Groundnuts" if key=="groundnut" else "Maize",
            "diagnosis":classes[i],
            "confidence":float(probs[i]),
            "probabilities":{c:float(probs[j]) for j,c in enumerate(classes)},
            "action":leaf_action(classes[i]),
            "model_file":str(path.relative_to(BASE_DIR))
        }
    except Exception as e:
        raise HTTPException(500,str(e))
