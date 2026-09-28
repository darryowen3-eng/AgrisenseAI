import os
import requests
import pandas as pd
import streamlit as st
from streamlit_js_eval import get_geolocation

st.set_page_config(page_title="AgriSenseAI",page_icon="🌾",layout="wide")
API=os.getenv("AGRISENSE_API_URL","http://127.0.0.1:8000")

for k,v in {"lat":-12.9587,"lon":28.6336,"gps":"Waiting for GPS…",
            "plan":None,"leaf_result":None}.items():
    st.session_state.setdefault(k,v)

try:
    loc=get_geolocation()
    if loc and "coords" in loc:
        st.session_state.lat=float(loc["coords"]["latitude"])
        st.session_state.lon=float(loc["coords"]["longitude"])
        st.session_state.gps="📍 GPS location detected"
    elif loc and "error" in loc:
        st.session_state.gps="GPS permission unavailable"
except Exception:
    st.session_state.gps="GPS unavailable — enter coordinates manually"

st.markdown("""
<style>
.hero{padding:30px;border-radius:24px;background:linear-gradient(135deg,#123d25,#3b7b4c);color:white;margin-bottom:20px}
.hero h1{font-size:44px;margin:0}.hero p{font-size:18px}
.rec{padding:16px;border-radius:14px;background:rgba(59,123,76,.10);margin:9px 0}
.risk-high{padding:15px;border-left:6px solid #d9534f;background:rgba(217,83,79,.09);border-radius:12px;margin:8px 0}
.risk-medium{padding:15px;border-left:6px solid #f0ad4e;background:rgba(240,173,78,.09);border-radius:12px;margin:8px 0}
.risk-info{padding:15px;border-left:6px solid #5bc0de;background:rgba(91,192,222,.09);border-radius:12px;margin:8px 0}
</style>
""",unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🌾 AgriSenseAI</h1><p>Turn your farm information into a practical crop plan.</p></div>',unsafe_allow_html=True)

st.subheader("📍 Farm location")
st.success(st.session_state.gps)
c1,c2=st.columns(2)
with c1: lat=st.number_input("Latitude",value=st.session_state.lat,format="%.6f")
with c2: lon=st.number_input("Longitude",value=st.session_state.lon,format="%.6f")
st.map(pd.DataFrame({"latitude":[lat],"longitude":[lon]}),latitude="latitude",longitude="longitude")

st.subheader("🌱 Farm & crop")
c1,c2,c3=st.columns(3)
with c1: crop=st.selectbox("Crop",["Maize","Groundnuts"])
with c2: province=st.selectbox("Province",["Central","Copperbelt","Eastern","Luapula","Lusaka","Muchinga","Northern","North Western","Southern","Western"])
with c3: region=st.selectbox("Area",["Rural","Urban"])
c1,c2,c3=st.columns(3)
with c1: size=st.number_input("Farm size (ha)",min_value=.01,value=2.0,step=.1)
with c2: district=st.text_input("District","")
with c3: ward=st.text_input("Ward","")

st.subheader("🧑🏾‍🌾 Field information")
c1,c2,c3=st.columns(3)
with c1: variety=st.selectbox("Crop variety / maturity",["Not specified","Early","Medium","Late"])
with c2: previous=st.selectbox("What did you grow here last time?",["Not specified","Maize","Groundnuts","Other"])
with c3: water=st.selectbox("Water source",["Rain-fed","Irrigated"])
c1,c2,c3=st.columns(3)
with c1: soil_type=st.selectbox("Soil type",["Not specified","Sandy","Clay","Loam","Sandy loam","Other"])
with c2: ph=st.number_input("Soil pH",min_value=0.,max_value=14.,value=5.8,step=.1)
with c3: month=st.selectbox("Main planting month",["October 2024","November 2024","December 2024","January 2025","February 2025","March 2025"],index=2)

st.subheader("🧺 Fertilizer")
c1,c2=st.columns(2)
with c1: fert_type=st.selectbox("Fertilizer type",["None / not specified","D-Compound","Urea","CAN","Other"])
with c2: fert_amount=st.number_input("Amount (kg/ha)",min_value=0.,value=200.,step=10.)

payload={
"latitude":lat,"longitude":lon,"crop":crop,"province":province,"region":region,
"planting_month":month,"farm_size_ha":size,"soil_ph":ph,
"fertilizer_type":fert_type,"fertilizer_amount_kg_ha":fert_amount,
"soil_type":None if soil_type=="Not specified" else soil_type,
"crop_variety":None if variety=="Not specified" else variety,
"previous_crop":None if previous=="Not specified" else previous,
"water_source":water,"district":district or None,"ward":ward or None}

if st.button("🚜 BUILD MY FARM PLAN",type="primary",use_container_width=True):
    try:
        r=requests.post(f"{API}/predict",json=payload,timeout=120)
        r.raise_for_status()
        st.session_state.plan=r.json()
    except Exception as e: st.error(f"Farm plan failed: {e}")

if st.session_state.plan:
    x=st.session_state.plan
    st.divider();st.subheader("🌾 Your Farm Plan")
    a,b,c=st.columns(3)
    a.metric("Predicted yield",f"{x['predicted_yield_mt_per_ha']:.2f} MT/ha")
    b.metric("Expected harvest",f"{x['predicted_total_harvest_mt']:.2f} MT")
    c.metric("Risk level",x["risk_level"])
    st.caption(x["yield_band"])

    st.subheader("🧠 Recommendations from the model")
    for r in x["recommendations"]:
        st.markdown(f'<div class="rec"><b>{r["title"]}</b><br>{r["message"]}</div>',unsafe_allow_html=True)

    st.subheader("⚠️ Risks to watch")
    for r in x["risks"]:
        st.markdown(f'<div class="risk-{r["severity"]}"><b>{r["title"]}</b><br>{r["message"]}</div>',unsafe_allow_html=True)

st.divider()
st.subheader("🍃 AgriSense Crop Doctor")
st.write("Upload a clear leaf photo. The image stays visible while the CNN returns the diagnosis, confidence and probability breakdown.")
leaf=st.file_uploader("📸 Upload leaf image",type=["jpg","jpeg","png"])

if leaf:
    left,right=st.columns([1,1])
    with left: st.image(leaf,caption="Uploaded leaf",use_container_width=True)
    with right:
        if st.button("🔍 ANALYZE LEAF",type="primary",use_container_width=True):
            try:
                with st.spinner("CNN is examining the leaf…"):
                    r=requests.post(f"{API}/predict-midseason",params={"crop":crop},
                                    files={"image":(leaf.name,leaf.getvalue(),leaf.type)},timeout=180)
                    r.raise_for_status()
                    st.session_state.leaf_result=r.json()
            except Exception as e: st.error(f"Leaf diagnosis failed: {e}")

if st.session_state.leaf_result:
    x=st.session_state.leaf_result
    st.success(f"Diagnosis: **{x['diagnosis'].replace('_',' ').title()}**")
    a,b=st.columns(2)
    a.metric("CNN confidence",f"{x['confidence']:.1%}")
    b.metric("Crop",x["crop"])
    st.markdown(f"### 🩺 What to do next\n{x['action']}")
    p=pd.DataFrame({"Condition":list(x["probabilities"].keys()),
                    "Probability":list(x["probabilities"].values())})
    st.bar_chart(p.set_index("Condition"))
