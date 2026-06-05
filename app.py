import streamlit as st
import pickle
import numpy as np
import pandas as pd
import google.generativeai as genai
import os
import joblib
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# ── Load model, scaler, features ───────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
                        
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def get_ai_advice(patient_data: dict, risk_probability: float) -> str:
    risk_level = "HIGH" if risk_probability >= 0.5 else "LOW"
    risk_pct   = round(risk_probability * 100, 1)
    prompt = f"""
You are a compassionate and professional medical AI assistant helping
healthcare providers understand patient readmission risk.

PATIENT PROFILE:
- Age group: {patient_data.get('age', 'Unknown')}
- Time in hospital: {patient_data.get('time_in_hospital', 'Unknown')} days
- Number of medications: {patient_data.get('num_medications', 'Unknown')}
- Number of diagnoses: {patient_data.get('number_diagnoses', 'Unknown')}
- Number of previous inpatient visits: {patient_data.get('number_inpatient', 'Unknown')}
- Number of emergency visits: {patient_data.get('number_emergency', 'Unknown')}
- Lab procedures done: {patient_data.get('num_lab_procedures', 'Unknown')}
- Diabetes medication: {patient_data.get('diabetesMed', 'Unknown')}
- A1C result: {patient_data.get('A1Cresult', 'Unknown')}
- Insulin: {patient_data.get('insulin', 'Unknown')}

ML MODEL PREDICTION:
- Readmission Risk: {risk_level}
- Probability of readmission within 30 days: {risk_pct}%

Please provide:
1. **Risk Summary** (2 sentences)
2. **Key Risk Factors** (bullet points)
3. **Recommended Actions** (3-5 steps for the care team)
4. **Patient Discharge Advice** (2-3 home tips)

Be specific to this patient's data. Respond as a clinical decision support tool.
"""
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    response     = gemini_model.generate_content(prompt)
    return response.text

# ── Encoding maps (must match preprocessing) ───────────────
AGE_MAP = {
    '[0-10)':0,'[10-20)':1,'[20-30)':2,'[30-40)':3,'[40-50)':4,
    '[50-60)':5,'[60-70)':6,'[70-80)':7,'[80-90)':8,'[90-100)':9
}
INSULIN_MAP   = {'No':0, 'Steady':1, 'Up':2, 'Down':3}
A1C_MAP       = {'None':0, 'Norm':1, '>7':2, '>8':3}
YESNO_MAP     = {'No':0, 'Yes':1}
CHANGE_MAP    = {'No':0, 'Ch':1}

# ── UI ──────────────────────────────────────────────────────
st.title("🏥 Diabetes Readmission Risk Predictor")
st.markdown("Predict whether a diabetic patient will be readmitted within **30 days** of discharge — powered by Random Forest + Gemini AI.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Patient Info")
    age = st.selectbox("Age Group", list(AGE_MAP.keys()), index=7)
    gender = st.selectbox("Gender", ["Male", "Female"])
    time_in_hospital = st.slider("Days in Hospital", 1, 14, 5)
    number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)

with col2:
    st.subheader("💊 Medications")
    num_medications   = st.slider("Number of Medications", 1, 81, 15)
    num_lab_procedures = st.slider("Lab Procedures", 1, 132, 45)
    num_procedures    = st.slider("Non-Lab Procedures", 0, 6, 1)
    insulin           = st.selectbox("Insulin", list(INSULIN_MAP.keys()))
    diabetesMed       = st.selectbox("On Diabetes Medication?", ["Yes", "No"])
    A1Cresult         = st.selectbox("A1C Result", list(A1C_MAP.keys()))

with col3:
    st.subheader("🏨 Visit History")
    number_inpatient  = st.slider("Previous Inpatient Visits", 0, 21, 1)
    number_emergency  = st.slider("Emergency Visits", 0, 76, 0)
    number_outpatient = st.slider("Outpatient Visits", 0, 42, 0)
    change            = st.selectbox("Medication Change?", ["No", "Ch"])
    metformin         = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"])

st.divider()

# ── Predict button ──────────────────────────────────────────
if st.button("🔍 Predict Readmission Risk", use_container_width=True, type="primary"):

    # Build a row of zeros matching all 45 feature names
    input_dict = dict.fromkeys(feature_names, 0)

    # Fill in what we know from the UI
    input_dict['time_in_hospital']   = time_in_hospital
    input_dict['num_lab_procedures'] = num_lab_procedures
    input_dict['num_procedures']     = num_procedures
    input_dict['num_medications']    = num_medications
    input_dict['number_outpatient']  = number_outpatient
    input_dict['number_emergency']   = number_emergency
    input_dict['number_inpatient']   = number_inpatient
    input_dict['number_diagnoses']   = number_diagnoses
    input_dict['age']                = AGE_MAP[age]
    input_dict['insulin']            = INSULIN_MAP[insulin]
    input_dict['A1Cresult']          = A1C_MAP[A1Cresult]
    input_dict['diabetesMed']        = YESNO_MAP[diabetesMed]
    input_dict['change']             = CHANGE_MAP[change]
    input_dict['gender']             = 1 if gender == "Male" else 0
    input_dict['metformin']          = INSULIN_MAP.get(metformin, 0)

    # Scale and predict
    input_df     = pd.DataFrame([input_dict])[feature_names]
    input_scaled = scaler.transform(input_df)
    probability  = model.predict_proba(input_scaled)[0][1]
    prediction   = int(probability >= 0.5)

    # ── Results ─────────────────────────────────────────────
    st.subheader("📊 Prediction Results")
    r1, r2, r3 = st.columns(3)

    with r1:
        risk_label = "🔴 HIGH RISK" if prediction == 1 else "🟢 LOW RISK"
        st.metric("Readmission Risk", risk_label)
    with r2:
        st.metric("Probability", f"{probability*100:.1f}%")
    with r3:
        st.metric("Model Confidence", f"{max(probability, 1-probability)*100:.1f}%")

    # Probability gauge bar
    st.progress(float(probability))
    if prediction == 1:
        st.error(f"⚠️ This patient has a **{probability*100:.1f}%** probability of readmission within 30 days. Immediate follow-up care is recommended.")
    else:
        st.success(f"✅ This patient has a **{probability*100:.1f}%** probability of readmission. Standard discharge protocols apply.")

    # ── Gemini AI Advice ────────────────────────────────────
    st.divider()
    st.subheader("🤖 AI Health Advisor — Gemini 2.5 Flash")

    if not GEMINI_API_KEY:
        st.warning("Add your GEMINI_API_KEY in Hugging Face Space secrets to enable AI advice.")
    else:
        with st.spinner("Generating personalised clinical recommendations..."):
            patient_data = {
                'age': age, 'time_in_hospital': time_in_hospital,
                'num_medications': num_medications,
                'number_diagnoses': number_diagnoses,
                'number_inpatient': number_inpatient,
                'number_emergency': number_emergency,
                'num_lab_procedures': num_lab_procedures,
                'diabetesMed': diabetesMed,
                'A1Cresult': A1Cresult, 'insulin': insulin
            }
            advice = get_ai_advice(patient_data, probability)
        st.markdown(advice)

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption("Built with Random Forest (90.97% accuracy · ROC-AUC 0.9573) · Gemini 2.5 Flash · Streamlit · UCI Diabetes Dataset")