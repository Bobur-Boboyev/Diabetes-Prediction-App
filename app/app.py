import streamlit as st
import numpy as np
import joblib
import os

@st.cache_resource
def load_model(relative_path):
    base_dir = os.path.dirname(__file__)
    model_path = os.path.abspath(os.path.join(base_dir, relative_path))
    return joblib.load(model_path)

model = load_model("../models/diabetes_lr_model.pkl")
scaler = load_model("../models/scaler.pkl")
    
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter the correct information and determine the probability of diabetes.")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

if st.button("🔍 Diagnosing diabetes"):
    try:
        input_data = np.array([[pregnancies, glucose, blood_pressure,skin_thickness,insulin, bmi, dpf, age]])
            
        input_scaled = scaler.transform(input_data)

        pred = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)[0][1]

        if pred[0] == 1:
            st.error(f"⚠️ Diabetes diagnosed! Probably: {prob * 100:.2f}%")
        else:
            st.success(f"✅ Diabetes was not detected. Possibly: {prob * 100:.2f}%")

    except Exception as e:
        st.warning(f"Error: {str(e)}")