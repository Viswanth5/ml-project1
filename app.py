import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("riskmodel.pkl")

# Page configuration
st.set_page_config(page_title="Disease Risk Prediction", layout="centered")

# App title and subtitle
st.title("🧬 Disease Risk Prediction System")
st.markdown("Predict your risk level based on health details.")

# User input fields
age = st.slider("Age", 0, 120, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
smoking = st.selectbox("Smoking Habit", ["Yes", "No"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)
blood_pressure = st.number_input("Blood Pressure (Systolic)", min_value=80, max_value=200, value=120)
cholesterol = st.selectbox("Cholesterol Level", ["Normal", "High", "Very High"])

# Encode categorical values
gender_val = 1 if gender == "Male" else 0
smoking_val = 1 if smoking == "Yes" else 0
cholesterol_val = {"Normal": 0, "High": 1, "Very High": 2}[cholesterol]

# Create input DataFrame
input_df = pd.DataFrame({
    "age": [age],
    "gender": [gender_val],
    "smoking": [smoking_val],
    "bmi": [bmi],
    "blood_pressure": [blood_pressure],
    "cholesterol": [cholesterol_val]
})

# Prediction button
if st.button("Predict Risk"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"🩺 Predicted Disease Risk: **{prediction}**")
    except Exception as e:
        st.error(f"Error: {e}")
