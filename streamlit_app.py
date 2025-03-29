import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Load model
model = joblib.load("diabetes_model.pkl")

st.title("🧠 Diabetes Predictor with Explainable AI (SHAP)")

# User Input
glucose = st.slider("Glucose", 0, 200, 100)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
age = st.slider("Age", 20, 80, 33)
insulin = st.slider("Insulin", 0, 300, 80)
pregnancies = st.slider("Pregnancies", 0, 10, 2)
blood_pressure = st.slider("Blood Pressure", 40, 140, 70)
skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

input_data = pd.DataFrame({
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
})

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: {'Diabetic' if prediction == 1 else 'Not Diabetic'}")

    st.subheader("🔍 Explanation (SHAP Summary Plot)")
    summary_img = Image.open("shap_plots/summary_plot.png")
    st.image(summary_img, caption="Feature Impact Summary")

    st.subheader("📌 Force Plot for Sample Case")
    force_img = Image.open("shap_plots/force_plot_0.png")
    st.image(force_img, caption="SHAP Force Plot (Sample)")

