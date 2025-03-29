import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load model safely
try:
    model = joblib.load('model.pkl')  # Ensure this is a trained model, not an array
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load and show model accuracy
try:
    with open("accuracy.txt", "r") as f:
        acc = float(f.read())
    st.sidebar.subheader("📊 Model Accuracy")
    st.sidebar.write(f"{acc * 100:.2f}%")
except FileNotFoundError:
    st.sidebar.warning("⚠️ Accuracy info not found.")

st.title("🧠 Explainable AI - Diabetes Prediction")

# User inputs
st.header("Enter Patient Details:")
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 140)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 10, 100)

if st.button("Predict"):
    input_df = pd.DataFrame([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]],
                            columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
    
    try:
        result = model.predict(input_df)[0]
        st.subheader("Prediction Result:")
        st.write("🔴 Diabetic" if result == 1 else "🟢 Not Diabetic")
        
        st.subheader("Model Explanation with SHAP:")
        image = Image.open("shap_plots/summary_plot.png")
        st.image(image, caption="SHAP Summary Plot", use_column_width=True)
    except Exception as e:
        st.error(f"Prediction error: {e}")
