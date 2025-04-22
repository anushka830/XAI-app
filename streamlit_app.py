import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load trained model
try:
    model = joblib.load('model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar: Show accuracy
st.sidebar.title("📈 Model Info")
try:
    with open("accuracy.txt", "r") as f:
        acc = float(f.read())
    st.sidebar.subheader("✅ Model Accuracy")
    st.sidebar.write(f"{acc * 100:.2f}%")
except FileNotFoundError:
    st.sidebar.warning("⚠️ Accuracy info not found.")

# App title
st.title("🧠 Explainable AI - Diabetes Prediction")

# Input form
st.header("📝 Enter Patient Details")
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 140)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 10, 100)

# Predict button
if st.button("🚀 Predict"):
    input_df = pd.DataFrame([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]],
                             columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                      "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

    try:
        result = model.predict(input_df)[0]

        st.subheader("🔎 Prediction Result:")
        st.success("🟢 Not Diabetic" if result == 0 else "🔴 Diabetic")

        # SHAP Explanation
        st.subheader("📊 Model Explanation with SHAP")
        image = Image.open("shap_plots/summary_plot.png")
        st.image(image, caption="SHAP Summary Plot", use_column_width=True)

        # Explanation for non-tech users
        with st.expander("🔍 What does this plot mean?"):
            st.markdown("""
            - This graph shows how different health factors (like **Glucose** or **BMI**) influence the AI’s decision.
            - **Each dot** represents a person in our dataset.
            - **Color of dots**:
              - 🔴 Red/Pink = Higher values (e.g. more glucose or insulin)
              - 🔵 Blue = Lower values (e.g. lower BMI or blood pressure)
            - **Right side (positive values)** means the feature is pushing towards being **Diabetic**.
            - **Left side (negative values)** means it's pushing towards **Not Diabetic**.
            
            ✅ This helps make the prediction more transparent and understandable.
            """)

    except Exception as e:
        st.error(f"Prediction error: {e}")

