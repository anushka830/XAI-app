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
        if result == 1:
            st.error("🔴 Diabetic")
            
            # Home Remedies
            with st.expander("💡 Home Remedies & Lifestyle Tips"):
                st.markdown("""
                - 🥗 **Eat low-carb, high-fiber foods** to stabilize blood sugar.
                - 🚶 **Exercise daily**, like a 30-minute walk.
                - 🍵 **Try herbal teas** such as fenugreek, ginger, or cinnamon.
                - 🧘 **Practice stress relief** through yoga or meditation.
                - 💤 **Sleep well** – aim for 7-8 hours nightly.
                - 🧂 **Avoid processed and sugary foods**.
                - ❗ Always consult your doctor before trying new remedies.
                """)
        else:
            st.success("🟢 Not Diabetic")
            st.info("✅ Keep up the healthy lifestyle!")

        # SHAP Explanation
        st.subheader("📊 Model Explanation with SHAP")
        image = Image.open("shap_plots/summary_plot.png")
        st.image(image, caption="SHAP Summary Plot", use_column_width=True)

        # SHAP Explanation for users
        with st.expander("🔍 What does this plot mean?"):
            st.markdown("""
            - This graph shows how different health factors (like **Glucose** or **BMI**) influence the AI’s decision.
            - **Each dot** represents a person in our dataset.
            - **Dot Colors**:
              - 🔴 Red = Higher feature values (e.g. high glucose)
              - 🔵 Blue = Lower feature values (e.g. low insulin)
            - **Dots on the right** → feature supports diabetic prediction.
            - **Dots on the left** → feature supports not diabetic.
            """)

    except Exception as e:
        st.error(f"Prediction error: {e}")


