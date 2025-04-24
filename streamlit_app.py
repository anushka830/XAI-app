import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import shap
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

        # SHAP Summary Plot
        st.subheader("📊 Model Explanation with SHAP")
        image = Image.open("shap_plots/summary_plot.png")
        st.image(image, caption="SHAP Summary Plot", use_column_width=True)

        with st.expander("🔍 What does this plot mean?"):
            st.markdown("""
            - This graph shows how different health factors (like **Glucose** or **BMI**) influence the AI’s decision.
            - **Each dot** represents a person in our dataset.
            - 🔴 Red = High value (e.g., high glucose), 🔵 Blue = Low value
            """)

        # ✅ Feature Contribution Bar
        st.subheader("📌 Feature Contribution for This Prediction")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        shap_df = pd.DataFrame({
            'Feature': input_df.columns,
            'SHAP Value': shap_values[0][0]  # only index 0 needed!
        }).sort_values(by="SHAP Value")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color='skyblue')
        ax.set_title("SHAP Value Impact (This Prediction)")
        ax.set_xlabel("Contribution to Diabetic Prediction")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction error: {e}")
