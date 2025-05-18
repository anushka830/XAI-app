import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as components

# Load model
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar: Accuracy info
st.sidebar.title("📈 Model Info")
try:
    with open("accuracy.txt", "r") as f:
        acc = float(f.read())
    st.sidebar.subheader("✅ Model Accuracy")
    st.sidebar.write(f"{acc * 100:.2f}%")
except FileNotFoundError:
    st.sidebar.warning("⚠️ Accuracy info not found.")

# Chatbot in sidebar
st.sidebar.subheader("💬 Ask the Diabetes Bot")
user_question = st.sidebar.text_input("Ask me anything about diabetes")

if user_question:
    response = "🤖 I'm not sure. Please consult a doctor."
    q = user_question.lower()
    if "what is diabetes" in q:
        response = "🩸 Diabetes affects how your body uses blood sugar (glucose)."
    elif "symptoms" in q:
        response = "⚠️ Common symptoms: frequent urination, fatigue, blurred vision, thirst."
    elif "prevent" in q:
        response = "✅ Prevention: eat healthy, exercise, maintain a healthy weight."
    elif "type 1" in q:
        response = "🧬 Type 1 is an autoimmune condition – body attacks insulin-producing cells."
    elif "type 2" in q:
        response = "🍔 Type 2 is lifestyle-related – linked to insulin resistance."
    st.sidebar.info(response)

# Main UI
st.title("🧠 Explainable AI - Diabetes Prediction")
st.header("📝 Enter Patient Details")

# User inputs
pregnancies = st.number_input("Pregnancies", 0, 20, value=0)
glucose = st.number_input("Glucose", 0, 200, value=0)
bp = st.number_input("Blood Pressure", 0, 140, value=0)
skin_thickness = st.number_input("Skin Thickness", 0, 100, value=0)
insulin = st.number_input("Insulin", 0, 900, value=0)
bmi = st.number_input("BMI", 0.0, 70.0, value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, value=0.0)
age = st.number_input("Age", 10, 100, value=10)

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

        # SHAP Explanation
        st.subheader("📊 Model Explanation with SHAP")

        try:
            image = Image.open("shap_plots/summary_plot.png")
            st.image(image, caption="SHAP Summary Plot", use_column_width=True)
        except FileNotFoundError:
            st.warning("SHAP summary plot image not found.")

        # SHAP interaction
        st.subheader("🎨 SHAP Interaction Plot")
        with st.spinner("Generating SHAP values..."):
            explainer = shap.Explainer(model.predict, input_df)
            shap_values = explainer(input_df)

        selected_feature = st.selectbox("🔍 Select a feature to view interaction impact:", input_df.columns)

        fig, ax = plt.subplots()
        shap.plots.scatter(shap_values[:, selected_feature], color=shap_values, ax=ax, show=False)
        st.pyplot(fig)

        st.markdown(f"### 🎯 Interaction Effect of **{selected_feature}** on Prediction")

        with st.expander("ℹ️ How to read this interaction plot?"):
            st.markdown("""
            - **What it shows**: How the selected feature interacts with others to influence the prediction.
            - **X-axis**: SHAP interaction value (how much the feature changes the model output).
            - **Dots**: Each dot represents a patient.
            - 🔵 **Blue** = low value of the interacting feature.
            - 🔴 **Red** = high value.
            - ✅ Right = more likely diabetic, ❌ Left = less likely diabetic.
            """)

        # Feature contribution
        st.subheader("📌 Feature Contribution for This Prediction")

        if hasattr(model, "predict_proba"):
            pred_func = model.predict_proba
        else:
            pred_func = model.predict

        shap_values_full = shap.Explainer(pred_func, input_df)(input_df)

        if len(shap_values_full.values.shape) == 3:
            shap_vals_for_class1 = shap_values_full.values[0, :, 1]
        else:
            shap_vals_for_class1 = shap_values_full.values[0]

        shap_df = pd.DataFrame({
            'Feature': input_df.columns,
            'SHAP Value': shap_vals_for_class1,
            'Input Value': input_df.iloc[0].values
        }).sort_values('SHAP Value')

        fig = px.bar(
            shap_df,
            x="SHAP Value",
            y="Feature",
            orientation='h',
            hover_data=['Input Value'],
            title="🔍 SHAP Feature Impact (with Your Inputs)",
            color="SHAP Value",
            color_continuous_scale="Tealrose"
        )
        fig.update_layout(xaxis_title="Contribution to Prediction", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
