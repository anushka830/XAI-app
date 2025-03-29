import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "diabetes_model.pkl")

# SHAP Explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Save SHAP summary plot
shap.summary_plot(shap_values[1], X_test, show=False)
plt.savefig("shap_plots/summary_plot.png", bbox_inches="tight")

# Force plot for first prediction
shap.initjs()
force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0], matplotlib=True)
plt.savefig("shap_plots/force_plot_0.png", bbox_inches="tight")

