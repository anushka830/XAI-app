import shap
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

# Create folder if not exists
os.makedirs("shap_plots", exist_ok=True)

# Use TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Debug: check shapes
print(f"shap_values[0] shape: {shap_values[0].shape}")
print(f"X_test shape: {X_test.shape}")

# Check if binary classification
if isinstance(shap_values, list) and len(shap_values) == 2:
    # Plot for class 1 (positive class)
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.savefig("shap_plots/summary_plot.png")
else:
    # For regression or other models
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("shap_plots/summary_plot.png")
from sklearn.metrics import accuracy_score
# After training
accuracy = accuracy_score(y_test, model.predict(X_test))
with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))
