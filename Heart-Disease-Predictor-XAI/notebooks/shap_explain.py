import pandas as pd
import joblib
import shap
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

pipeline = joblib.load(BASE_DIR / "models" / "heart_pipeline.pkl")

df = pd.read_csv(BASE_DIR / "data" / "heart.csv")

X = df.drop("target", axis=1)

sample_patient = pd.DataFrame([{
    "age": 55,
    "sex": 1,
    "cp": 4,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 130,
    "exang": 1,
    "oldpeak": 2.0,
    "slope": 2,
    "ca": 1,
    "thal": 7
}])

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

X_transformed = preprocessor.transform(X)
sample_transformed = preprocessor.transform(sample_patient)

feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample_transformed)

if isinstance(shap_values, list):
    shap_values_for_class_1 = shap_values[1][0]
else:
    shap_values_for_class_1 = shap_values[0, :, 1]

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "shap_value": shap_values_for_class_1
})

feature_importance["impact"] = feature_importance["shap_value"].abs()

feature_importance = feature_importance.sort_values(
    by="impact",
    ascending=False
)

print("SHAP EXPLANATION FOR SAMPLE PATIENT")
print("-" * 50)

print("\nTop factors influencing heart disease prediction:\n")

for index, row in feature_importance.head(8).iterrows():
    direction = "increased risk" if row["shap_value"] > 0 else "decreased risk"
    print(f"{row['feature']} → {direction} ({row['shap_value']:.4f})")