import pandas as pd
import joblib
import shap
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

shap_df = pd.DataFrame({
    "encoded_feature": feature_names,
    "shap_value": shap_values_for_class_1
})

def get_original_feature(encoded_name):
    name = encoded_name.replace("num__", "").replace("cat__", "")

    original_features = [
        "age", "trestbps", "chol", "thalach", "oldpeak",
        "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
    ]

    for feature in original_features:
        if name == feature or name.startswith(feature + "_"):
            return feature

    return name

shap_df["original_feature"] = shap_df["encoded_feature"].apply(get_original_feature)

grouped_shap = shap_df.groupby("original_feature")["shap_value"].sum().reset_index()
grouped_shap["impact"] = grouped_shap["shap_value"].abs()

grouped_shap = grouped_shap.sort_values(by="impact", ascending=False)

feature_meanings = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting blood sugar",
    "restecg": "Resting ECG result",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression during exercise",
    "slope": "Slope of ST segment",
    "ca": "Number of major vessels",
    "thal": "Thalassemia result"
}

print("CLEAN SHAP EXPLANATION FOR SAMPLE PATIENT")
print("-" * 50)

print("\nTop original features influencing prediction:\n")

for _, row in grouped_shap.head(8).iterrows():
    feature = row["original_feature"]
    direction = "increased risk" if row["shap_value"] > 0 else "decreased risk"
    value = sample_patient[feature].iloc[0]

    print(
        f"{feature_meanings.get(feature, feature)} "
        f"(value: {value}) → {direction} ({row['shap_value']:.4f})"
    )