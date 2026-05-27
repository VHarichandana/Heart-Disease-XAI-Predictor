import pandas as pd
import joblib
from pathlib import Path
from input_validation import validate_patient_data

BASE_DIR = Path(__file__).resolve().parents[1]

model = joblib.load(BASE_DIR / "models" / "heart_pipeline.pkl")

patient_dict = {
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
}

warnings = validate_patient_data(patient_dict)

print("DATA QUALITY WARNINGS")
print("-" * 40)

if warnings:
    for warning in warnings:
        print("Warning:", warning)
else:
    print("No data quality issues found.")

sample_patient = pd.DataFrame([patient_dict])

prediction = model.predict(sample_patient)[0]
probability = model.predict_proba(sample_patient)[0][1]

print("\nPREDICTION RESULT")
print("-" * 40)

if prediction == 1:
    print("Prediction: Heart disease risk detected")
else:
    print("Prediction: No heart disease risk detected")

print(f"Risk Probability: {probability * 100:.2f}%")