import pandas as pd
import joblib
from pathlib import Path
import dice_ml
from raiutils.exceptions import UserConfigValidationException

BASE_DIR = Path(__file__).resolve().parents[1]

df = pd.read_csv(BASE_DIR / "data" / "heart.csv")

pipeline = joblib.load(BASE_DIR / "models" / "best_heart_pipeline.pkl")

continuous_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

feature_columns = continuous_features + categorical_features


class PipelineWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def _prepare_input(self, data):
        if isinstance(data, pd.DataFrame):
            data_df = data.copy()
        else:
            data_df = pd.DataFrame(data, columns=feature_columns)

        if "target" in data_df.columns:
            data_df = data_df.drop("target", axis=1)

        data_df = data_df[feature_columns]

        for col in continuous_features:
            data_df[col] = pd.to_numeric(data_df[col], errors="coerce").astype(float)

        for col in categorical_features:
            data_df[col] = pd.to_numeric(data_df[col], errors="coerce").astype(int)

        return data_df

    def predict_proba(self, data):
        data_df = self._prepare_input(data)
        return self.pipeline.predict_proba(data_df)

    def predict(self, data):
        data_df = self._prepare_input(data)
        return self.pipeline.predict(data_df)


df_clean = df.copy()

for col in continuous_features:
    df_clean[col] = df_clean[col].astype(float)

for col in categorical_features:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    df_clean[col] = df_clean[col].astype(int).astype(str)

df_clean["target"] = df_clean["target"].astype(int)

data = dice_ml.Data(
    dataframe=df_clean,
    continuous_features=continuous_features,
    outcome_name="target"
)

wrapped_model = PipelineWrapper(pipeline)

model = dice_ml.Model(
    model=wrapped_model,
    backend="sklearn",
    model_type="classifier"
)

dice_explainer = dice_ml.Dice(
    data,
    model,
    method="random"
)

sample_patient = pd.DataFrame([{
    "age": 55.0,
    "trestbps": 140.0,
    "chol": 250.0,
    "thalach": 130.0,
    "oldpeak": 2.0,
    "sex": "1",
    "cp": "4",
    "fbs": "0",
    "restecg": "1",
    "exang": "1",
    "slope": "2",
    "ca": "1",
    "thal": "7"
}])

sample_for_prediction = sample_patient.copy()

for col in categorical_features:
    sample_for_prediction[col] = sample_for_prediction[col].astype(int)

original_prediction = pipeline.predict(sample_for_prediction)[0]
original_probability = pipeline.predict_proba(sample_for_prediction)[0][1]

print("ORIGINAL PATIENT PREDICTION")
print("-" * 60)
print("Prediction:", "Heart disease risk" if original_prediction == 1 else "No heart disease risk")
print(f"Risk Probability: {original_probability * 100:.2f}%")

reports_dir = BASE_DIR / "reports"
reports_dir.mkdir(exist_ok=True)


def print_counterfactuals(cf_df, file_name):
    cf_df.to_csv(reports_dir / file_name, index=False)

    print("\nCOUNTERFACTUAL EXPLANATIONS")
    print("-" * 60)
    print(cf_df)

    print("\nPlain-English Explanation:")
    print(
        "These counterfactuals show possible feature changes that could move "
        "the model prediction from high-risk to lower-risk. These are model-based "
        "explanations, not medical advice."
    )

    for i, row in cf_df.iterrows():
        cf_patient = row.drop("target").to_frame().T

        for col in continuous_features:
            cf_patient[col] = pd.to_numeric(cf_patient[col], errors="coerce").astype(float)

        for col in categorical_features:
            cf_patient[col] = pd.to_numeric(cf_patient[col], errors="coerce").astype(int)

        cf_probability = pipeline.predict_proba(cf_patient)[0][1]

        print(f"\nCounterfactual {i + 1}:")
        print(f"New Risk Probability: {cf_probability * 100:.2f}%")

        for feature in feature_columns:
            original_value = sample_for_prediction[feature].iloc[0]
            new_value = cf_patient[feature].iloc[0]

            if float(original_value) != float(new_value):
                print(f"- {feature}: {original_value} → {new_value}")

    print(f"\nSaved counterfactuals at reports/{file_name}")


print("\nTrying actionable counterfactuals first...")
print("-" * 60)

try:
    counterfactuals = dice_explainer.generate_counterfactuals(
        sample_patient,
        total_CFs=3,
        desired_class=0,
        features_to_vary=[
            "trestbps",
            "chol",
            "thalach",
            "oldpeak"
        ],
        permitted_range={
            "trestbps": [90, 180],
            "chol": [120, 300],
            "thalach": [90, 200],
            "oldpeak": [0, 6]
        },
        sample_size=10000
    )

    cf_df = counterfactuals.cf_examples_list[0].final_cfs_df

    if cf_df is not None and not cf_df.empty:
        print_counterfactuals(cf_df, "dice_counterfactuals_actionable.csv")
    else:
        raise UserConfigValidationException("No actionable counterfactuals found.")

except Exception:
    print("No counterfactuals found using only actionable numeric features.")
    print("Trying broader clinical counterfactual search...")

    try:
        counterfactuals = dice_explainer.generate_counterfactuals(
            sample_patient,
            total_CFs=3,
            desired_class=0,
            features_to_vary=[
                "trestbps",
                "chol",
                "thalach",
                "oldpeak",
                "cp",
                "exang",
                "slope",
                "ca",
                "thal"
            ],
            permitted_range={
                "trestbps": [90, 180],
                "chol": [120, 300],
                "thalach": [90, 200],
                "oldpeak": [0, 6],
                "cp": ["1", "2", "3", "4"],
                "exang": ["0", "1"],
                "slope": ["1", "2", "3"],
                "ca": ["0", "1", "2", "3"],
                "thal": ["3", "6", "7"]
            },
            sample_size=20000
        )

        cf_df = counterfactuals.cf_examples_list[0].final_cfs_df

        if cf_df is not None and not cf_df.empty:
            print_counterfactuals(cf_df, "dice_counterfactuals_clinical.csv")
        else:
            print("No broader clinical counterfactuals found.")

    except Exception as e:
        print("\nDiCE still could not generate counterfactuals.")
        print("Reason:")
        print(e)

        print("\nThis can happen when the sample patient is classified with extremely high confidence.")
        print("For this patient, the original predicted risk is very high, so small feature changes may not flip the class.")