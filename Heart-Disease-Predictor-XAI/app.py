import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_heart_pipeline.pkl"
DATA_PATH = BASE_DIR / "data" / "heart.csv"

pipeline = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)

st.set_page_config(
    page_title="Heart Disease XAI Predictor",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Explainable Heart Disease Risk Predictor")
st.write(
    "This app predicts heart disease risk and explains the prediction using "
    "data quality checks, SHAP, LIME, PDP, and DiCE counterfactual explanations."
)

st.warning(
    "Medical Disclaimer: This project is for educational purposes only. "
    "It is not a medical diagnosis tool. Always consult a qualified healthcare professional."
)


def validate_patient_data(patient):
    warnings = []

    if patient["age"] < 1 or patient["age"] > 120:
        warnings.append("Age value is outside the valid human range.")

    if patient["trestbps"] <= 0:
        warnings.append("Resting blood pressure must be greater than 0.")

    if patient["chol"] <= 0:
        warnings.append("Cholesterol must be greater than 0.")

    if patient["thalach"] <= 0:
        warnings.append("Maximum heart rate must be greater than 0.")

    if patient["oldpeak"] < 0:
        warnings.append("Oldpeak cannot be negative.")

    if patient["trestbps"] > 200:
        warnings.append("Resting blood pressure is unusually high.")

    if patient["chol"] > 500:
        warnings.append("Cholesterol value is unusually high.")

    if patient["thalach"] > 220:
        warnings.append("Maximum heart rate is unusually high.")

    return warnings


def percentile_text(feature, value):
    if feature not in df.columns:
        return ""

    percentile = (df[feature] <= value).mean() * 100

    return f"{value} is around the {percentile:.1f}th percentile in this dataset."


def get_clean_shap_explanation(patient_df):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    patient_transformed = preprocessor.transform(patient_df)

    if hasattr(patient_transformed, "toarray"):
        patient_transformed = patient_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_transformed)

    if isinstance(shap_values, list):
        shap_values_for_class_1 = shap_values[1][0]
    else:
        shap_array = np.array(shap_values)

        if shap_array.ndim == 3:
            if shap_array.shape[0] == 2:
                shap_values_for_class_1 = shap_array[1][0]
            else:
                shap_values_for_class_1 = shap_array[0, :, 1]
        elif shap_array.ndim == 2:
            shap_values_for_class_1 = shap_array[0]
        else:
            shap_values_for_class_1 = shap_array

    shap_df = pd.DataFrame({
        "encoded_feature": feature_names,
        "shap_value": shap_values_for_class_1
    })

    original_features = [
        "age", "trestbps", "chol", "thalach", "oldpeak",
        "sex", "cp", "fbs", "restecg", "exang",
        "slope", "ca", "thal"
    ]

    def get_original_feature(encoded_name):
        name = encoded_name.replace("num__", "").replace("cat__", "")

        for feature in original_features:
            if name == feature or name.startswith(feature + "_"):
                return feature

        return name

    shap_df["original_feature"] = shap_df["encoded_feature"].apply(get_original_feature)

    grouped = shap_df.groupby("original_feature")["shap_value"].sum().reset_index()
    grouped["impact"] = grouped["shap_value"].abs()
    grouped = grouped.sort_values(by="impact", ascending=False)

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

    grouped["feature_name"] = grouped["original_feature"].map(feature_meanings)
    grouped["feature_name"] = grouped["feature_name"].fillna(grouped["original_feature"])

    return grouped


def show_image_if_exists(path, caption):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{caption} not found. Run the related notebook/script first.")


def show_csv_if_exists(path):
    if path.exists():
        data = pd.read_csv(path)
        st.dataframe(data, use_container_width=True)
        return data
    else:
        st.info(f"File not found: {path.name}")
        return None


st.sidebar.header("Patient Input Details")

age = st.sidebar.slider("Age", 1, 120, 55)

sex = st.sidebar.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.sidebar.selectbox(
    "Chest Pain Type",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-anginal Pain",
        4: "Asymptomatic"
    }[x]
)

trestbps = st.sidebar.number_input(
    "Resting Blood Pressure",
    min_value=0.0,
    max_value=300.0,
    value=140.0
)

chol = st.sidebar.number_input(
    "Cholesterol",
    min_value=0.0,
    max_value=700.0,
    value=250.0
)

fbs = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

restecg = st.sidebar.selectbox(
    "Resting ECG Result",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy"
    }[x]
)

thalach = st.sidebar.number_input(
    "Maximum Heart Rate Achieved",
    min_value=0.0,
    max_value=300.0,
    value=130.0
)

exang = st.sidebar.selectbox(
    "Exercise Induced Angina",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

oldpeak = st.sidebar.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1
)

slope = st.sidebar.selectbox(
    "Slope of ST Segment",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "Upsloping",
        2: "Flat",
        3: "Downsloping"
    }[x]
)

ca = st.sidebar.selectbox(
    "Number of Major Vessels",
    options=[0, 1, 2, 3]
)

thal = st.sidebar.selectbox(
    "Thalassemia Result",
    options=[3, 6, 7],
    format_func=lambda x: {
        3: "Normal",
        6: "Fixed Defect",
        7: "Reversible Defect"
    }[x]
)

patient_dict = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}

patient_df = pd.DataFrame([patient_dict])

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Prediction",
    "Model Comparison",
    "SHAP",
    "LIME vs SHAP",
    "PDP",
    "Counterfactuals"
])


with tab1:
    st.header("Patient Risk Prediction")

    st.subheader("Patient Data")
    st.dataframe(patient_df, use_container_width=True)

    st.subheader("Data Quality Check")
    warnings = validate_patient_data(patient_dict)

    if warnings:
        for warning in warnings:
            st.warning(warning)
    else:
        st.success("No data quality issues found.")

    prediction = pipeline.predict(patient_df)[0]
    probability = pipeline.predict_proba(patient_df)[0][1]

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("Heart disease risk detected")
        else:
            st.success("No heart disease risk detected")

    with col2:
        st.metric("Risk Probability", f"{probability * 100:.2f}%")

    st.subheader("Plain-English Risk Context")

    st.write(f"**Cholesterol:** {percentile_text('chol', chol)}")
    st.write(f"**Resting blood pressure:** {percentile_text('trestbps', trestbps)}")
    st.write(f"**Maximum heart rate:** {percentile_text('thalach', thalach)}")
    st.write(f"**Oldpeak:** {percentile_text('oldpeak', oldpeak)}")

    st.info(
        "For medical-style problems, recall is especially important because "
        "missing a high-risk patient can be more serious than giving an extra warning."
    )


with tab2:
    st.header("Model Comparison")

    st.write(
        "Three models were trained and compared: Logistic Regression, Random Forest, and XGBoost. "
        "The best model was selected by prioritizing recall first, then ROC-AUC."
    )

    comparison_path = BASE_DIR / "reports" / "model_comparison_results.csv"
    comparison_df = show_csv_if_exists(comparison_path)

    if comparison_df is not None:
        best_row = comparison_df.iloc[0]
        st.success(
            f"Best Model: {best_row['Model']} | "
            f"Recall: {best_row['Recall']:.4f} | "
            f"ROC-AUC: {best_row['ROC-AUC']:.4f}"
        )


with tab3:
    st.header("SHAP Explainability")

    st.subheader("Local SHAP Explanation for Current Patient")

    shap_result = get_clean_shap_explanation(patient_df)
    top_features = shap_result.head(8).copy()

    for _, row in top_features.iterrows():
        direction = "increased risk" if row["shap_value"] > 0 else "decreased risk"
        st.write(
            f"**{row['feature_name']}** → {direction} "
            f"({row['shap_value']:.4f})"
        )

    chart_data = top_features[["feature_name", "impact"]].set_index("feature_name")
    st.bar_chart(chart_data)

    st.subheader("Global SHAP Beeswarm Plot")
    st.write(
        "This plot shows which features influence model predictions across many patients."
    )
    show_image_if_exists(
        BASE_DIR / "plots" / "shap_beeswarm.png",
        "SHAP Beeswarm Plot"
    )

    st.subheader("Sample SHAP Waterfall Plot")
    show_image_if_exists(
        BASE_DIR / "plots" / "shap_waterfall_sample.png",
        "SHAP Waterfall Plot for Sample Patient"
    )


with tab4:
    st.header("LIME vs SHAP Comparison")

    st.write(
        "SHAP and LIME are both local explainability methods, but they work differently. "
        "SHAP estimates feature contribution values, while LIME fits a simple local surrogate model "
        "around one prediction. Their numeric values are not directly equal, but their top features "
        "and directions can be compared."
    )

    lime_img = BASE_DIR / "plots" / "lime_explanation_sample.png"
    show_image_if_exists(lime_img, "LIME Explanation for Sample Patient")

    st.subheader("SHAP vs LIME Agreement Table")
    comparison = show_csv_if_exists(
        BASE_DIR / "reports" / "shap_vs_lime_comparison_sample.csv"
    )

    if comparison is not None and "direction_agreement" in comparison.columns:
        agree_count = (comparison["direction_agreement"] == "Agree").sum()
        total_count = len(comparison)
        st.success(
            f"SHAP and LIME agreed on direction for {agree_count} out of {total_count} features."
        )


with tab5:
    st.header("Partial Dependence Plots")

    st.write(
        "PDP plots show how the model's predicted risk changes as one feature changes "
        "across the dataset. Unlike SHAP and LIME, PDP focuses on feature effect trends."
    )

    pdp_features = ["chol", "thalach", "oldpeak", "age", "trestbps"]

    for feature in pdp_features:
        st.subheader(f"PDP for {feature}")
        show_image_if_exists(
            BASE_DIR / "plots" / f"pdp_{feature}.png",
            f"Partial Dependence Plot for {feature}"
        )


with tab6:
    st.header("DiCE Counterfactual Explanations")

    st.write(
        "Counterfactual explanations show what feature changes could move the model "
        "from a high-risk prediction to a lower-risk prediction. These are model-based "
        "explanations, not medical advice."
    )

    cf_path = BASE_DIR / "reports" / "dice_counterfactuals_clinical.csv"
    cf_df = show_csv_if_exists(cf_path)

    if cf_df is not None:
        st.subheader("Counterfactual Summary")

        original_patient = pd.DataFrame([{
            "age": 55.0,
            "sex": 1,
            "cp": 4,
            "trestbps": 140.0,
            "chol": 250.0,
            "fbs": 0,
            "restecg": 1,
            "thalach": 130.0,
            "exang": 1,
            "oldpeak": 2.0,
            "slope": 2,
            "ca": 1,
            "thal": 7
        }])

        original_prob = pipeline.predict_proba(original_patient)[0][1]

        st.write(f"Original sample patient risk probability: **{original_prob * 100:.2f}%**")

        for i, row in cf_df.iterrows():
            cf_patient = row.drop("target").to_frame().T

            for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
                cf_patient[col] = pd.to_numeric(cf_patient[col], errors="coerce").astype(float)

            for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
                cf_patient[col] = pd.to_numeric(cf_patient[col], errors="coerce").astype(int)

            cf_prob = pipeline.predict_proba(cf_patient)[0][1]

            st.markdown(f"### Counterfactual {i + 1}")
            st.write(f"New risk probability: **{cf_prob * 100:.2f}%**")

            changes = []

            for feature in [
                "trestbps", "chol", "thalach", "oldpeak",
                "cp", "exang", "slope", "ca", "thal"
            ]:
                original_value = original_patient[feature].iloc[0]
                new_value = cf_patient[feature].iloc[0]

                if float(original_value) != float(new_value):
                    changes.append(f"- `{feature}`: {original_value} → {new_value}")

            if changes:
                st.markdown("\n".join(changes))
            else:
                st.write("No major feature changes shown.")