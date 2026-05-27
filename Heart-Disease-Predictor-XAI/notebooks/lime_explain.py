import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import re

from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

BASE_DIR = Path(__file__).resolve().parents[1]

df = pd.read_csv(BASE_DIR / "data" / "heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

pipeline = joblib.load(BASE_DIR / "models" / "best_heart_pipeline.pkl")

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

plots_dir = BASE_DIR / "plots"
reports_dir = BASE_DIR / "reports"

plots_dir.mkdir(exist_ok=True)
reports_dir.mkdir(exist_ok=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

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

original_features = list(X.columns)

categorical_features = [
    X.columns.get_loc("sex"),
    X.columns.get_loc("cp"),
    X.columns.get_loc("fbs"),
    X.columns.get_loc("restecg"),
    X.columns.get_loc("exang"),
    X.columns.get_loc("slope"),
    X.columns.get_loc("ca"),
    X.columns.get_loc("thal")
]


def prediction_function(data):
    data_df = pd.DataFrame(data, columns=X.columns)
    return pipeline.predict_proba(data_df)


# -------------------------
# LIME LOCAL EXPLANATION
# -------------------------

lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=["No Heart Disease", "Heart Disease"],
    categorical_features=categorical_features,
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

lime_exp = lime_explainer.explain_instance(
    data_row=sample_patient.iloc[0].values,
    predict_fn=prediction_function,
    num_features=10,
    labels=[1]
)

lime_list = lime_exp.as_list(label=1)

lime_df = pd.DataFrame(lime_list, columns=["lime_condition", "lime_weight"])

lime_df["lime_direction"] = lime_df["lime_weight"].apply(
    lambda x: "increased risk" if x > 0 else "decreased risk"
)


def extract_feature_from_lime(condition):
    for feature in sorted(original_features, key=len, reverse=True):
        pattern = r"(?<![A-Za-z_])" + re.escape(feature) + r"(?![A-Za-z_])"
        if re.search(pattern, condition):
            return feature
    return condition


lime_df["original_feature"] = lime_df["lime_condition"].apply(extract_feature_from_lime)

lime_grouped = lime_df.groupby("original_feature")["lime_weight"].sum().reset_index()
lime_grouped["lime_impact"] = lime_grouped["lime_weight"].abs()
lime_grouped["lime_direction"] = lime_grouped["lime_weight"].apply(
    lambda x: "increased risk" if x > 0 else "decreased risk"
)

lime_grouped = lime_grouped.sort_values(by="lime_impact", ascending=False)

lime_df.to_csv(reports_dir / "lime_explanation_sample.csv", index=False)
lime_grouped.to_csv(reports_dir / "lime_grouped_explanation_sample.csv", index=False)

lime_exp.save_to_file(str(plots_dir / "lime_explanation_sample.html"))

fig = lime_exp.as_pyplot_figure(label=1)
plt.tight_layout()
plt.savefig(plots_dir / "lime_explanation_sample.png", dpi=300, bbox_inches="tight")
plt.close()


# -------------------------
# SHAP LOCAL EXPLANATION
# -------------------------

X_train_processed = preprocessor.transform(X_train)
sample_processed = preprocessor.transform(sample_patient)

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

if hasattr(sample_processed, "toarray"):
    sample_processed = sample_processed.toarray()

feature_names = preprocessor.get_feature_names_out()

shap_explainer = shap.TreeExplainer(model)
shap_values = shap_explainer.shap_values(sample_processed)

if isinstance(shap_values, list):
    shap_values_for_class_1 = shap_values[1][0]
else:
    if len(shap_values.shape) == 3:
        shap_values_for_class_1 = shap_values[0, :, 1]
    else:
        shap_values_for_class_1 = shap_values[0]

shap_df = pd.DataFrame({
    "encoded_feature": feature_names,
    "shap_value": shap_values_for_class_1
})


def get_original_feature(encoded_name):
    name = encoded_name.replace("num__", "").replace("cat__", "")

    for feature in original_features:
        if name == feature or name.startswith(feature + "_"):
            return feature

    return name


shap_df["original_feature"] = shap_df["encoded_feature"].apply(get_original_feature)

shap_grouped = shap_df.groupby("original_feature")["shap_value"].sum().reset_index()
shap_grouped["shap_impact"] = shap_grouped["shap_value"].abs()
shap_grouped["shap_direction"] = shap_grouped["shap_value"].apply(
    lambda x: "increased risk" if x > 0 else "decreased risk"
)

shap_grouped = shap_grouped.sort_values(by="shap_impact", ascending=False)

shap_grouped.to_csv(reports_dir / "shap_grouped_explanation_sample.csv", index=False)


# -------------------------
# SHAP VS LIME COMPARISON
# -------------------------

comparison = pd.merge(
    shap_grouped,
    lime_grouped,
    on="original_feature",
    how="outer"
)

comparison["shap_value"] = comparison["shap_value"].fillna(0)
comparison["lime_weight"] = comparison["lime_weight"].fillna(0)
comparison["shap_impact"] = comparison["shap_impact"].fillna(0)
comparison["lime_impact"] = comparison["lime_impact"].fillna(0)

comparison["shap_direction"] = comparison["shap_value"].apply(
    lambda x: "increased risk" if x > 0 else ("decreased risk" if x < 0 else "neutral")
)

comparison["lime_direction"] = comparison["lime_weight"].apply(
    lambda x: "increased risk" if x > 0 else ("decreased risk" if x < 0 else "neutral")
)

comparison["direction_agreement"] = comparison.apply(
    lambda row: "Agree" if row["shap_direction"] == row["lime_direction"] else "Different",
    axis=1
)

comparison = comparison.sort_values(
    by=["shap_impact", "lime_impact"],
    ascending=False
)

comparison.to_csv(reports_dir / "shap_vs_lime_comparison_sample.csv", index=False)


print("LIME + SHAP VS LIME COMPARISON COMPLETED")
print("-" * 70)

print("\nLIME local explanation:")
print(lime_grouped.head(10))

print("\nSHAP local explanation:")
print(shap_grouped.head(10))

print("\nSHAP vs LIME comparison:")
print(comparison.head(10))

print("\nSaved files:")
print("plots/lime_explanation_sample.html")
print("plots/lime_explanation_sample.png")
print("reports/lime_explanation_sample.csv")
print("reports/lime_grouped_explanation_sample.csv")
print("reports/shap_grouped_explanation_sample.csv")
print("reports/shap_vs_lime_comparison_sample.csv")