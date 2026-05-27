import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]

df = pd.read_csv(BASE_DIR / "data" / "heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

pipeline = joblib.load(BASE_DIR / "models" / "best_heart_pipeline.pkl")

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

if hasattr(X_train_processed, "toarray"):
    X_train_processed = X_train_processed.toarray()

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

feature_names = preprocessor.get_feature_names_out()

explainer = shap.Explainer(
    model,
    X_train_processed,
    feature_names=feature_names
)

shap_values = explainer(X_test_processed)

plots_dir = BASE_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

plt.figure()
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.tight_layout()
plt.savefig(plots_dir / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
shap.plots.waterfall(shap_values[0], max_display=12, show=False)
plt.tight_layout()
plt.savefig(plots_dir / "shap_waterfall_sample.png", dpi=300, bbox_inches="tight")
plt.close()

global_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": abs(shap_values.values).mean(axis=0)
})

global_importance = global_importance.sort_values(
    by="mean_abs_shap",
    ascending=False
)

global_importance.to_csv(
    BASE_DIR / "reports" / "shap_global_importance.csv",
    index=False
)

print("SHAP GLOBAL + LOCAL EXPLANATION COMPLETED")
print("-" * 60)

print("\nTop 15 global important features:")
print(global_importance.head(15))

print("\nSaved files:")
print("plots/shap_beeswarm.png")
print("plots/shap_waterfall_sample.png")
print("reports/shap_global_importance.csv")