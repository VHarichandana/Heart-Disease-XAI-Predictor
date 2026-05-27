import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay

BASE_DIR = Path(__file__).resolve().parents[1]

df = pd.read_csv(BASE_DIR / "data" / "heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

for col in numeric_features:
    X[col] = X[col].astype(float)

pipeline = joblib.load(BASE_DIR / "models" / "best_heart_pipeline.pkl")

plots_dir = BASE_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

features_to_plot = [
    "chol",
    "thalach",
    "oldpeak",
    "age",
    "trestbps"
]

for feature in features_to_plot:
    fig, ax = plt.subplots(figsize=(8, 5))

    PartialDependenceDisplay.from_estimator(
        pipeline,
        X_test,
        features=[feature],
        response_method="predict_proba",
        ax=ax
    )

    plt.title(f"Partial Dependence Plot for {feature}")
    plt.tight_layout()

    output_path = plots_dir / f"pdp_{feature}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved PDP plot: plots/pdp_{feature}.png")

print("\nPDP analysis completed.")