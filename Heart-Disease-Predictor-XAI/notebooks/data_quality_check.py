import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
df = pd.read_csv(BASE_DIR / "data" / "heart.csv")

print("DATA QUALITY REPORT")
print("-" * 40)

print("\n1. Dataset Shape:")
print(df.shape)

print("\n2. Missing Values:")
print(df.isnull().sum())

print("\n3. Duplicate Rows:")
print(df.duplicated().sum())

print("\n4. Target Distribution:")
print(df["target"].value_counts())

print("\n5. Invalid / Suspicious Values:")

checks = {
    "age": (0, 120),
    "trestbps": (70, 250),
    "chol": (100, 600),
    "thalach": (60, 250),
    "oldpeak": (0, 10),
    "ca": (0, 3),
    "thal": (3, 7)
}

for col, (low, high) in checks.items():
    invalid_count = df[(df[col] < low) | (df[col] > high)][col].count()
    print(f"{col}: {invalid_count} suspicious values")

print("\n6. Summary Statistics:")
print(df.describe())

print("\nData quality check completed.")