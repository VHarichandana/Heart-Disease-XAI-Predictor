from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features
y = heart_disease.data.targets

df = pd.concat([X, y], axis=1)

df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
df.drop("num", axis=1, inplace=True)

os.makedirs("data", exist_ok=True)

df.to_csv("data/heart.csv", index=False)

print("Dataset loaded successfully!")
print(df.head())
print("Shape:", df.shape)
print("Missing values:")
print(df.isnull().sum())