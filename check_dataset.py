import pandas as pd

df = pd.read_csv("customer_churn_data.csv")

print("Columns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

if "Churn" in df.columns:
    cleaned = df["Churn"].astype(str).str.strip()

    print("\nChurn value counts:")
    print(cleaned.value_counts(dropna=False))

    bad = df[~cleaned.isin(["Yes", "No"])]
    print("\nInvalid Churn rows:")
    print(bad)
