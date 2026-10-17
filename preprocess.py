import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]

TARGET_COLUMN = "Churn"


def load_and_clean_data(csv_path=DATA_PATH):
    """
    Loads raw Telco customer dataset and applies cleaning transformations:
      1. Converts TotalCharges whitespace/blank strings to numeric 0.0 (for tenure=0 customers).
      2. Maps target 'Churn' ('Yes' -> 1, 'No' -> 0).
      3. Drops non-predictive identifier 'customerID'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Handle whitespace in TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce").fillna(0.0)

    # 2. Ensure SeniorCitizen is treated as categorical string/int
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # 3. Target encoding
    if TARGET_COLUMN in df.columns:
        df["Churn_Binary"] = (df[TARGET_COLUMN].str.strip().str.lower() == "yes").astype(int)

    return df


def build_preprocessor():
    """
    Constructs a robust Scikit-learn ColumnTransformer:
      - Numerical: StandardScaler
      - Categorical: OneHotEncoder(drop='first', handle_unknown='ignore')
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


if __name__ == "__main__":
    df = load_and_clean_data()
    print("=" * 60)
    print("Telco Dataset Loaded & Cleaned Successfully")
    print("=" * 60)
    print(f"Total Rows:    {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Churn Rate:    {df['Churn_Binary'].mean() * 100:.2f}% ({df['Churn_Binary'].sum()} churned customers)")
    print("\nNumerical Summary:")
    print(df[NUMERICAL_FEATURES].describe())
    print("\nCategorical Features Sample:")
    print(df[CATEGORICAL_FEATURES].head(3))
