import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

from preprocess import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PIPELINE_FILE = os.path.join(MODEL_DIR, "churn_pipeline.pkl")
METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.json")

# Default baseline fallback profile
DEFAULT_CUSTOMER_PROFILE = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0
}


class ChurnPredictor:
    """
    Inference and Customer Retention Engine.
    Computes churn probability, risk classification, key risk drivers,
    and personalized customer retention recommendations.
    """

    def __init__(self, pipeline_path=PIPELINE_FILE, metadata_path=METADATA_FILE):
        self.pipeline_path = pipeline_path
        self.metadata_path = metadata_path
        self.pipeline = None
        self.metadata = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.pipeline_path) and os.path.exists(self.metadata_path):
            try:
                self.pipeline = joblib.load(self.pipeline_path)
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load churn pipeline: {e}")
                self.pipeline = None
                self.metadata = None

    def is_available(self) -> bool:
        return self.pipeline is not None

    def predict_single(self, customer_data: dict) -> dict:
        if not self.is_available():
            self._load_model()
            if not self.is_available():
                raise FileNotFoundError("Churn ML pipeline is not trained yet. Run 'python train_model.py'.")

        # Fill missing fields with defaults
        payload = dict(DEFAULT_CUSTOMER_PROFILE)
        payload.update(customer_data)

        # Enforce numerical types
        payload["tenure"] = max(0, int(payload.get("tenure", 0)))
        payload["MonthlyCharges"] = max(0.0, float(payload.get("MonthlyCharges", 0.0)))
        
        # Calculate TotalCharges if 0 or missing
        if "TotalCharges" not in payload or payload["TotalCharges"] == 0:
            payload["TotalCharges"] = round(payload["tenure"] * payload["MonthlyCharges"], 2)
        else:
            payload["TotalCharges"] = max(0.0, float(payload["TotalCharges"]))

        payload["SeniorCitizen"] = int(payload.get("SeniorCitizen", 0))

        # Create 1-row DataFrame matching training feature columns
        feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        df_input = pd.DataFrame([{col: payload[col] for col in feature_cols}])

        # Inference
        prob_churn = float(self.pipeline.predict_proba(df_input)[0, 1])
        prob_percent = round(prob_churn * 100, 1)
        pred_label = 1 if prob_churn >= 0.50 else 0

        # Risk Classification
        if prob_percent >= 60.0:
            risk_tier = "High Risk"
            risk_badge = "badge-danger"
        elif prob_percent >= 35.0:
            risk_tier = "Medium Risk"
            risk_badge = "badge-warning"
        else:
            risk_tier = "Low Risk"
            risk_badge = "badge-success"

        # Diagnose Risk Drivers & Generate Retention Strategies
        risk_drivers = []
        retention_actions = []

        if payload["Contract"] == "Month-to-month":
            risk_drivers.append("Month-to-month contract provides no commitment friction.")
            retention_actions.append("Offer a 15% discount on an annual contract upgrade to secure commitment.")

        if payload["tenure"] <= 6:
            risk_drivers.append(f"Early customer lifecycle (tenure = {payload['tenure']} months; highest churn hazard zone).")
            retention_actions.append("Enroll customer in VIP Onboarding Concierge program with priority support.")

        if payload["InternetService"] == "Fiber optic" and payload["MonthlyCharges"] > 75.0:
            risk_drivers.append(f"High recurring billing (${payload['MonthlyCharges']}/mo) on Fiber optic plan.")
            retention_actions.append("Review billing package or offer bundled streaming services to increase perceived value.")

        if payload["PaymentMethod"] == "Electronic check":
            risk_drivers.append("Manual Electronic check payment correlates with higher delinquency and cancellation.")
            retention_actions.append("Offer a $10 one-time bill credit for switching to Automatic Credit Card or Bank Transfer.")

        if payload["OnlineSecurity"] == "No" and payload["TechSupport"] == "No":
            risk_drivers.append("Customer has no Tech Support or Online Security attached.")
            retention_actions.append("Offer complimentary 6-month cybersecurity & technical protection trial.")

        if not risk_drivers:
            risk_drivers.append("Stable long-term account indicators with low churn volatility.")
            retention_actions.append("Maintain high customer satisfaction and offer loyalty referral rewards.")

        return {
            "prediction": "Will Churn" if pred_label == 1 else "Will Stay",
            "prediction_binary": pred_label,
            "churn_probability": prob_percent,
            "stay_probability": round(100.0 - prob_percent, 1),
            "risk_tier": risk_tier,
            "risk_badge": risk_badge,
            "risk_drivers": risk_drivers,
            "retention_actions": retention_actions,
            "customer_profile": payload,
            "model_metadata": {
                "model_name": self.metadata.get("selected_model", "Gradient Boosting Classifier") if self.metadata else "Gradient Boosting",
                "accuracy": self.metadata.get("metrics", {}).get("accuracy") if self.metadata else 0.801,
                "roc_auc": self.metadata.get("metrics", {}).get("roc_auc") if self.metadata else 0.843
            }
        }


predictor = ChurnPredictor()


if __name__ == "__main__":
    # Test High Risk Profile
    high_risk_sample = {
        "tenure": 2,
        "MonthlyCharges": 89.5,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "OnlineSecurity": "No",
        "TechSupport": "No"
    }
    res = predictor.predict_single(high_risk_sample)
    print("=" * 60)
    print("High Risk Sample Test Result:")
    print(json.dumps(res, indent=2))
