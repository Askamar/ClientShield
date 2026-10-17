import os
import logging
from flask import Flask, jsonify, request, render_template

from predictor import predictor
from preprocess import load_and_clean_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("churn_system")

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts customer profile attributes and returns churn probability,
    risk level, identified risk drivers, and personalized retention actions.
    """
    try:
        data = request.get_json() or {}
        result = predictor.predict_single(data)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """
    Returns baseline statistics calculated from the Telco dataset.
    """
    try:
        df = load_and_clean_data()
        total_customers = len(df)
        churn_count = int(df["Churn_Binary"].sum())
        churn_rate = round(float(df["Churn_Binary"].mean()) * 100, 2)
        avg_tenure = round(float(df["tenure"].mean()), 1)
        avg_monthly = round(float(df["MonthlyCharges"].mean()), 2)

        # Contract Breakdown
        contract_dist = df.groupby("Contract")["Churn_Binary"].agg(["count", "mean"]).reset_index()
        contract_stats = [
            {
                "contract": r["Contract"],
                "customers": int(r["count"]),
                "churn_rate": round(float(r["mean"]) * 100, 1)
            }
            for _, r in contract_dist.iterrows()
        ]

        # Internet Service Breakdown
        internet_dist = df.groupby("InternetService")["Churn_Binary"].agg(["count", "mean"]).reset_index()
        internet_stats = [
            {
                "service": r["InternetService"],
                "customers": int(r["count"]),
                "churn_rate": round(float(r["mean"]) * 100, 1)
            }
            for _, r in internet_dist.iterrows()
        ]

        return jsonify({
            "total_customers": total_customers,
            "churn_count": churn_count,
            "retained_count": total_customers - churn_count,
            "churn_rate_percent": churn_rate,
            "avg_tenure_months": avg_tenure,
            "avg_monthly_charges": avg_monthly,
            "contract_breakdown": contract_stats,
            "internet_breakdown": internet_stats,
            "model_info": predictor.metadata if predictor.metadata else {}
        }), 200
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": "Failed to compute statistics", "details": str(e)}), 500


@app.route("/api/presets", methods=["GET"])
def get_presets():
    """
    Returns preset customer archetypes for rapid 1-click testing.
    """
    presets = [
        {
            "id": "high_risk",
            "name": "🚨 High Risk: New Fiber User (Month-to-Month)",
            "description": "Short tenure (2 months), high monthly charges on Fiber optic with no tech support or security.",
            "data": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 94.5,
                "TotalCharges": 189.0
            }
        },
        {
            "id": "moderate_risk",
            "name": "⚠️ Moderate Risk: Mid-Tenure Streamer",
            "description": "Moderate tenure (18 months), DSL plan, month-to-month contract with bank transfer.",
            "data": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 18,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 64.0,
                "TotalCharges": 1152.0
            }
        },
        {
            "id": "loyal_low_risk",
            "name": "🛡️ Low Risk: 2-Year Contract Veteran",
            "description": "Long tenure (60 months), Two-year contract, bundled security, credit card auto-pay.",
            "data": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 60,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 105.0,
                "TotalCharges": 6300.0
            }
        }
    ]
    return jsonify(presets), 200


if __name__ == "__main__":
    port = int(os.environ.get("CHURN_PORT", 5001))
    print(f"Starting Customer Churn Prediction Web Server on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)
