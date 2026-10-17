import unittest
import json
from app import app

class TestCustomerChurnAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_page(self):
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Customer Churn", response.data)

    def test_get_presets(self):
        response = self.app.get("/api/presets")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 3)
        self.assertIn("id", data[0])
        self.assertIn("data", data[0])

    def test_get_stats(self):
        response = self.app.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["total_customers"], 7043)
        self.assertIn("churn_rate_percent", data)
        self.assertIn("contract_breakdown", data)
        self.assertIn("internet_breakdown", data)
        self.assertIn("model_info", data)

    def test_predict_high_risk(self):
        sample = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 1,
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
            "MonthlyCharges": 95.0,
            "TotalCharges": 95.0
        }
        response = self.app.post("/api/predict", data=json.dumps(sample), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        res = json.loads(response.data)
        self.assertIn("churn_probability", res)
        self.assertIn("risk_tier", res)
        self.assertIn("risk_drivers", res)
        self.assertIn("retention_actions", res)
        self.assertGreaterEqual(res["churn_probability"], 60.0)
        self.assertEqual(res["risk_tier"], "High Risk")
        self.assertEqual(res["prediction_binary"], 1)

    def test_predict_low_risk(self):
        sample = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 65,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 45.0,
            "TotalCharges": 2925.0
        }
        response = self.app.post("/api/predict", data=json.dumps(sample), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        res = json.loads(response.data)
        self.assertIn("churn_probability", res)
        self.assertIn("risk_tier", res)
        self.assertLessEqual(res["churn_probability"], 35.0)
        self.assertEqual(res["risk_tier"], "Low Risk")
        self.assertEqual(res["prediction_binary"], 0)

    def test_predict_empty_payload_fallback(self):
        response = self.app.post("/api/predict", data=json.dumps({}), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        res = json.loads(response.data)
        self.assertIn("churn_probability", res)
        self.assertIn("risk_tier", res)
        self.assertIn("stay_probability", res)

if __name__ == "__main__":
    unittest.main()
