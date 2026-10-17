# Customer Churn Prediction System

An end-to-end Machine Learning web platform that predicts customer churn probability for telecommunications subscribers using real historical behavioral, contractual, and demographic data. It provides real-time risk tiering, automated identification of top churn drivers, and actionable, personalized customer retention playbooks.

---

## 🚀 Key Features

- **Trained on Real Benchmark Data**: Built on the IBM Telco Customer Churn dataset with 7,043 subscriber records and 21 attributes.
- **Robust Preprocessing Pipeline**: Scikit-Learn `ColumnTransformer` with `StandardScaler` for numeric distributions and `OneHotEncoder(handle_unknown='ignore')` for multi-class categorical features.
- **Multi-Model Benchmarking**: Evaluated Logistic Regression, Random Forest, and Gradient Boosting Classifiers using stratified 80/20 train/test splitting.
- **Production Gradient Boosting Classifier**: Achieves **80.06% Accuracy** and **0.8433 ROC-AUC** on held-out test data.
- **Explainability & Risk Driver Engine**: Deconstructs subscriber traits in real-time to isolate the primary churn hazards (e.g., month-to-month contracts, electronic check billing, fiber optic without security add-ons).
- **Automated Retention Playbook**: Generates targeted customer success interventions based on individual risk profiles.
- **Interactive Dark-Mode Dashboard**: Sleek web interface running on Flask with 1-click test archetypes (High Risk, Moderate Risk, Loyal Low Risk) and instant radial probability gauges.
- **RESTful API**: Fast JSON endpoints for single-customer churn scoring, cohort analytics, and preset scenarios.

---

## 📊 Dataset Profile & Preprocessing

- **Source**: IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Total Records**: 7,043 customers
- **Target Variable**: `Churn` (1,869 Churned = 26.54%, 5,174 Retained = 73.46%)
- **Key Data Cleaning Steps**:
  - `TotalCharges` contained 11 blank whitespace values (`" "`) corresponding to brand-new customers with `tenure == 0`. These were converted to numeric and filled with `0.0`.
  - Non-predictive identifier `customerID` was removed from the feature matrix.
  - Target `Churn` ("Yes"/"No") was encoded to binary `1`/`0`.

### Feature Engineering & Transformation Matrix:
- **Numerical Features** (`StandardScaler`):
  - `tenure`: Months the customer has stayed with the company.
  - `MonthlyCharges`: Current monthly fee amount.
  - `TotalCharges`: Cumulative charges incurred to date.
- **Categorical Features** (`OneHotEncoder`):
  - Demographic: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Core Services: `PhoneService`, `MultipleLines`, `InternetService`
  - Value-Add Services: `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  - Billing & Contract: `Contract`, `PaperlessBilling`, `PaymentMethod`

---

## 🏆 Model Evaluation & Benchmarks

All models were evaluated on an identical, held-out stratified test set of **1,409 subscribers (20%)**:

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** (Baseline) | 73.81% | 50.41% | **79.41%** | 0.6169 | 0.8417 | Benchmark |
| **Random Forest Classifier** | 75.59% | 52.88% | 74.33% | 0.6179 | 0.8401 | Benchmark |
| **Gradient Boosting Classifier** | **80.06%** | **65.41%** | 52.67% | **0.5837** | **0.8433** | **Selected Production Model** |

### Why Gradient Boosting?
- Highest overall accuracy (80.06%) and superior ROC-AUC (0.8433), showing superior ranking and discrimination ability across all classification thresholds.
- Higher precision (65.41%) significantly reduces false positives, preventing wasteful discount expenditures on customers who were not actually planning to leave.

---

## 🛡️ Risk Tiering & Business Logic

The inference engine translates predicted probabilities $P(\text{Churn} = 1)$ into business action tiers:

| Churn Probability | Risk Tier | Business Response |
| :---: | :---: | :--- |
| **$\ge$ 60%** | **High Risk** | Immediate account manager intervention, retention discounts, loyalty incentives. |
| **35% – 59.9%** | **Medium Risk** | Proactive onboarding outreach, feature adoption campaigns, payment method migration. |
| **$<$ 35%** | **Low Risk** | Standard service, upsell opportunities, loyalty milestone rewards. |

---

## 📁 Project Structure

```text
Customer_churn_prediction_System/
│
├── app.py                     # Flask web server and REST API routes
├── predictor.py               # Inference engine, risk tiering & retention recommender
├── preprocess.py              # Data cleaning, type conversion, and ColumnTransformer pipeline
├── train_model.py             # Model training, multi-model evaluation, serialization
├── inspect_data.py            # Diagnostic script for dataset inspection
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # 7,043 telco customer records
│
├── models/
│   ├── churn_pipeline.pkl     # Production trained GradientBoosting pipeline
│   └── model_metadata.json    # Model metadata, benchmark scores, feature list
│
├── static/
│   ├── css/
│   │   └── style.css          # Glassmorphism dark-mode responsive design
│   └── js/
│       └── app.js             # Client-side reactivity, API calls, dynamic chart gauge
│
├── templates/
│   └── index.html             # Single-page dashboard interface
│
├── tests/
│   └── test_api.py            # Automated test suite (100% passing)
│
└── venv/                      # Python virtual environment
```

---

## 🔌 API Reference

### 1. Predict Churn
- **Endpoint**: `POST /api/predict`
- **Payload Example**:
  ```json
  {
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
  ```
- **Response**:
  ```json
  {
    "churn_probability": 82.2,
    "stay_probability": 17.8,
    "prediction": "Will Churn",
    "prediction_binary": 1,
    "risk_tier": "High Risk",
    "risk_badge": "badge-danger",
    "risk_drivers": [
      "Month-to-month contract provides no commitment friction.",
      "Early customer lifecycle (tenure = 2 months; highest churn hazard zone).",
      "High recurring billing ($94.5/mo) on Fiber optic plan.",
      "Manual Electronic check payment correlates with higher delinquency and cancellation.",
      "Customer has no Tech Support or Online Security attached."
    ],
    "retention_actions": [
      "Offer a 15% discount on an annual contract upgrade to secure commitment.",
      "Enroll customer in VIP Onboarding Concierge program with priority support.",
      "Review billing package or offer bundled streaming services to increase perceived value.",
      "Offer a $10 one-time bill credit for switching to Automatic Credit Card or Bank Transfer.",
      "Offer complimentary 6-month cybersecurity & technical protection trial."
    ],
    "model_metadata": {
      "model_name": "Gradient Boosting Classifier (Production)",
      "accuracy": 0.8006,
      "roc_auc": 0.8433
    }
  }
  ```

### 2. Dataset & Cohort Statistics
- **Endpoint**: `GET /api/stats`
- **Response**: Summary metrics including total customer base (7,043), overall churn rate (26.54%), average tenure (32.4 months), and churn breakdowns by contract and internet service.

### 3. Preset Scenarios
- **Endpoint**: `GET /api/presets`
- **Response**: 3 pre-configured customer profiles for instantaneous UI testing.

---

## 🛠️ Installation & Running Locally

### 1. Prerequisites
- Python 3.10+ installed
- Virtual environment created and activated

### 2. Install Dependencies
```powershell
.\venv\Scripts\pip install flask scikit-learn pandas numpy joblib
```

### 3. Train the Model
```powershell
.\venv\Scripts\python train_model.py
```

### 4. Run Automated Tests
```powershell
.\venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
```

### 5. Launch the Web Application
```powershell
.\venv\Scripts\python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5001`**
