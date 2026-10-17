# Customer Churn Prediction System - Interview Preparation Guide

This document contains comprehensive preparation materials for technical and behavioral interviews regarding the **Customer Churn Prediction System**.

---

## 🎯 60-Second Elevator Pitch

> *"The Customer Churn Prediction System is an end-to-end Machine Learning web application designed to proactively identify telecommunications subscribers at risk of canceling their subscriptions before they leave.*
> 
> *Trained on 7,043 subscriber records with 21 behavioral, contractual, and demographic features, the system features a leak-free Scikit-Learn preprocessing pipeline with numerical standardization and one-hot encoding. I benchmarked Logistic Regression, Random Forest, and Gradient Boosting, with Gradient Boosting emerging as the production winner achieving **80.06% accuracy** and an **0.8433 ROC-AUC** on held-out test data.*
> 
> *Beyond raw probabilities, the platform features an explainability engine that extracts individual risk factors—such as month-to-month contracts or high monthly charges without security add-ons—and maps them directly to tailored retention playbooks. The Flask-powered web interface and REST API enable customer success teams to assess subscriber risk in under 100 milliseconds and trigger targeted interventions."*

---

## 💡 Top 20 Technical Interview Questions & Answers

### 1. Problem Framing & Business Value

#### Q1: Why is customer churn prediction critical for a subscription business?
**Answer:**
- In subscription and SaaS businesses, **Customer Acquisition Cost (CAC)** is typically 5x to 7x higher than **Customer Retention Cost**.
- A high churn rate creates a "leaky bucket" where revenue gains from new customer sign-ups are negated by cancellations.
- Churn prediction shifts retention strategy from **reactive** (attempting to win back a customer after they have requested cancellation, which has a very low success rate) to **proactive** (intervening when risk signals first emerge).

#### Q2: How do you translate churn probability into measurable business ROI?
**Answer:**
- By segmenting customers into probability tiers ($\ge 60\%$ High Risk, $35\%-60\%$ Medium Risk, $< 35\%$ Low Risk), we allocate limited retention budgets efficiently.
- Offering discounts to customers with low churn risk is wasteful, while offering discounts to high-risk customers with high Customer Lifetime Value (LTV) protects recurring revenue.
- Formula: $\text{Expected Value} = P(\text{Churn}) \times \text{LTV} \times \text{Campaign Success Rate} - \text{Intervention Cost}$.

---

### 2. Data Engineering & Preprocessing

#### Q3: What anomalies or data hygiene issues were found in the dataset, and how did you resolve them?
**Answer:**
- In the IBM Telco dataset, the column `TotalCharges` is stored as an object/string rather than float because 11 records contained blank whitespace (`" "`).
- Crucially, inspecting these 11 rows revealed that all had `tenure == 0`, representing newly acquired customers who had not yet completed their first monthly billing cycle.
- Instead of dropping them or imputing with the global mean/median (which would introduce significant bias), I converted the blanks to `0.0`, reflecting their true historical billing reality.
- `customerID` was removed because it is a random alphanumeric string with zero predictive generalization power and would risk overfitting if hashed or encoded.

#### Q4: Why use Scikit-Learn's `ColumnTransformer` and `Pipeline` instead of manual pandas transformations?
**Answer:**
- **Data Leakage Prevention**: When scaling features using `StandardScaler`, fitting parameters ($\mu, \sigma$) must be learned **strictly on the training partition** and then applied via `transform()` to the test set and production inference. Fitting on the entire dataset leaks test distribution statistics into the training process.
- **Reproducibility & Serialization**: Encapsulating the preprocessing within a unified `Pipeline` ensures that during live inference, raw JSON input passes through the exact same transformations without manual synchronization.
- **Handling Unseen Categories**: `OneHotEncoder(handle_unknown='ignore')` gracefully handles novel categorical values in production without throwing runtime exceptions.

---

### 3. Modeling & Evaluation

#### Q5: Walk through your model selection process. Why Gradient Boosting over Random Forest or Logistic Regression?
**Answer:**
- I evaluated three distinct model families on a stratified 80/20 train/test split:
  1. **Logistic Regression (Baseline)**: 73.81% accuracy, 0.8417 ROC-AUC, 79.41% recall, but low precision (50.41%).
  2. **Random Forest Classifier**: 75.59% accuracy, 0.8401 ROC-AUC, 74.33% recall, 52.88% precision.
  3. **Gradient Boosting Classifier (Production)**: **80.06% accuracy**, **0.8433 ROC-AUC**, 65.41% precision.
- **Decision Rationale**:
  - Gradient Boosting achieved the highest discrimination power (**0.8433 ROC-AUC**), meaning it ranks churners above non-churners more reliably across all threshold cutoffs.
  - In our business scenario, a 65.41% precision significantly reduces false alarms. High false positive rates lead to unnecessary promotional expenditure (e.g., granting discount credits to customers who were never planning to churn).

#### Q6: How do you handle class imbalance in this dataset?
**Answer:**
- The dataset has a 73.5% / 26.5% non-churn vs. churn ratio (approx. 2.8:1).
- While not extreme (such as fraud detection at 0.1%), standard models can bias toward the majority class if not monitored.
- I addressed this by:
  1. Using **Stratified splitting** (`stratify=y`) to ensure training and test folds maintain the exact 26.54% class distribution.
  2. Monitoring **ROC-AUC**, **Precision**, and **Recall** rather than raw accuracy alone.
  3. Calibrating the decision threshold in the inference engine: by outputting raw calibrated probabilities $P(\text{Churn})$, business stakeholders can adjust the decision boundary according to campaign capacity.

#### Q7: What is the difference between Precision and Recall in churn prediction, and how do you choose a threshold?
**Answer:**
- **Recall (Sensitivity)**: What percentage of actual churners did we catch? High recall ensures very few churners slip through undetected.
- **Precision**: When we predict a customer will churn, how often are we right? High precision ensures retention team time and budget are not wasted on loyal customers.
- **Threshold Selection**:
  - If the retention action is **low cost** (e.g., automated email with feature tips), we lower the threshold to 0.35 to maximize recall.
  - If the retention action is **high cost** (e.g., sending an account executive or offering a 30% discount), we raise the threshold to 0.60 to maximize precision.

---

### 4. Feature Importance & Explainability

#### Q8: What are the strongest indicators of customer churn in this dataset?
**Answer:**
1. **Contract Type**: `Month-to-month` subscribers have a **42.7% churn rate**, compared to **11.3%** for One-year and **2.8%** for Two-year contracts.
2. **Tenure**: Customers in their first 6 months have the highest cancellation hazard. Once tenure passes 24 months, churn drops dramatically.
3. **Internet Service Type**: `Fiber optic` users exhibit a **41.9% churn rate** versus **19.0%** for DSL and **7.4%** for No Internet. This is often driven by high prices combined with lack of technical support or setup friction.
4. **Payment Method**: `Electronic check` users churn at **45.3%**, whereas automated bank transfer and credit card auto-pay churn at under **17%**.
5. **Add-on Services**: Lack of `OnlineSecurity`, `TechSupport`, or `DeviceProtection` heavily correlates with churn. Bundled services create product stickiness and higher switching friction.

#### Q9: How did you implement real-time explainability without adding excessive latency?
**Answer:**
- Heavy explainability tools like full Kernel SHAP can add 500ms–2000ms of compute time per prediction, which degrades interactive web UI responsiveness.
- To maintain sub-10ms response times, I implemented a hybrid rule-assisted feature attribution engine in `predictor.py`:
  - It checks critical high-hazard feature thresholds (contract type, tenure hazard zone, monthly charge spikes, payment friction, missing support services).
  - It maps these specific customer attributes into human-readable risk drivers and matches them with concrete retention playbooks.

---

### 5. System Architecture & Engineering

#### Q10: Describe the system architecture and data flow.
**Answer:**
- **Data Layer**: Cleaned tabular data loaded through `preprocess.py`.
- **Pipeline Layer**: Serialized scikit-learn `Pipeline` (`churn_pipeline.pkl`) containing preprocessing transformers and the trained Gradient Boosting estimator.
- **Inference Layer (`predictor.py`)**: Singleton class that loads model artifacts once into memory on startup, validates input dictionaries, runs `predict_proba()`, computes risk tiers, and generates action recommendations.
- **Web & API Layer (`app.py`)**: Flask web service exposing REST endpoints (`/api/predict`, `/api/stats`, `/api/presets`) with structured JSON responses and error handling.
- **Frontend Layer (`index.html`, `app.js`, `style.css`)**: Vanilla JavaScript SPA utilizing SVG/Canvas radial gauges, interactive range sliders, and dynamic DOM rendering.

#### Q11: How do you ensure the model doesn't crash on incomplete or missing API payloads?
**Answer:**
- In `predictor.py`, I implemented a comprehensive fallback default profile:
  ```python
  DEFAULT_PROFILE = {
      "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
      "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
      "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
      "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
      "StreamingMovies": "No", "Contract": "Month-to-month",
      "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
      "MonthlyCharges": 65.0, "TotalCharges": 780.0
  }
  ```
- If an API caller omits fields or passes `None`, defaults are merged safely, types are coerced (`int`, `float`), and `TotalCharges` is dynamically computed if zero.

#### Q12: How would you monitor this model for drift once deployed in production?
**Answer:**
- **Data Drift (Covariate Shift)**: Track population distributions of incoming features (e.g., changes in average `MonthlyCharges` or shifts from DSL to Fiber optic) using Kolmogorov-Smirnov tests or Population Stability Index (PSI).
- **Concept Drift**: Track the relationship between features and target over time. When a competitor launches a new promotion, churn patterns may alter even if customer demographic features remain identical.
- **Performance Monitoring**: Set up a feedback loop logging customer outcomes after 30/60/90 days to compare predicted probabilities against actual churn events, recalculating Brier score and ROC-AUC monthly.
- **Automated Retraining**: Schedule quarterly model retraining runs using a rolling window of the most recent 12 months of customer data.

---

## 📋 Quick Technical Fact Sheet

| Dimension | Metric / Specification |
| :--- | :--- |
| **Dataset Records** | 7,043 rows, 21 columns |
| **Target Distribution** | 26.54% Churn (1,869), 73.46% Retained (5,174) |
| **Best Model** | Gradient Boosting Classifier |
| **Accuracy** | 80.06% |
| **ROC-AUC** | 0.8433 |
| **Precision** | 65.41% |
| **Test Set Split** | Stratified 80% Train (5,634) / 20% Test (1,409) |
| **Preprocessing** | `StandardScaler` + `OneHotEncoder(handle_unknown='ignore')` |
| **API Latency** | ~4-8 ms per single-customer inference |
| **Web Server Port** | `http://127.0.0.1:5001` |
| **Automated Tests** | 6 unit tests (100% pass in 0.12s) |
