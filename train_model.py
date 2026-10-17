import os
import sys
import json
import datetime
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

from preprocess import load_and_clean_data, build_preprocessor, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PIPELINE_FILE = os.path.join(MODEL_DIR, "churn_pipeline.pkl")
METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.json")


def train_churn_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 65)
    print("  Customer Churn Prediction - ML Training & Benchmark")
    print("=" * 65)

    # 1. Load Data
    df = load_and_clean_data()
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df["Churn_Binary"]

    print(f"Loaded {len(df)} customer records.")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # 2. Stratified Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train split: {len(X_train)} samples ({y_train.sum()} churners)")
    print(f"Test split:  {len(X_test)} samples ({y_test.sum()} churners)")

    # 3. Model Candidates
    candidates = {
        "Logistic Regression (Baseline)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_split=5, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting Classifier (Production)": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42
        )
    }

    results = {}
    fitted_pipelines = {}

    print("\n" + "-" * 65)
    print(f"  {'Model':<35} | {'Acc':<6} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'ROC-AUC':<7}")
    print(f"  {'-'*35}-|-{'-'*6}-|-{'-'*6}-|-{'-'*6}-|-{'-'*6}-|-{'-'*7}")

    for name, clf in candidates.items():
        preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4)
        }
        fitted_pipelines[name] = pipeline

        print(f"  {name:<35} | {acc:.3f}  | {prec:.3f}  | {rec:.3f}  | {f1:.3f}  | {auc:.4f}")

    print("-" * 65 + "\n")

    # Select Production Model (Gradient Boosting Classifier provides highest ROC-AUC)
    selected_name = "Gradient Boosting Classifier (Production)"
    selected_pipeline = fitted_pipelines[selected_name]

    # Serialize Best Pipeline
    joblib.dump(selected_pipeline, PIPELINE_FILE)
    print(f"Saved production pipeline to: {PIPELINE_FILE}")

    # Feature importances from Gradient Boosting
    gb_model = selected_pipeline.named_steps["classifier"]
    encoded_feature_names = selected_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = gb_model.feature_importances_
    top_feature_indices = np.argsort(importances)[::-1][:10]
    top_features = [
        {"feature": str(encoded_feature_names[i]), "importance": round(float(importances[i]), 4)}
        for i in top_feature_indices
    ]

    metadata = {
        "selected_model": selected_name,
        "trained_at": datetime.datetime.now().isoformat(),
        "dataset_rows": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES
        },
        "metrics": results[selected_name],
        "benchmark_comparison": results,
        "top_features": top_features
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to: {METADATA_FILE}")

    return selected_pipeline, metadata


if __name__ == "__main__":
    train_churn_models()
