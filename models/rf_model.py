import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report
)


def train_random_forest(X_train, y_train, random_state=42):
    """
    Train the tuned Random Forest classifier with balanced class weights.
    """
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=40,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_random_forest(model, X_test, y_test):
    """
    Evaluate the model on the test set and return key metrics and predictions.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'F1-Score': round(f1_score(y_test, y_pred), 4)
    }

    print("=== RANDOM FOREST TEST SET EVALUATION ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics, y_pred, y_proba


def get_risk_scores(model, X_new):
    """
    Calculate diabetes risk probabilities (0-100%) and assign risk tiers.
    """
    probabilities = model.predict_proba(X_new)[:, 1]
    risk_pct = (probabilities * 100).round(1)

    def assign_tier(prob):
        if prob < 0.35:
            return "Low Risk"
        elif prob < 0.65:
            return "Moderate Risk"
        return "High Risk"

    return pd.DataFrame({
        'Risk_Score_Pct': risk_pct,
        'Risk_Tier': [assign_tier(p) for p in probabilities]
    })


def save_model(model, filepath="models/random_forest.joblib", compress=3):
    """
    Save the trained model using joblib with compression to stay well below GitHub's 50 MiB limit.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath, compress=compress)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Model saved to {filepath} (Size: {size_mb:.2f} MiB)")