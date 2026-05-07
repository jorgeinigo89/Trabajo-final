"""Evaluation helpers for classification models."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute key classification metrics.

    Returns a dict with:
        accuracy, roc_auc, confusion_matrix, classification_report (as string)
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["no", "yes"]
        ),
    }
