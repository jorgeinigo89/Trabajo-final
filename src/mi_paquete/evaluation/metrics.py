"""Evaluation helpers for classification models."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute key classification metrics.

    Returns a dict with:
        accuracy, roc_auc, confusion_matrix, classification_report (as string)
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["no", "yes"]
        ),
        "fpr": fpr,
        "tpr": tpr,
    }


def evaluate_mlp(
    model: torch.nn.Module,
    scaler,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a BankMLP model.

    Returns a dict with:
        accuracy, roc_auc, confusion_matrix, classification_report (as string)
    """
    X_t = torch.tensor(scaler.transform(X_test).tolist(), dtype=torch.float32)
    y_true = y_test.values

    model.eval()
    with torch.no_grad():
        probs_tensor = torch.sigmoid(model(X_t).squeeze())
        probs: np.ndarray = np.array([x.item() for x in probs_tensor])

    preds = (probs >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_true, probs)

    return {
        "accuracy": accuracy_score(y_true, preds),
        "roc_auc": roc_auc_score(y_true, probs),
        "confusion_matrix": confusion_matrix(y_true, preds),
        "classification_report": classification_report(
            y_true, preds, target_names=["no", "yes"]
        ),
        "fpr": fpr,
        "tpr": tpr,
    }
