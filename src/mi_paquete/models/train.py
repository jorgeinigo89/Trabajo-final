"""Training utilities for the Bank Marketing classification task."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Return (X_train, X_test, y_train, y_test)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Fit and return a Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def feature_importances(model: RandomForestClassifier, feature_names: list) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances."""
    return (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
