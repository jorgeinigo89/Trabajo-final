"""Feature engineering for the Bank Marketing dataset."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
TARGET_COL = "y"


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Label-encode all categorical columns and binary-encode the target.

    Returns:
        df_encoded: DataFrame ready for modelling.
        target_encoder: fitted LabelEncoder for the target column.
    """
    df_encoded = df.copy()

    for col in CATEGORICAL_COLS:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    target_encoder = LabelEncoder()
    df_encoded[TARGET_COL] = target_encoder.fit_transform(
        df_encoded[TARGET_COL].astype(str)
    )

    return df_encoded, target_encoder


def get_X_y(df_encoded: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split encoded dataframe into features X and target y."""
    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL]
    return X, y
