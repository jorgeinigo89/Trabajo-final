"""Training utilities for the Bank Marketing classification task."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Return (X_train, X_test, y_train, y_test)."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


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


def feature_importances(
    model: RandomForestClassifier, feature_names: list
) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances."""
    return (
        pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ── Neural Network (MLP) ──────────────────────────────────────────────────────


class BankMLP(nn.Module):
    """Three-hidden-layer MLP for binary classification (Bank Marketing)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_epochs: int = 30,
    lr: float = 1e-3,
    random_state: int = 42,
) -> tuple:
    """Fit a BankMLP; return (model, scaler, train_losses, val_losses)."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    scaler = StandardScaler()
    X_tr = torch.tensor(scaler.fit_transform(X_train).tolist(), dtype=torch.float32)
    X_v = torch.tensor(scaler.transform(X_val).tolist(), dtype=torch.float32)
    y_tr = torch.tensor(y_train.values.tolist(), dtype=torch.float32)
    y_v = torch.tensor(y_val.values.tolist(), dtype=torch.float32)

    model = BankMLP(X_tr.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr).squeeze(), y_tr)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v).squeeze(), y_v)
        val_losses.append(val_loss.item())

    return model, scaler, train_losses, val_losses
