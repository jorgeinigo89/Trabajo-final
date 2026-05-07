# Bank Marketing — Classification Project

Implementation of a neural network with **PyTorch** to predict whether a client will subscribe to a term deposit, based on direct marketing campaigns (phone calls) from a Portuguese banking institution.

Project structure based on MLOps with Python.

## Dataset

**Source:** [UCI ML Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

| Attribute | Value |
|---|---|
| Instances | 45,211 |
| Features | 16 |
| Task type | Binary classification |
| Feature types | Categorical, Integer |
| Target variable | `y` — whether the client subscribed a term deposit (`yes`/`no`) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The data is related to direct marketing campaigns of a Portuguese banking institution. The goal is to predict whether the client will subscribe a term deposit based on socioeconomic attributes and contact history.

## Models

Two classifiers are implemented and compared via the Streamlit dashboard:

### Random Forest (scikit-learn)

Default hyperparameters: `n_estimators=100`, `max_depth=10`.

| Metric | Value |
|---|---|
| Accuracy | 0.9055 |
| ROC-AUC | 0.9214 |
| Precision (yes) | 0.6890 |
| Recall (yes) | 0.3497 |
| F1-score (yes) | 0.4639 |

### Neural Network MLP (PyTorch)

Architecture: `Linear(input→128) → BatchNorm → ReLU → Dropout(0.5)` × 3 hidden layers → `Linear(32→1)`.
Trained with `BCEWithLogitsLoss`, `AdamW`, `StepLR(step_size=10, gamma=0.1)` for 30 epochs.

| Metric | Value |
|---|---|
| Accuracy | 0.8040 |
| ROC-AUC | 0.8475 |
| Precision (yes) | 0.3403 |
| Recall (yes) | 0.7193 |
| F1-score (yes) | 0.4621 |

> Metrics computed on a held-out test set (20 % of the full dataset, `random_state=42`).
> The RF achieves higher precision while the MLP achieves higher recall for the minority class ("yes").

## Project Structure

```
src/
  mi_paquete/
    data/         # Data loading and preprocessing
    features/     # Feature engineering
    models/       # Model definition and training
    evaluation/   # Metrics and evaluation
    app/          # Streamlit interface
notebooks/        # Experimentation and exploratory analysis
configs/          # Model and pipeline configurations
data/             # Raw and processed data
models/           # Trained model artifacts
```

## Installation

```bash
pip install -e .
