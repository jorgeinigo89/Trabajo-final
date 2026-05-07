# Bank Marketing — Classification Project

Predict whether a bank client will subscribe to a **term deposit** based on direct
phone-call marketing campaigns. Implements two classifiers (**Random Forest** and a
**PyTorch MLP**) with an interactive **Streamlit dashboard**, managed with **uv** for
dependency resolution and **DVC** for data versioning.

> **Course:** Module I — Professional Certificate in AI & LLMs in Financial Markets
> **Institution:** ITAM · **Lecturer:** Noe Hernandez · **Author:** Jorge Inigo

---

## Dataset

**Source:** [UCI ML Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

| Attribute | Value |
|---|---|
| Instances | 45,211 |
| Features | 16 |
| Task | Binary classification |
| Feature types | Categorical, Integer |
| Target | `y` — subscribed to term deposit? (`yes` / `no`) |
| Class balance | ~88 % "no" / ~12 % "yes" |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Raw data is tracked with **DVC** and is not stored in git.

---

## Models

Two classifiers are trained and evaluated interactively via the dashboard.

### Random Forest (scikit-learn) — default `n_estimators=100`, `max_depth=10`

| Metric | Value |
|---|---|
| Accuracy | 0.9055 |
| ROC-AUC | 0.9214 |
| Precision (yes) | 0.6890 |
| Recall (yes) | 0.3497 |
| F1-score (yes) | 0.4639 |

### Neural Network MLP (PyTorch) — default 30 epochs

Architecture: `Linear → BatchNorm → ReLU → Dropout(0.5)` × 3 layers (128 → 64 → 32 → 1).
Optimizer: `AdamW` · Loss: `BCEWithLogitsLoss` · Scheduler: `StepLR(step_size=10, γ=0.1)`.

| Metric | Value |
|---|---|
| Accuracy | 0.8040 |
| ROC-AUC | 0.8475 |
| Precision (yes) | 0.3403 |
| Recall (yes) | 0.7193 |
| F1-score (yes) | 0.4621 |

> Metrics on a held-out test set (20 % of data, `random_state=42`).
> RF has higher precision; MLP has higher recall for the minority class.

---

## Project Structure

```
.
├── data/                    # Raw data (git-ignored; managed by DVC)
│   ├── bank-full.csv.dvc    # DVC pointer to bank-full.csv
│   └── *.dvc                # Other DVC-tracked splits
├── src/mi_paquete/
│   ├── data/                # Data loading utilities
│   ├── features/            # Feature engineering & preprocessing
│   ├── models/              # Model training (RF + MLP)
│   ├── evaluation/          # Metrics (accuracy, ROC-AUC, F1, …)
│   └── app/
│       └── dashboard.py     # Streamlit dashboard
├── notebooks/               # Exploratory analysis (main.ipynb)
├── configs/                 # Pipeline configurations
├── Dockerfile               # Container image
├── pyproject.toml           # Project metadata & dependencies (uv)
├── uv.lock                  # Pinned dependency lockfile
└── dvc.yaml                 # DVC pipeline definition
```

---

## Quick Start

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **[DVC](https://dvc.org/)** — data version control (installed via uv)
- **Docker** (optional, for containerised deployment)

### 1 · Clone the repository

```bash
git clone https://github.com/jorgeinigo89/Trabajo-final.git
cd "Trabajo-final"
```

### 2 · Create the virtual environment and install dependencies

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock` and installs all dependencies into `.venv/`.

### 3 · Pull the dataset with DVC

The raw data is not stored in git. Download it from the DVC remote:

```bash
uv run dvc pull
```

> If the remote is not configured for your machine, add it first:
> ```bash
> uv run dvc remote add -d myremote <remote-url>
> uv run dvc pull
> ```

### 4 · Run the Streamlit dashboard

```bash
uv run streamlit run src/mi_paquete/app/dashboard.py
```

Open **http://localhost:8501** in your browser.

---

## Docker

Build and run the containerised dashboard:

```bash
# Build
docker build -t finance-risk-app:latest .

# Run
docker run -p 8501:8501 finance-risk-app:latest
```

Open **http://localhost:8501** in your browser.

---

## Development

### Run linting / formatting

```bash
uv run ruff check src/
uv run ruff format src/
```

### Run pre-commit hooks manually

```bash
uv run pre-commit run --all-files
```

### DVC pipeline

Reproduce the full data pipeline (split → train → evaluate):

```bash
uv run dvc repro
```

---

## Citation

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset].
UCI Machine Learning Repository. <https://doi.org/10.24432/C5K306>
