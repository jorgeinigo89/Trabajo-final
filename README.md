# Bank Marketing — Neural Network Classifier

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

## Model

Neural network implemented with **PyTorch** for binary classification.

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