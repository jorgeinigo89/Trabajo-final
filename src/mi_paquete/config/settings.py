from pathlib import Path

import yaml

# Project root is three levels up from this file: config/ -> mi_paquete/ -> src/ -> root
_ROOT = Path(__file__).resolve().parents[3]
_HYPERPARAMS_PATH = _ROOT / "configs" / "hyperparameters.yaml"


def load_hyperparams(path: Path = _HYPERPARAMS_PATH) -> dict:
    """Load hyperparameters from the YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
