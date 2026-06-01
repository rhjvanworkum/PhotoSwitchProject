"""Train a random-forest regressor on the bundled photoswitch sample data.

Loads precomputed RDKit descriptors plus E-isomer pi-pi* wavelength labels,
then cross-validates a small random forest and prints a summary.
"""

# Run with: uv run examples/train_random_forest.py

from __future__ import annotations

from pathlib import Path

from photoswitch.data import load_features_and_labels
from photoswitch.models.random_forest import train_rf_model

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"
FEATURE_FILE = SAMPLE_DIR / "rdkit_descriptors_sample.csv"
SMILES_FILE = SAMPLE_DIR / "photoswitches_sample.csv"
TASK = "e_iso_pi"


def main() -> None:
    X, y = load_features_and_labels(str(FEATURE_FILE), str(SMILES_FILE), TASK)
    print(f"Loaded features X with shape {X.shape}")
    print(f"Loaded labels y with shape {y.shape}")
    print(f"Training random forest for task '{TASK}'...")

    model, x_scaler, y_scaler = train_rf_model(
        X,
        y,
        n_estimators=20,
        n_folds=3,
        test_set_size=0.3,
    )

    print(
        f"Summary: trained {type(model).__name__} with "
        f"{model.n_estimators} trees on {X.shape[0]} samples "
        f"({X.shape[1]} features); see cross-validated metrics above."
    )


if __name__ == "__main__":
    main()
