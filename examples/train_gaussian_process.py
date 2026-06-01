"""Train a Tanimoto-kernel Gaussian process on the bundled photoswitch sample data.

Loads the small RDKit-descriptor features and E-isomer pi-pi* wavelengths, runs a
tiny cross-validation, and prints a couple of GP mean/variance predictions.
"""
# Run with: uv run examples/train_gaussian_process.py

from __future__ import annotations

from pathlib import Path

from photoswitch.data import load_features_and_labels
from photoswitch.models.gaussian_process import train_gp_model

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "sample"


def main() -> None:
    feature_file = SAMPLE_DIR / "rdkit_descriptors_sample.csv"
    smiles_file = SAMPLE_DIR / "photoswitches_sample.csv"

    X, y = load_features_and_labels(str(feature_file), str(smiles_file), task="e_iso_pi")
    X = X.astype(float)
    y = y.astype(float)
    print(f"Loaded {X.shape[0]} molecules with {X.shape[1]} features each.")

    model, x_scaler, y_scaler = train_gp_model(X, y, n_folds=2, test_set_size=0.3)

    print("Sample GP predictions (scaled feature space):")
    mean, var = model.predict_f(x_scaler.transform(X[:3]))
    mean = mean.numpy().ravel()
    var = var.numpy().ravel()
    for i in range(len(mean)):
        print(f"  mol {i}: mean={mean[i]:.4f}  variance={var[i]:.4f}")


if __name__ == "__main__":
    main()
