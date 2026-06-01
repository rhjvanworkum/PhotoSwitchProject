"""Shared fixtures for the test suite.

The fixtures build *tiny* CSV files that mirror the real data schema
(``raw_data/photoswitches.csv`` and the descriptor files under
``processed_data/``) so that tests never touch the large bundled data files or
the network.
"""

from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# A handful of valid SMILES; the labels intentionally contain one NaN so tests
# can exercise the invalid-row dropping in load_features_and_labels.
SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CN1N=NC(=N1)N=NC2=CC=CC=C2", "CCN"]
E_ISO_PI = [310.0, 320.0, float("nan"), 305.0, 330.0]
Z_ISO_PI = [290.0, 295.0, 280.0, 286.0, float("nan")]


@pytest.fixture
def photoswitch_csv(tmp_path: Path) -> Path:
    """A miniature stand-in for raw_data/photoswitches.csv.

    Keeps the leading unnamed index column and the ``SMILES`` column, plus the
    two wavelength columns the tests use.
    """
    df = pd.DataFrame(
        {
            "SMILES": SMILES,
            "E isomer pi-pi* wavelength in nm": E_ISO_PI,
            "Z isomer pi-pi* wavelength in nm": Z_ISO_PI,
        }
    )
    path = tmp_path / "photoswitches.csv"
    df.to_csv(path)  # writes the unnamed leading index column, like the real file
    return path


@pytest.fixture
def feature_csv(tmp_path: Path) -> Path:
    """A miniature feature file: leading index column, SMILES, then features."""
    df = pd.DataFrame(
        {
            "SMILES": SMILES,
            "f0": [1.0, 2.0, 3.0, 4.0, 5.0],
            "f1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "f2": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    path = tmp_path / "features.csv"
    df.to_csv(path)
    return path


@pytest.fixture
def sample_orca_out() -> Path:
    """Path to a small ORCA TD-DFT output file with an absorption-spectrum block."""
    return FIXTURES / "sample_orca.out"
