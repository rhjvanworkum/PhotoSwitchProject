# PhotoSwitchProject

A synthetic chemist's guide through organic photoswitch chemical space:
featurise molecules, predict their transition wavelengths, virtually screen a
generated library, and optimize candidates in a learned latent space.

Photoswitches are molecules with strong potential in solar energy storage,
photopharmacology and photoelectronics. Because the space of possible
photoswitches is enormous, this project builds a virtual-screening toolkit to
surface the most promising candidates — currently focused on hitting a desired
transition wavelength.

## Install

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # core: data, library generation, random forest, DFT parsing
uv sync --extra ml      # optional: Gaussian-process & neural-network models (TF/GPflow)
```

`uv sync` recreates the exact environment from `uv.lock` using the Python
version pinned in `.python-version` (3.10).

| Extra         | Adds                | Enables                                            |
| ------------- | ------------------- | -------------------------------------------------- |
| `ml`          | tensorflow, gpflow  | `models.gaussian_process`, `models.neural_network` |
| `torch`       | torch               | latent-space optimization (Optimization notebook)  |
| `descriptors` | mordred             | Mordred descriptor featurisation                   |
| `dft`         | chemml              | generating ORCA TD-DFT input files                 |

## Quickstart

```python
from photoswitch.data import load_features_and_labels
from photoswitch.models.random_forest import train_rf_model

# X: features, y: E-isomer pi-pi* transition wavelengths (nm)
X, y = load_features_and_labels(
    "data/sample/rdkit_descriptors_sample.csv",
    "data/sample/photoswitches_sample.csv",
    "e_iso_pi",
)
model, x_scaler, y_scaler = train_rf_model(X, y, n_estimators=20, n_folds=3)
```

## Examples

Runnable, self-contained scripts using the small bundled data in `data/sample/`.
Run them from the repo root:

```bash
uv run examples/generate_library.py        # combinatorial SMILES library generation
uv run examples/train_random_forest.py      # cross-validated random-forest wavelength model
uv run examples/parse_dft_output.py          # parse pi-pi*/n-pi* peaks from an ORCA output
uv run examples/train_gaussian_process.py    # Tanimoto-kernel GP model (needs: uv sync --extra ml)
```

## Notebooks

The original research workflow lives in the notebooks (run them after
`uv sync` so the `photoswitch` package is importable):

1. `Featurisation.ipynb` — generating input features (Morgan, RDKit, Mordred,
   MolBERT, JTNN).
2. `Features.ipynb` / `Feature_selection.ipynb` — EDA and feature selection.
3. `model_selection.ipynb` — comparing random forest, GP and neural network.
4. `Screening.ipynb` — generating a photoswitch library and selecting candidates.
5. `Optimization.ipynb` — optimizing over chemical space with a JT-VAE.

Some notebook cells rely on external tools that are **not** reproducible here
(the ORCA binary, and cloned MolBERT / Junction-Tree VAE repositories with
pretrained checkpoints); those cells are documented but not run in CI.

## Project structure

```
src/photoswitch/
  data.py            # load property labels and feature matrices
  preprocessing.py   # standard scaling, train/test splitting, PCA
  viz.py             # PCA scatter plots
  library.py         # combinatorial SMILES library generation (RDKit)
  dft.py             # ORCA TD-DFT input generation + output parsing
  models/
    common.py        # shared cross-validation loop
    random_forest.py # sklearn random forest        (core)
    gaussian_process.py  # Tanimoto-kernel GP        (ml extra)
    neural_network.py    # dense Keras network       (ml extra)
data/sample/         # tiny bundled data for examples and tests
examples/            # runnable example scripts
tests/               # pytest suite mirroring the package
raw_data/, processed_data/   # full datasets used by the notebooks
```

## Results (from the original study)

- A combination of Mordred descriptors and fine-tuned MolBERT fingerprints
  predicted transition wavelengths well.
- A Gaussian-process regressor with the Tanimoto kernel fit the (small) dataset
  best and provides calibrated uncertainty.
- Virtual screening was run for target wavelengths of 450 nm and 650 nm.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup and how to run tests,
lint, formatting and type checks. Licensed under the [MIT License](LICENSE).
