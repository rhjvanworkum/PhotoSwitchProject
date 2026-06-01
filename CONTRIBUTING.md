# Contributing

Thanks for your interest in improving PhotoSwitchProject! This project uses
[uv](https://docs.astral.sh/uv/) for environment management.

## Set up the dev environment

```bash
# install uv (see https://docs.astral.sh/uv/getting-started/installation/)
uv sync                 # core deps + dev tools, into .venv, from uv.lock
uv sync --extra ml      # add the Gaussian-process / neural-network stack (optional)
```

`uv sync` reads the pinned Python version from `.python-version` and the locked
dependencies from `uv.lock`, so everyone gets the same environment.

Optional extras:

| Extra         | Pulls in            | Needed for                                    |
| ------------- | ------------------- | --------------------------------------------- |
| `ml`          | tensorflow, gpflow  | `models.gaussian_process`, `models.neural_network` |
| `torch`       | torch               | the latent-space optimization notebook        |
| `descriptors` | mordred             | Mordred descriptor featurisation              |
| `dft`         | chemml              | generating ORCA TD-DFT input files            |

## Run the checks

```bash
uv run pytest                 # tests (GP/NN tests are skipped without the ml extra)
uv run ruff check .           # lint
uv run ruff format --check .  # formatting
uv run mypy                   # type checking (lenient; tighten over time)
```

`ruff format .` (without `--check`) applies formatting fixes; `ruff check --fix .`
applies safe lint fixes.

## Conventions

- Source lives under `src/photoswitch/`; tests mirror it under `tests/`.
- Keep I/O, core logic and configuration separated; prefer small functions with
  explicit inputs/outputs over hidden global state.
- Add type hints and a docstring to new public functions.
- New behavior needs a test. When fixing a bug, add a regression test first.
- Tests must not touch the network or the large data files under `raw_data/` and
  `processed_data/` — use the tiny fixtures in `tests/` or `data/sample/`.

## Things that can't run in CI

Some functionality depends on external tools that are intentionally **not** part
of the dependency graph and are not exercised in CI:

- `photoswitch.dft.run_orca_calculation` shells out to the external **ORCA**
  quantum-chemistry program.
- The MolBERT and Junction-Tree VAE pipelines in the notebooks clone and run
  external repositories and pretrained models.

Please don't add tests that require these.
