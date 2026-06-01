"""Tools for navigating organic photoswitch chemical space.

The package bundles the building blocks used across the project notebooks:

* :mod:`photoswitch.utils` -- data loading, scaling and PCA helpers.
* :mod:`photoswitch.library` -- combinatorial SMILES library generation.
* :mod:`photoswitch.random_forest` / :mod:`photoswitch.gaussian_process` /
  :mod:`photoswitch.neural_network` -- property-prediction models.
* :mod:`photoswitch.dft` -- ORCA TD-DFT input generation and output parsing.

The model and DFT modules depend on optional extras (``ml``, ``torch``,
``descriptors``, ``dft``); importing them without those extras installed will
raise :class:`ImportError`.
"""

__version__ = "0.1.0"
