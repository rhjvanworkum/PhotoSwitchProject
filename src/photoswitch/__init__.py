"""Tools for navigating organic photoswitch chemical space.

The package bundles the building blocks used across the project notebooks:

* :mod:`photoswitch.data` -- loading property labels and feature matrices.
* :mod:`photoswitch.preprocessing` -- scaling and train/test splitting.
* :mod:`photoswitch.viz` -- PCA scatter plots of the feature space.
* :mod:`photoswitch.library` -- combinatorial SMILES library generation.
* :mod:`photoswitch.models` -- random-forest, Gaussian-process and
  neural-network property-prediction models.
* :mod:`photoswitch.dft` -- ORCA TD-DFT input generation and output parsing.

The Gaussian-process/neural-network models and the DFT input generator depend
on optional extras (``ml``, ``torch``, ``descriptors``, ``dft``); importing
them without those extras installed will raise :class:`ImportError`.
"""

__version__ = "0.1.0"
