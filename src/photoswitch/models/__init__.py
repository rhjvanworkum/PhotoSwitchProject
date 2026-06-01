"""Property-prediction models.

Import the trainer you need from its submodule, e.g.::

    from photoswitch.models.random_forest import train_rf_model

The Gaussian-process and neural-network submodules require the ``ml`` extra
(``pip install photoswitch[ml]``); the random-forest submodule needs only the
core dependencies. Trainers are intentionally *not* re-exported here so that
importing this package does not pull in TensorFlow/GPflow.
"""
