"""
model.py
========
Factory function for building a ``RandomForestClassifier`` that complies with
every strict experimental constraint of this project:

    * ``random_state=RANDOM_SEED`` (1234) is always set.
    * Only ``max_depth`` is exposed as a tunable parameter.
    * No other hyperparameters (n_estimators, min_samples_leaf, …) are tuned.

Usage
-----
    from src.model import build_rf

    model = build_rf()           # max_depth=None  (unlimited)
    model = build_rf(max_depth=5)
"""

from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_SEED


def build_rf(max_depth: int | None = None) -> RandomForestClassifier:
    """Create a ``RandomForestClassifier`` with the project's fixed seed.

    Parameters
    ----------
    max_depth : int | None
        Maximum depth of each decision tree.  Must be one of the values
        defined in ``config.MAX_DEPTH_VALUES`` (3, 5, or ``None``).
        ``None`` (default) means nodes are expanded until all leaves are
        pure or contain fewer than ``min_samples_split`` samples.

    Returns
    -------
    RandomForestClassifier
        An *unfitted* Random Forest instance ready for cross-validation.

    Notes
    -----
    Per the project's Strict Requirements:

    * ``random_state`` is **always** ``RANDOM_SEED`` (1234).
    * No other hyperparameter (``n_estimators``, ``min_samples_leaf``, etc.)
      is varied.  Sklearn defaults are used as-is to keep the experiment
      focused on ``max_depth`` and ``k`` only.
    """
    return RandomForestClassifier(
        max_depth=max_depth,
        random_state=RANDOM_SEED,
    )
