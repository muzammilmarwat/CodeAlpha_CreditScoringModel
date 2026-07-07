"""Factory functions for reusable feature scalers.

These helpers return scaler objects without fitting or transforming any data.
They are intended for use inside the shared preprocessing pipeline.
"""

from __future__ import annotations

from sklearn.preprocessing import RobustScaler, StandardScaler


def build_standard_scaler() -> StandardScaler:
    """Return a standard scaler configured for later fitting.

    Returns:
        StandardScaler: A reusable scaler object.
    """
    return StandardScaler()


def build_robust_scaler() -> RobustScaler:
    """Return a robust scaler configured for later fitting.

    Returns:
        RobustScaler: A reusable scaler object.
    """
    return RobustScaler()
