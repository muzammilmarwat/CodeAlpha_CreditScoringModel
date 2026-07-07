"""Reusable ordinal encoder definitions for the German Credit preprocessing pipeline.

The project uses explicit category ordering for ordinal variables. These orders
are declared here so that training, evaluation, and future inference all use
identical semantics.
"""

from __future__ import annotations

import logging
from typing import Sequence

from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)

# The ordering below is intentionally explicit and does not rely on the order
# observed in the dataset. It should be reviewed by a domain expert before any
# production training run.
CHECKING_ACCOUNT_STATUS_ORDER: Sequence[str] = (
    "A11",
    "A12",
    "A13",
    "A14",
)

# Employment duration is treated as an ordered measure of stability. The list
# is ordered from shorter tenure to longer tenure.
EMPLOYMENT_DURATION_ORDER: Sequence[str] = (
    "A71",
    "A72",
    "A73",
    "A74",
    "A75",
)

# Savings balance is treated as an ordered proxy for financial resilience.
SAVINGS_ACCOUNT_ORDER: Sequence[str] = (
    "A61",
    "A62",
    "A63",
    "A64",
    "A65",
)


def build_ordinal_encoder() -> OrdinalEncoder:
    """Create an ordinal encoder with explicit categories for ordinal features.

    Returns:
        OrdinalEncoder: A configured encoder that is ready to be used inside a
            scikit-learn pipeline. It is not fitted here.
    """
    categories = [
        list(CHECKING_ACCOUNT_STATUS_ORDER),
        list(EMPLOYMENT_DURATION_ORDER),
        list(SAVINGS_ACCOUNT_ORDER),
    ]

    logger.info("Configured ordinal encoder with explicit category ordering.")
    return OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=-1)
