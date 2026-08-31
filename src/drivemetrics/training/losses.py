"""The single pinned loss policy shared by every approved P1 training run."""

from __future__ import annotations

IGNORE_INDEX = 255


def cross_entropy_spec() -> dict[str, object]:
    """Return an independent copy of the approved loss configuration.

    ``ignore_index`` must equal the protocol mask pad value, so symmetric
    padding never contributes a loss term and padded pixels are never counted as
    real classification errors.
    """

    return {"loss": "cross_entropy", "ignore_index": IGNORE_INDEX, "reduction": "mean"}
