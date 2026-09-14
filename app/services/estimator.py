"""
estimator.py — the estimator the demo runs with.

In production this is loaded from the model registry. Here it is a coefficient
table standing in for one that the DS lead has fitted and validated against
held-out clear optical passes.

THE COEFFICIENTS AND RMSE VALUES BELOW ARE FABRICATED. They are plausible in
shape — RMSE tightening as canopy closes, widening at senescence — but they are
not measured, and nothing derived from them should be shown to a client or
quoted as an accuracy figure. `demo_fitted` is in the version string so it is
visible in every reason line and every ledger row.

The unfitted default in the core is what runs when this is absent, and it sends
everything to Blind. That is deliberate: availability of radar is not validation
of radar.
"""

from __future__ import annotations

import os

from elai_confidence_core.estimators import BaselineSarRegression


class DemoFittedEstimator(BaselineSarRegression):
    version = "1.0.0-demo_fitted"

    COEFFICIENTS = {
        # crop:stage -> (slope, intercept, measured_rmse)
        "maize:vegetative":   (0.0295, 1.080, 0.082),
        "maize:flowering":    (0.0288, 1.065, 0.074),
        "maize:grain_fill":   (0.0280, 1.060, 0.070),
        "maize:maturation":   (0.0272, 1.045, 0.095),
        "cotton:vegetative":  (0.0301, 1.095, 0.089),
        "cotton:flowering":   (0.0292, 1.070, 0.078),
        "cotton:grain_fill":  (0.0285, 1.058, 0.081),
    }
    VALIDATED = list(COEFFICIENTS.keys())


class UnfittedEstimator(BaselineSarRegression):
    """The honest starting position: radar available, model not yet proven."""
    version = "0.1.0-unfitted"


def active_estimator():
    """`ELAI_ESTIMATOR=unfitted` shows what the platform does before the model exists."""
    if os.getenv("ELAI_ESTIMATOR", "demo_fitted") == "unfitted":
        return UnfittedEstimator()
    return DemoFittedEstimator()
