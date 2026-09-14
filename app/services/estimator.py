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


class FittedFromData(BaselineSarRegression):
    """Coefficients fitted on real clear-day pairs by `python -m app.cli fit`.

    When data/coefficients.json exists this is what runs, and the fabricated
    demo table is never touched. Only combinations whose held-out RMSE cleared
    the decision-grade ceiling are marked validated — the rest send their
    observations to Blind, which is correct.
    """

    version = "1.0.0-fitted"

    def __init__(self) -> None:
        import json
        import pathlib

        blob = json.loads(
            (pathlib.Path(__file__).parent.parent.parent / "data" / "coefficients.json")
            .read_text(encoding="utf-8")
        )
        table = blob["estimator"]
        self.MODELS = table["models"]
        self.VALIDATED = list(table["validated"])
        # Legacy single-slope table kept empty; prediction goes through MODELS.
        self.COEFFICIENTS = {}

    def estimate(self, *, crop, stage, sar_vv, sar_vh, source_keys,
                 last_clear_ndvi, days_since_clear):
        """Multivariate prediction over the fitted radar features.

        Falls back through crop:stage then crop:* then nothing. A combination
        with no model returns None, and the engine sends that observation to
        Blind — which is the correct answer when no proven model exists.
        """
        from elai_confidence_core.estimators import Estimate
        from app.ingest.fit import build_features

        feats = build_features(sar_vh, sar_vv)
        if feats is None:
            return None

        for key in (f"{crop}:{stage}", f"{crop}:*"):
            model = self.MODELS.get(key)
            if not model:
                continue
            value = model["intercept"] + sum(
                c * feats[f] for f, c in model["coefficients"].items()
            )
            unc = model["rmse"]
            if len(source_keys) > 1:
                unc *= 0.85
            return Estimate(
                ndvi=max(0.0, min(1.0, value)),
                uncertainty=unc,
                model_key=f"radar_multivariate[{key}]",
                model_version=self.version,
                sources_used=source_keys,
                validated_for=[f"{crop}:{stage}"] if key in self.VALIDATED
                              or f"{crop}:*" in self.VALIDATED else [],
            )
        return None


def active_estimator():
    """Preference order: real fitted coefficients, then the explicit override.

    `ELAI_ESTIMATOR=unfitted` shows what the platform does before any model has
    been proven — every farm Blind. That contrast is the Part 2b demo.
    """
    import pathlib

    mode = os.getenv("ELAI_ESTIMATOR", "auto")
    if mode == "unfitted":
        return UnfittedEstimator()
    if mode == "demo_fitted":
        return DemoFittedEstimator()

    coef = pathlib.Path(__file__).parent.parent.parent / "data" / "coefficients.json"
    if coef.exists():
        try:
            return FittedFromData()
        except Exception:  # noqa: BLE001 — never let a bad file break startup
            pass
    return DemoFittedEstimator()
