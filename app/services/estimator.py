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
    #: Why the real fitted table was not used. Empty when chosen deliberately.
    fallback_reason = ""

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

            # A model that was fitted but did NOT clear the decision-grade
            # ceiling is not usable. Returning it anyway and letting the engine
            # judge invites exactly the accident found on the first real run:
            # cotton was rejected at RMSE 0.1309, then a 15% multi-source
            # "narrowing" I had invented brought it to 0.111 and slipped it
            # under the 0.12 gate. A rejected model reached a client because of
            # a factor nobody measured.
            if key not in self.VALIDATED:
                continue

            value = model["intercept"] + sum(
                c * feats[f] for f, c in model["coefficients"].items()
            )

            # The uncertainty is the measured held-out RMSE, used as measured.
            # No narrowing for extra radar sources: that band came from a
            # hold-out set, and shrinking it with an unmeasured heuristic is
            # over-claiming of the precise kind this system exists to stop.
            # If fusing sources genuinely narrows the error, fit the fused
            # model and measure it.
            return Estimate(
                ndvi=max(0.0, min(1.0, value)),
                uncertainty=model["rmse"],
                model_key=f"radar_multivariate[{key}]",
                model_version=self.version,
                sources_used=source_keys,
                validated_for=[f"{crop}:{stage}", f"{crop}:*"],
            )
        return None


class FusionEstimator(FittedFromData):
    """The full stack: phenology prior, persistence, and radar, fused by variance.

    Radar no longer has to produce the answer alone. It contributes one of three
    weak sources, weighted by its measured error, and the posterior is tighter
    than any input. This is what brings cotton back from Blind — a radar sigma of
    0.131 is unusable by itself but still informative when weighted honestly.

    The claimed band is then INFLATED by the measured calibration factor from
    `python -m app.cli fit`. Held-out validation showed the raw fusion is
    overconfident by roughly 1.65x on anomalous fields, which are the only fields
    that matter. Keeping the flattering number would be the exact over-claim this
    system exists to prevent.
    """

    version = "2.0.0-fusion"

    def __init__(self) -> None:
        super().__init__()
        import json
        import pathlib as _p

        blob = json.loads(
            (_p.Path(__file__).parent.parent.parent / "data" / "climatology.json")
            .read_text(encoding="utf-8")
        )
        self.CLIMATOLOGY = blob["climatology"]
        self.DECAY = {c: d.get("decay_per_day")
                      for c, d in blob["persistence_decay"].items() if d.get("fitted")}
        self.CALIBRATION = {c: v.get("calibration", 1.0)
                            for c, v in blob["fusion_calibration"].items()
                            if v.get("validated")}

    def estimate(self, *, crop, stage, sar_vv, sar_vh, source_keys,
                 last_clear_ndvi, days_since_clear, obs_date=None, **_):
        from elai_confidence_core.estimators import Estimate
        from .fusion import build_components, describe, fuse, inflate_for_calibration

        # Radar leg — only from a model that cleared validation.
        radar_ndvi = radar_sigma = None
        radar_detail = ""
        base = super().estimate(
            crop=crop, stage=stage, sar_vv=sar_vv, sar_vh=sar_vh,
            source_keys=source_keys, last_clear_ndvi=last_clear_ndvi,
            days_since_clear=days_since_clear,
        )
        if base is not None:
            radar_ndvi, radar_sigma = base.ndvi, base.uncertainty
            radar_detail = base.model_key

        decay = self.DECAY.get(crop)
        if decay is None:
            # Persistence cannot be weighted without a measured decay, so it is
            # omitted rather than guessed.
            decay = 0.0

        components = build_components(
            obs_date=obs_date, crop=crop,
            radar_ndvi=radar_ndvi, radar_sigma=radar_sigma, radar_detail=radar_detail,
            last_clear_ndvi=last_clear_ndvi if decay else None,
            days_since_clear=days_since_clear,
            climatology_model=self.CLIMATOLOGY.get(crop),
            persistence_decay=decay or 0.015,
        )
        est = fuse(components)
        if est is None:
            return None

        sigma = inflate_for_calibration(est.sigma, self.CALIBRATION.get(crop, 1.0))

        return Estimate(
            ndvi=est.ndvi,
            uncertainty=sigma,
            model_key=f"fusion[{est.dominant} dominant]",
            model_version=self.version,
            sources_used=list(source_keys) + [c.name for c in est.components],
            # Validated for a crop once the fusion itself has been calibrated for
            # it — not merely because a radar model exists.
            validated_for=([f"{crop}:{stage}", f"{crop}:*"]
                           if crop in self.CALIBRATION else []),
        )


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

    data = pathlib.Path(__file__).parent.parent.parent / "data"
    coef, clim = data / "coefficients.json", data / "climatology.json"

    if mode == "radar_only" and coef.exists():
        return FittedFromData()

    if coef.exists() and clim.exists():
        try:
            return FusionEstimator()
        except Exception as exc:  # noqa: BLE001
            est = DemoFittedEstimator()
            est.fallback_reason = f"fusion unavailable: {type(exc).__name__}: {exc}"
            return est

    if coef.exists():
        try:
            return FittedFromData()
        except Exception as exc:  # noqa: BLE001 — startup must not break
            # Previously this swallowed the error and quietly served the
            # FABRICATED table instead. The service looked healthy while every
            # number came from invented coefficients, and there was no way to
            # tell from outside. Falling back is still right; doing it silently
            # is not. The reason now surfaces on /healthz.
            est = DemoFittedEstimator()
            est.fallback_reason = f"{type(exc).__name__}: {exc}"
            return est
    est = DemoFittedEstimator()
    est.fallback_reason = f"no coefficients.json at {coef}"
    return est
