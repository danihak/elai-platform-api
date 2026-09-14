"""
repository.py — the store.

In-memory for now, behind an interface, so the Mongo and Postgres adapters slot
in without touching a router. Two properties are implemented here rather than
deferred, because they are the ones the product argument depends on:

1. Observations are append-only. Nothing mutates one.
2. Every computed state is written to the ledger with the rule version and
   estimator version that produced it, so September's answer is reproducible in
   March. This is what ClaimChronos and the RBI evidence pack are built on.

The read cache is keyed with the rule version, so a threshold change is a cache
miss by construction and there is no purge job to forget.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

from elai_confidence_core import CURRENT, Observation, ScoredObservation, score
from elai_confidence_core.estimators import NdviEstimator

from ..services.estimator import active_estimator

from ..seed import FARMS, FARMS_BY_ID, Farm, observations_for

#: The season date the demo runs "as of". Fixed so the fixtures are stable.
AS_OF = date(2026, 9, 6)


@dataclass(frozen=True)
class LedgerEntry:
    written_at: str
    farm_id: str
    obs_date: str
    das: int
    state: str
    rung_used: str
    source: str
    valid_pixel_fraction: float
    cloudy: bool
    days_since_last_clear_optical: int
    reason: str
    rule_version: str
    actor: str


class Repository:
    def __init__(self, estimator: Optional[NdviEstimator] = None) -> None:
        self._lock = threading.Lock()
        self._estimator = estimator or active_estimator()
        self._observations: Dict[str, List[Observation]] = {}
        self._ledger: List[LedgerEntry] = []
        self._cache: Dict[Tuple[str, str], List[ScoredObservation]] = {}
        self._feedback: List[dict] = []
        self._load()

    # ------------------------------------------------------------------ setup

    def _load(self) -> None:
        """Real Sentinel observations when they have been ingested; synthetic otherwise.

        `python -m app.cli ingest` writes data/observations.json. Once it exists,
        every number the platform serves traces to an actual satellite pass over
        an actual polygon. Without it the service still runs on the synthetic
        seed, and `data_source` says so on every response.
        """
        import json
        import pathlib

        real = pathlib.Path(__file__).parent.parent.parent / "data" / "observations.json"
        if real.exists():
            try:
                self._load_real(json.loads(real.read_text(encoding="utf-8")))
                self.data_source = "sentinel-hub"
                return
            except Exception:  # noqa: BLE001 — never let a bad file break startup
                pass

        for farm in FARMS:
            self._observations[farm.farm_id] = observations_for(farm, AS_OF)
        self.data_source = "synthetic"

    def _load_real(self, bundle: dict) -> None:
        """Rebuild Observations from the ingested rows.

        The cached file holds one row per interval. The confidence engine also
        needs the gap BEFORE each reading, so that is recomputed here by walking
        the series in date order — the same quantity the engine uses to decide
        when interpolation has gone stale.
        """
        from datetime import date as _date

        for farm in FARMS:
            rows = bundle.get("farms", {}).get(farm.farm_id, [])
            series: List[Observation] = []
            last_date = last_das = last_ndvi = None

            for r in sorted(rows, key=lambda x: x["date"]):
                d = _date.fromisoformat(r["date"])
                das = r.get("das", (d - farm.sowing_date).days)
                gap = 999 if last_date is None else (d - last_date).days

                series.append(Observation(
                    farm_id=farm.farm_id,
                    obs_date=d,
                    das=das,
                    crop=farm.crop,
                    stage=r.get("stage", "unknown"),
                    valid_pixel_fraction=r.get("valid_pixel_fraction", 0.0),
                    mean_cloud_probability=r.get("mean_cloud_probability", 1.0),
                    cloudy=r.get("cloudy", True),
                    ndvi=r.get("ndvi"),
                    ndwi=r.get("ndwi"),
                    gci=r.get("gci"),
                    sar_available=r.get("sar_vh") is not None,
                    sar_vv=r.get("sar_vv"),
                    sar_vh=r.get("sar_vh"),
                    radar_sources=tuple(r.get("radar_sources") or ()),
                    days_since_last_clear_optical=gap,
                    last_clear_optical_date=last_date,
                    last_clear_optical_das=last_das,
                    last_clear_optical_ndvi=last_ndvi,
                ))

                if r.get("ndvi") is not None and r.get("valid_pixel_fraction", 0) >= 0.5:
                    last_date, last_das, last_ndvi = d, das, r["ndvi"]

            self._observations[farm.farm_id] = series

    # ------------------------------------------------------------------ farms

    def farms(self, **filters) -> List[Farm]:
        out = list(FARMS)
        for key, value in filters.items():
            if value is None:
                continue
            out = [f for f in out if str(getattr(f, key, "")).lower() == str(value).lower()]
        return out

    def farm(self, farm_id: str) -> Optional[Farm]:
        return FARMS_BY_ID.get(farm_id)

    # ------------------------------------------------------- observations

    def raw_observations(self, farm_id: str) -> List[Observation]:
        return list(self._observations.get(farm_id, []))

    def scored(self, farm_id: str, estimator: Optional[NdviEstimator] = None) -> List[ScoredObservation]:
        """Score a farm's season, cached by (farm_id, rule_version)."""
        est = estimator or self._estimator
        key = (farm_id, f"{CURRENT.version}:{getattr(est, 'version', 'na')}")

        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                return hit

            out = [score(o, CURRENT, est) for o in self._observations.get(farm_id, [])]
            self._cache[key] = out

            now = datetime.now(timezone.utc).isoformat()
            for s in out:
                self._ledger.append(LedgerEntry(
                    written_at=now,
                    farm_id=s.farm_id,
                    obs_date=s.obs_date.isoformat(),
                    das=s.das,
                    state=s.state,
                    rung_used=s.rung_used,
                    source=s.source,
                    valid_pixel_fraction=s.valid_pixel_fraction,
                    cloudy=s.cloudy,
                    days_since_last_clear_optical=s.days_since_last_clear_optical,
                    reason=s.reason,
                    rule_version=s.rule_version,
                    actor="compute-worker",
                ))
            return out

    def latest(self, farm_id: str, estimator: Optional[NdviEstimator] = None) -> Optional[ScoredObservation]:
        series = self.scored(farm_id, estimator)
        return series[-1] if series else None

    # ------------------------------------------------------------ ledger

    def ledger(self, farm_id: Optional[str] = None) -> List[dict]:
        rows = self._ledger if farm_id is None else [e for e in self._ledger if e.farm_id == farm_id]
        return [asdict(e) for e in rows]

    # ---------------------------------------------------------- feedback

    def add_feedback(self, record: dict) -> None:
        """Feedback is an ingress, not an endpoint.

        A farmer's reply enters the same pipeline as a satellite pass, with the
        responder recorded, and invalidates the cache so the badge can move.
        """
        with self._lock:
            record["received_at"] = datetime.now(timezone.utc).isoformat()
            self._feedback.append(record)
            self._cache = {k: v for k, v in self._cache.items() if k[0] != record["farm_id"]}
            self._ledger.append(LedgerEntry(
                written_at=record["received_at"],
                farm_id=record["farm_id"],
                obs_date=record.get("obs_date", ""),
                das=record.get("das", 0),
                state="ground_truth",
                rung_used="field_feedback",
                source=record.get("responder", "farmer"),
                valid_pixel_fraction=1.0,
                cloudy=False,
                days_since_last_clear_optical=0,
                reason=record.get("note", "two-tap reply"),
                rule_version=CURRENT.version,
                actor=record.get("responder", "farmer"),
            ))

    def feedback(self, farm_id: Optional[str] = None) -> List[dict]:
        if farm_id is None:
            return list(self._feedback)
        return [f for f in self._feedback if f["farm_id"] == farm_id]


repo = Repository()
