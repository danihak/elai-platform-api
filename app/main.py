"""
main.py — the ELAI platform API.

One deployable service. Product routers are isolated modules over one write
model, so extracting any of them into its own service later is moving a
directory, not a rewrite.

Everything a client is ever shown passes through elai-confidence-core. No router
formats a number itself.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from elai_confidence_core import CURRENT

from .routers import governance, agent, coverage, farms, feedback, lenderbook, portfolio, trust

DESCRIPTION = """
The confidence state engine, exposed as an API.

Every number returned carries four things: the value, the confidence state that
governs it, a pre-formatted `display` string, and the `rule_version` that
produced it. Clients render `display` — they do not format numbers themselves,
because a client that formats can print a point value on a field nobody could
see, which is the defect this system exists to remove.

States:

* **Verified** — measured directly from clear optical.
* **Degraded** — estimated from radar or interpolation, with a stated error band.
* **Blind** — not observed, and no estimate is trustworthy enough to show.
"""

app = FastAPI(
    title="ELAI Platform API",
    description=DESCRIPTION,
    version="0.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# The console apps are separate Vercel projects on their own domains.
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(farms.router)
app.include_router(portfolio.router)
app.include_router(coverage.router)
app.include_router(agent.router)
app.include_router(feedback.router)
app.include_router(lenderbook.router)
app.include_router(trust.router)
app.include_router(governance.router)


@app.get("/healthz", tags=["ops"], summary="Liveness and readiness")
def healthz() -> dict:
    """Checked by the load balancer every 10 seconds.

    Verifies the core loaded and which rule version is live — not just that the
    process is running. A task serving the wrong rule version is worse than a
    task that is down.
    """
    from .store.repository import repo
    from .services.estimator import active_estimator

    est = active_estimator()
    return {
        "status": "ok",
        "rule_version": CURRENT.version,
        "core_states": ["Verified", "Degraded", "Blind"],
        "data_source": getattr(repo, "data_source", "unknown"),
        "estimator_version": getattr(est, "version", "unknown"),
        "validated_models": sorted(getattr(est, "VALIDATED", [])),
        "estimator_fallback_reason": getattr(est, "fallback_reason", "") or None,
        "estimator_mode_env": os.getenv("ELAI_ESTIMATOR", "auto"),
    }


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": "elai-platform-api", "docs": "/docs", "health": "/healthz"}
