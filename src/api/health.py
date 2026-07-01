# =============================================================================
# src/api/health.py — Health Check Endpoint
# =============================================================================
# Context:
#   Simple liveness probe. Returns 200 with system status so load balancers
#   and container orchestrators know the app is running.
#
# Architectural Reasoning:
#   - Separate router from the main app for modularity. Adding a new
#     endpoint group means creating a new router file, not bloating
#     main.py. This scales to dozens of endpoints without pain.
#   - Returns version number so CI/CD can verify the correct deployment
#     is serving traffic. Without it, you can't tell if your new release
#     is actually live.
#   - No external dependencies (no DB check, no LLM ping). A health
#     check should never fail because a downstream service is down —
#     that's what separate readiness probes are for.
#
# Trade-offs:
#   - No readiness vs liveness distinction. Kubernetes supports both,
#     but combining them into one endpoint is fine for this portfolio's
#     scope. In production, /health (liveness) and /ready (readiness +
#     dependency checks) would be separate.
#
# Opinionated Stance:
#   Every API needs a health check. If your service doesn't have one,
#   you can't deploy it in Kubernetes, and if you can't deploy it in
#   Kubernetes, it's not production-ready. This is table stakes.
# =============================================================================

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}
