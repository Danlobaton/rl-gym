"""Procedurally generate an incident's materialized state from a seed.

Produces a per-episode workdir at <base_dir>/ep-<seed>/ containing:
  logs/<service>.log              plain text, one log line per entry
  metrics/<metric>.csv            t,value time series
  kubectl_describe.txt            pod-status blob

The contents are deterministic from the seed (so `replay` keeps working) and
causally consistent with the hidden root cause (so the agent has real evidence
to reason from instead of canned strings).
"""
import csv
import hashlib
import random
import shutil
from pathlib import Path

from incidents import Incident


def subrng(parent: int | str | random.Random, label: str) -> random.Random:
    """Derive a child RNG from a parent (seed / string / RNG), namespaced by `label`.

    Composable: pass a derived RNG back in to nest namespaces.
        rng_inc  = subrng(seed, "incident")
        rng_logs = subrng(rng_inc, "logs")
        rng_aff  = subrng(rng_logs, "affected")

    Collision-safe: same (parent, label) → same child; different labels under the
    same parent → different children. Identical labels at different nesting
    levels don't bleed because their parents differ.

    Discipline: don't mix draws and derivations on the same parent. Either treat
    an RNG as a *namespace* (only call subrng on it) or as a *source of values*
    (only call .choice / .randint / etc. on it), not both. Mixing them couples
    children's seeds to whether you drew from the parent first, and reordering
    generation code silently changes outputs.
    """
    if isinstance(parent, random.Random):
        parent_id = parent.getstate()
    else:
        parent_id = parent
    blob = repr((parent_id, label)).encode("utf-8")
    sub_seed = int.from_bytes(hashlib.sha256(blob).digest()[:8], "big", signed=False)
    return random.Random(sub_seed)


# --- catalog -------------------------------------------------------------

ROOT_CAUSES = ("db_pool_exhausted", "oom_killed", "bad_deploy", "missing_config")

CORRECT_ACTIONS = {
    "db_pool_exhausted": "scale_db_pool",
    "oom_killed": "increase_memory_limit",
    "bad_deploy": "rollback_deploy",
    "missing_config": "restore_config",
}

INCIDENT_TYPE = {
    "db_pool_exhausted": "high_latency",
    "oom_killed": "crashloop",
    "bad_deploy": "error_spike",
    "missing_config": "crashloop",
}

SERVICES = (
    "payments-api", "checkout-api", "search-svc", "auth-service",
    "user-service", "notification-svc", "inventory-svc", "recommendation-svc",
)


# --- entry point ---------------------------------------------------------


def generate_incident(seed: int, base_dir: Path) -> Incident:
    """Generate an incident's materialized state into base_dir/ep-<seed>/."""
    workdir = base_dir / f"ep-{seed}"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "logs").mkdir()
    (workdir / "metrics").mkdir()

    rng = subrng(seed, "incident")
    root_cause = subrng(rng, "root_cause").choice(ROOT_CAUSES)
    affected = subrng(rng, "affected_service").choice(SERVICES)

    related_pool = [s for s in SERVICES if s != affected]
    rng_related = subrng(rng, "related")
    n_related = subrng(rng_related, "count").randint(2, 3)
    related = subrng(rng_related, "pick").sample(related_pool, n_related)

    # Logs: affected gets telltale signal embedded in noise; related get benign.
    rng_logs = subrng(rng, "logs")
    logs: dict[str, list[str]] = {
        affected: _gen_affected_logs(root_cause, affected, subrng(rng_logs, "affected")),
    }
    for svc in related:
        logs[svc] = _gen_benign_logs(svc, subrng(rng_logs, f"related/{svc}"))
    for svc, lines in logs.items():
        (workdir / "logs" / f"{svc}.log").write_text("\n".join(lines) + "\n")

    # Metrics: shape matches the failure mode.
    metrics = _gen_metrics(root_cause, subrng(rng, "metrics"))
    for name, values in metrics.items():
        with (workdir / "metrics" / f"{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "value"])
            for i, v in enumerate(values):
                w.writerow([i, v])

    describe = _gen_kubectl_describe(root_cause, affected, subrng(rng, "kubectl"))
    (workdir / "kubectl_describe.txt").write_text(describe)

    alert = _gen_alert_text(root_cause, affected, subrng(rng, "alert"))

    return Incident(
        incident_type=INCIDENT_TYPE[root_cause],
        affected_service=affected,
        alert_text=alert,
        root_cause=root_cause,
        correct_action=CORRECT_ACTIONS[root_cause],
        workdir=workdir,
    )


# --- per-component generators -------------------------------------------


def _gen_alert_text(root_cause: str, service: str, rng: random.Random) -> str:
    if root_cause == "db_pool_exhausted":
        slo = rng.choice([200, 250, 300])
        actual = rng.randint(2000, 5000)
        return f"{service} p99 latency > {actual}ms for {rng.randint(3, 10)}min (SLO: {slo}ms)"
    if root_cause == "oom_killed":
        return f"{service} pod in CrashLoopBackOff ({rng.randint(3, 7)} restarts in {rng.randint(4, 10)}min)"
    if root_cause == "bad_deploy":
        rate = rng.randint(8, 15)
        baseline = rng.choice([0.05, 0.1, 0.2])
        return f"{service} 5xx error rate exceeded {rate}% (baseline {baseline}%)"
    if root_cause == "missing_config":
        return f"{service} fails to start ({rng.randint(4, 8)} restart attempts)"
    raise ValueError(f"unknown root_cause: {root_cause}")


def _gen_affected_logs(root_cause: str, service: str, rng: random.Random) -> list[str]:
    """Telltale signal embedded in normal-traffic noise. The agent has to read
    past the first few lines, not just grep for ERROR."""
    if root_cause == "missing_config":
        # Service crashes at startup — short log, no business traffic.
        missing = rng.choice(["DATABASE_URL", "API_KEY", "REDIS_HOST", "SERVICE_TOKEN", "SMTP_HOST"])
        return [
            f"[INFO] {service}: starting up",
            f"[INFO] {service}: reading config from /etc/{service}/config",
            f"[ERROR] {service}: required env var {missing} not set",
            f"[FATAL] {service}: cannot start without {missing}, exit 1",
        ]

    n = rng.randint(20, 35)
    lines = [
        rng.choice([
            f"[INFO] {service}: handling request id={rng.randint(10000, 99999)}",
            f"[INFO] {service}: GET /healthz 200",
            f"[INFO] {service}: cache hit ratio {rng.uniform(0.85, 0.99):.2f}",
            f"[INFO] {service}: heartbeat ok",
        ])
        for _ in range(n)
    ]

    if root_cause == "db_pool_exhausted":
        lines.extend([
            f"[WARN] {service}: db pool {rng.randint(40, 49)}/50, queueing requests",
            f"[WARN] {service}: db pool 50/50, queueing requests",
            f"[ERROR] {service}: timeout acquiring db connection after 5000ms",
            f"[ERROR] {service}: request id={rng.randint(10000, 99999)} failed: ConnectionTimeoutError",
            f"[ERROR] {service}: HikariCP timeout: 30000ms exceeded",
        ])
    elif root_cause == "oom_killed":
        # Memory-pressure warnings, then logs cut off (process killed mid-write).
        lines.extend([
            f"[INFO] {service}: heap usage 78%",
            f"[WARN] {service}: heap usage 92%, GC pressure rising",
            f"[WARN] {service}: heap usage 96%, full GC took {rng.randint(800, 2000)}ms",
        ])
    elif root_cause == "bad_deploy":
        sha = "".join(rng.choices("0123456789abcdef", k=7))
        deploy_t = rng.randint(5, 15)
        lines.extend([
            f"[INFO] {service}: deploy started build sha={sha} at t={deploy_t}",
            f"[INFO] {service}: rolling restart finished at t={deploy_t + 2}",
            f"[ERROR] {service}: NullPointerException in OrderHandler.process line {rng.randint(100, 250)}",
            f"[ERROR] {service}: 500 returned for POST /api/v1/order",
            f"[ERROR] {service}: NullPointerException in OrderHandler.process line {rng.randint(100, 250)}",
        ])

    rng.shuffle(lines)
    return lines


def _gen_benign_logs(service: str, rng: random.Random) -> list[str]:
    """Heartbeat / normal-traffic lines for related services. No anomalies —
    these exist as red herrings so the agent can't just grep for ERROR globally."""
    n = rng.randint(15, 25)
    return [
        rng.choice([
            f"[INFO] {service}: heartbeat ok",
            f"[INFO] {service}: GET /healthz 200",
            f"[INFO] {service}: cache hit ratio {rng.uniform(0.92, 0.99):.2f}",
            f"[INFO] {service}: handled request id={rng.randint(10000, 99999)}",
            f"[INFO] {service}: db query took {rng.randint(5, 30)}ms",
        ])
        for _ in range(n)
    ]


def _gen_metrics(root_cause: str, rng: random.Random) -> dict[str, list[int | float]]:
    """Time series whose shape reflects the failure mode."""
    n = rng.randint(40, 60)

    if root_cause == "oom_killed":
        limit = 256_000_000
        memory: list[int | float] = []
        for i in range(n):
            base = 50_000_000 + (limit - 50_000_000) * (i / max(1, n - 1))
            memory.append(min(limit, max(0, int(base + rng.randint(-2_000_000, 2_000_000)))))
        return {
            "memory_bytes": memory,
            "cpu_pct": [round(rng.uniform(20, 70), 1) for _ in range(n)],
            "restart_count": [(i * 4) // max(1, n) for i in range(n)],
        }

    if root_cause == "db_pool_exhausted":
        pool_size = 50
        pool: list[int | float] = []
        latency: list[int | float] = []
        for i in range(n):
            in_use = min(pool_size, 5 + int(i * 1.0) + rng.randint(-2, 2))
            pool.append(in_use)
            if in_use >= pool_size:
                latency.append(int(2000 + rng.randint(0, 3000)))
            else:
                latency.append(int(80 + rng.randint(-30, 30) + (in_use / pool_size) * 200))
        return {
            "db_pool_in_use": pool,
            "request_latency_p99_ms": latency,
            "request_count": [int(rng.uniform(800, 1200)) for _ in range(n)],
        }

    if root_cause == "bad_deploy":
        deploy_at = n // 2
        return {
            "errors_per_sec": [
                rng.randint(0, 2) if i < deploy_at else rng.randint(35, 55)
                for i in range(n)
            ],
            "request_count": [int(rng.uniform(800, 1200)) for _ in range(n)],
            "deploy_at_t": [deploy_at],
        }

    if root_cause == "missing_config":
        # Service can't get healthy — sparse, low-magnitude metrics.
        return {
            "memory_bytes": [12_000_000 + rng.randint(-500_000, 500_000) for _ in range(8)],
            "restart_count": list(range(rng.randint(5, 12))),
            "requests_per_sec": [0] * 12,
        }

    raise ValueError(f"unknown root_cause: {root_cause}")


def _gen_kubectl_describe(root_cause: str, service: str, rng: random.Random) -> str:
    pod_id = "".join(rng.choices("abcdefghij1234567890", k=10))
    pod = f"{service}-{pod_id}"
    age = rng.randint(5, 30)

    if root_cause == "oom_killed":
        return (
            f"Name:           {pod}\n"
            f"Namespace:      production\n"
            f"Status:         Running\n"
            f"Containers:\n"
            f"  app:\n"
            f"    State:           Running\n"
            f"    Last State:      Terminated\n"
            f"      Reason:        OOMKilled\n"
            f"      Exit Code:     137\n"
            f"      Started:       {age} minutes ago\n"
            f"      Finished:      {age - 1} minutes ago\n"
            f"    Restart Count:   {rng.randint(3, 7)}\n"
            f"    Limits:\n"
            f"      memory:        256Mi\n"
        )
    if root_cause == "db_pool_exhausted":
        return (
            f"Name:           {pod}\n"
            f"Namespace:      production\n"
            f"Status:         Running\n"
            f"Containers:\n"
            f"  app:\n"
            f"    State:           Running\n"
            f"    Restart Count:   0\n"
            f"    Readiness:       failure threshold exceeded\n"
            f"Events:\n"
            f"  Type     Reason       Age   Message\n"
            f"  Warning  Unhealthy    {age}m   Readiness probe failed: HTTP 503 (timeout)\n"
        )
    if root_cause == "bad_deploy":
        new_sha = "".join(rng.choices("0123456789abcdef", k=7))
        old_sha = "".join(rng.choices("0123456789abcdef", k=7))
        return (
            f"Name:           {pod}\n"
            f"Namespace:      production\n"
            f"Status:         Running\n"
            f"Containers:\n"
            f"  app:\n"
            f"    Image:           registry.example.com/{service}:{new_sha}\n"
            f"    State:           Running\n"
            f"    Restart Count:   0\n"
            f"Events:\n"
            f"  Type     Reason     Age    Message\n"
            f"  Normal   Pulled     {age}m   Pulled image registry.example.com/{service}:{new_sha} (was {old_sha})\n"
            f"  Normal   Created    {age}m   Created container app\n"
            f"  Normal   Started    {age}m   Started container app\n"
        )
    if root_cause == "missing_config":
        return (
            f"Name:           {pod}\n"
            f"Namespace:      production\n"
            f"Status:         CrashLoopBackOff\n"
            f"Containers:\n"
            f"  app:\n"
            f"    State:           Waiting\n"
            f"      Reason:        CrashLoopBackOff\n"
            f"    Last State:      Terminated\n"
            f"      Reason:        Error\n"
            f"      Exit Code:     1\n"
            f"    Restart Count:   {rng.randint(8, 20)}\n"
        )
    raise ValueError(f"unknown root_cause: {root_cause}")
