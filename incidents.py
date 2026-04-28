from dataclasses import dataclass

@dataclass
class Incident:
    incident_type: str
    affected_service: str
    alert_text: str
    logs: dict[str, list[str]] # key is the log source, value is the list of logs
    metrics: dict[str, list[int | float]]  # time series per metric name
    root_cause: str # the root cause of the incident e.g. "oomkilled"
    correct_action: str  # the correct action to take to resolve the incident

INCIDENTS = [
    # ---------- db_pool_exhausted ----------
    Incident(
        incident_type="high_latency",
        affected_service="payments-api",
        alert_text="payments-api p99 latency > 2000ms for 5min (SLO: 200ms)",
        logs={
            "payments-api": [
                "[INFO] payments-api: handling request id=48211",
                "[INFO] payments-api: handling request id=48212",
                "[WARN] payments-api: db pool 48/50, queueing requests",
                "[WARN] payments-api: db pool 50/50, queueing requests",
                "[ERROR] payments-api: timeout acquiring db connection after 5000ms",
                "[ERROR] payments-api: request id=48211 failed: ConnectionTimeoutError",
                "[ERROR] payments-api: request id=48212 failed: ConnectionTimeoutError",
            ],
        },
        metrics={
            "request_latency_p99_ms": [180, 220, 410, 890, 1840, 3200, 5000, 5000],
            "db_pool_in_use": [22, 28, 35, 44, 50, 50, 50, 50],
        },
        root_cause="db_pool_exhausted",
        correct_action="scale_db_pool",
    ),
    Incident(
        incident_type="high_latency",
        affected_service="checkout-api",
        alert_text="checkout-api p99 latency > 3000ms for 8min",
        logs={
            "checkout-api": [
                "[INFO] checkout-api: order submission id=ord_9921",
                "[INFO] checkout-api: order submission id=ord_9922",
                "[WARN] checkout-api: connection pool exhausted, waiting",
                "[WARN] checkout-api: 30 requests waiting on db connection",
                "[ERROR] checkout-api: HikariCP timeout: 30000ms exceeded",
                "[ERROR] checkout-api: order ord_9921 failed: db unavailable",
            ],
        },
        metrics={
            "request_latency_p99_ms": [220, 380, 750, 1500, 2800, 4200, 6000, 6000],
            "db_pool_in_use": [18, 25, 31, 40, 50, 50, 50, 50],
        },
        root_cause="db_pool_exhausted",
        correct_action="scale_db_pool",
    ),

    # ---------- oom_killed ----------
    Incident(
        incident_type="crashloop",
        affected_service="search-svc",
        alert_text="search-svc pod in CrashLoopBackOff (4 restarts in 6min)",
        logs={
            "search-svc": [
                "[INFO] search-svc: starting up build sha=4f2a18b",
                "[INFO] search-svc: loading config from /etc/config",
                "[INFO] search-svc: warming index cache (heap target 220MB)",
                "[INFO] search-svc: index cache 180MB / 220MB",
                "[INFO] search-svc: index cache 210MB / 220MB",
                # logs cut off mid-startup — process killed
            ],
        },
        metrics={
            "memory_bytes": [80_000_000, 130_000_000, 190_000_000, 240_000_000, 256_000_000],
            "restart_count": [0, 1, 2, 3, 4],
        },
        root_cause="oom_killed",
        correct_action="increase_memory_limit",
    ),
    Incident(
        incident_type="crashloop",
        affected_service="recommendation-svc",
        alert_text="recommendation-svc pod terminated unexpectedly (3 restarts)",
        logs={
            "recommendation-svc": [
                "[INFO] recommendation-svc: starting up",
                "[INFO] recommendation-svc: loading model weights from s3",
                "[INFO] recommendation-svc: model loaded, allocating embedding cache",
                "[INFO] recommendation-svc: embedding cache 380MB / 512MB",
                "[INFO] recommendation-svc: embedding cache 470MB / 512MB",
                "[INFO] recommendation-svc: serving traffic on :8080",
                # cut off — pod killed
            ],
        },
        metrics={
            "memory_bytes": [120_000_000, 280_000_000, 420_000_000, 500_000_000, 512_000_000],
            "restart_count": [0, 1, 2, 3, 3],
        },
        root_cause="oom_killed",
        correct_action="increase_memory_limit",
    ),

    # ---------- bad_deploy ----------
    Incident(
        incident_type="error_spike",
        affected_service="payments-api",
        alert_text="payments-api 5xx error rate exceeded 8% (was 0.1% baseline)",
        logs={
            "payments-api": [
                "[INFO] payments-api: serving build sha=7c91d44",
                "[INFO] payments-api: deploy completed at t=15",
                "[ERROR] payments-api: NullPointerException in PaymentHandler.process line 142",
                "[ERROR] payments-api: 500 returned for POST /api/v1/charge",
                "[ERROR] payments-api: NullPointerException in PaymentHandler.process line 142",
                "[ERROR] payments-api: 500 returned for POST /api/v1/charge",
                "[ERROR] payments-api: NullPointerException in PaymentHandler.process line 142",
            ],
        },
        metrics={
            "errors_per_sec": [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 47, 52, 48, 55, 49, 51],
            "deploy_at_t": [15],
        },
        root_cause="bad_deploy",
        correct_action="rollback_deploy",
    ),
    Incident(
        incident_type="error_spike",
        affected_service="auth-service",
        alert_text="auth-service 5xx error rate at 12% (baseline 0.05%)",
        logs={
            "auth-service": [
                "[INFO] auth-service: serving build sha=a83f001",
                "[INFO] auth-service: deploy completed, rolling restart finished at t=10",
                "[ERROR] auth-service: AssertionError: jwt signing key length mismatch",
                "[ERROR] auth-service: 500 returned for POST /api/v1/login",
                "[ERROR] auth-service: AssertionError: jwt signing key length mismatch",
                "[ERROR] auth-service: 500 returned for POST /api/v1/refresh",
                "[ERROR] auth-service: AssertionError: jwt signing key length mismatch",
            ],
        },
        metrics={
            "errors_per_sec": [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 38, 42, 45, 41, 39, 43, 40],
            "deploy_at_t": [10],
        },
        root_cause="bad_deploy",
        correct_action="rollback_deploy",
    ),

    # ---------- missing_config ----------
    Incident(
        incident_type="crashloop",
        affected_service="user-service",
        alert_text="user-service pod in CrashLoopBackOff (5 restarts in 4min)",
        logs={
            "user-service": [
                "[INFO] user-service: starting up",
                "[ERROR] user-service: required env var DATABASE_URL not set",
                "[FATAL] user-service: cannot start without DATABASE_URL, exiting",
            ],
        },
        metrics={
            "memory_bytes": [12_000_000, 12_000_000, 12_000_000, 12_000_000],
            "restart_count": [0, 1, 2, 3, 4, 5],
            "requests_per_sec": [0, 0, 0, 0, 0, 0],
        },
        root_cause="missing_config",
        correct_action="restore_config",
    ),
    Incident(
        incident_type="crashloop",
        affected_service="notification-svc",
        alert_text="notification-svc fails to start (6 restart attempts)",
        logs={
            "notification-svc": [
                "[INFO] notification-svc: starting up",
                "[INFO] notification-svc: reading config from /etc/notification/config.yaml",
                "[ERROR] notification-svc: required key SMTP_HOST not found in config",
                "[FATAL] notification-svc: missing required configuration, exiting code 1",
            ],
        },
        metrics={
            "memory_bytes": [15_000_000, 15_000_000, 15_000_000],
            "restart_count": [0, 2, 4, 6],
            "requests_per_sec": [0, 0, 0, 0],
        },
        root_cause="missing_config",
        correct_action="restore_config",
    ),

    # ---------- downstream_dep_slow ----------
    Incident(
        incident_type="high_latency",
        affected_service="checkout-api",
        alert_text="checkout-api p99 latency > 4000ms (SLO 250ms), upstream of inventory-svc",
        logs={
            "checkout-api": [
                "[INFO] checkout-api: handling request id=ord_3344",
                "[INFO] checkout-api: calling inventory-svc.check_stock",
                "[WARN] checkout-api: inventory-svc responded in 4823ms (p99 SLO 200ms)",
                "[INFO] checkout-api: response body=ok",
                "[INFO] checkout-api: handling request id=ord_3345",
                "[WARN] checkout-api: inventory-svc responded in 5102ms",
                "[WARN] checkout-api: inventory-svc responded in 4988ms",
            ],
        },
        metrics={
            "request_latency_p99_ms": [240, 380, 1100, 2400, 4200, 5100, 5000],
            "downstream_latency_p99_ms": [180, 320, 1040, 2350, 4180, 5080, 4970],
            "db_pool_in_use": [18, 19, 21, 22, 20, 22, 21],
        },
        root_cause="downstream_dep_slow",
        correct_action="page_owner_of_dep",
    ),
    Incident(
        incident_type="high_latency",
        affected_service="search-svc",
        alert_text="search-svc p99 latency > 3500ms, calls into ranker-svc",
        logs={
            "search-svc": [
                "[INFO] search-svc: query received q='blue running shoes'",
                "[INFO] search-svc: calling ranker-svc.score_results",
                "[WARN] search-svc: ranker-svc responded in 3820ms (expected <300ms)",
                "[INFO] search-svc: returning 24 results",
                "[INFO] search-svc: query received q='laptop sleeve 13 inch'",
                "[WARN] search-svc: ranker-svc responded in 4140ms",
                "[WARN] search-svc: ranker-svc responded in 3950ms",
            ],
        },
        metrics={
            "request_latency_p99_ms": [310, 540, 1280, 2400, 3700, 4150, 3950],
            "downstream_latency_p99_ms": [260, 490, 1230, 2360, 3680, 4140, 3940],
            "memory_bytes": [180_000_000, 182_000_000, 181_000_000, 183_000_000],
        },
        root_cause="downstream_dep_slow",
        correct_action="page_owner_of_dep",
    ),
]
