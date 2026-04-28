from dataclasses import dataclass
import random

@dataclass
class StepResult:
    observation: str # What the agent sees next
    reward: float # How well did you do in this turn?
    terminated: bool # episode is done and ended naturally
    truncated: bool # episode hit the step limit
    info: dict # contains debugging, ground truth, etc.

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
    Incident(
        incident_type="high_latency",
        affected_service="payments-api",
        alert_text="payments-api p99 latency > 2000ms for 5min",
        logs={
            "payments-api": [
                "[INFO] handling request id=12345",
                "[WARN] db pool 50/50, queueing",
                "[ERROR] timeout acquiring db connection",
                "[ERROR] request id=12345 failed: ConnectionTimeoutError",
            ]
        },
        metrics={
            "request_latency_p99_ms": [200, 350, 500, 1200, 5000],
        },
        root_cause="db_pool_exhausted",
        correct_action="scale_db_pool",
    ),
]

# Tools
def tail_logs(incident: Incident, service: str, lines: int = 50) -> str:
    if service not in incident.logs:
        return f"ERROR: no log file for {service}"
    return "\n".join(incident.logs[service][-lines:])

def query_metrics(incident: Incident, service: str, metric: str) -> str:
    if metric not in incident.metrics:
        return f"ERROR: metric {metric} not found"
    values = incident.metrics[metric]
    return "\n".join(f"t={i} value={v}" for i, v in enumerate(values))


def resolve(incident: Incident, root_cause: str, action: str) -> tuple[float, dict]:
    """Resolve the incident by calling the correct action to resolve the incident."""
    breakdown = {"completion": 0.5, "root_cause": 0.0, "action": 0.0}
    if root_cause == incident.root_cause:
        breakdown["root_cause"] = 1.5
    else:
        breakdown["root_cause"] = -1.0
    if action == incident.correct_action:
        breakdown["action"] = 1.0
    else:
        breakdown["action"] = -0.5
    return sum(breakdown.values()), breakdown

class IncidentEnv:
    MAX_STEPS = 10

    def reset(self, seed: int):
        rng = random.Random(seed)
        self.incident = rng.choice(INCIDENTS)
        self.step_count = 0
        self.done = False # This is internal environment state, not part of the API
        obs = self._render_alert()
        return obs, {"seed": seed}

    def step(self, action: dict):
        self.step_count += 1
        tool = action.get("tool")
        args = action.get("args", {})

        if tool == "resolve":
            reward, breakdown = resolve(self.incident, **args)
            self.done = True
            return StepResult(
                observation="(episode complete)",
                reward=reward,
                terminated=True,
                truncated=False,
                info={"breakdown": breakdown, "ground_truth": {
                    "root_cause": self.incident.root_cause,
                    "correct_action": self.incident.correct_action,
                }},
            )

        # read-only tool
        if tool == "tail_logs":
            obs = tail_logs(self.incident, **args)
        elif tool == "query_metrics":
            obs = query_metrics(self.incident, **args)
        else:
            obs = f"ERROR: unknown tool {tool}"

        truncated = self.step_count >= self.MAX_STEPS
        return StepResult(
            observation=obs,
            reward=-0.02,  # small per-step cost, see below
            terminated=False,
            truncated=truncated,
            info={},
        )

    def _render_alert(self) -> str:
        return f"""ALERT: {self.incident.alert_text}
        Affected: {self.incident.affected_service}

        Tools available:
        tail_logs(service, lines=50)
        query_metrics(service, metric)
        resolve(root_cause, action)
        """
