import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from incident_generator import generate_incident
from incidents import Incident

WORKDIR_ROOT = Path("/tmp/sregym")
TOOL_TIMEOUT_S = 5


@dataclass
class StepResult:
    observation: str # What the agent sees next
    reward: float # How well did you do in this turn?
    terminated: bool # episode is done and ended naturally
    truncated: bool # episode hit the step limit
    info: dict # contains debugging, ground truth, etc.


def _sh(cmd: str) -> str:
    """Run a shell command, return stdout with trailing newline trimmed.

    shell=True is intentional — pipes (`tail | grep`, `tail | awk`) are part of
    the tool surface. The trade-off: every interpolated arg must pass through
    shlex.quote so adversarial agent input can't break out of its slot.
    """
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=TOOL_TIMEOUT_S
    )
    return result.stdout.rstrip("\n")


def tail_logs(incident: Incident, service: str, lines: int = 50, grep: str | None = None) -> str:
    """tail -n {lines} logs/{service}.log [| grep -- {pattern}]"""
    log_path = incident.workdir / "logs" / f"{service}.log"
    if not log_path.exists():
        return f"ERROR: no log file for {service}"
    try:
        n = int(lines)  # int cast neutralizes any shell metachars in `lines`
    except (TypeError, ValueError):
        return f"ERROR: invalid lines argument: {lines!r}"

    cmd = f"tail -n {n} {shlex.quote(str(log_path))}"
    if grep:
        # `--` so a pattern starting with `-` isn't parsed as a flag
        cmd += f" | grep -- {shlex.quote(str(grep))}"
    return _sh(cmd)


def query_metrics(incident: Incident, service: str, metric: str) -> str:
    """tail -n +2 metrics/{metric}.csv | awk -F, '{print "t="$1" value="$2}'

    The awk pipe reformats CSV rows back into the legacy `t=N value=V` shape so
    replay traces written before this rewrite still match byte-for-byte.
    """
    csv_path = incident.workdir / "metrics" / f"{metric}.csv"
    if not csv_path.exists():
        return f"ERROR: metric {metric} not found"
    cmd = (
        f"tail -n +2 {shlex.quote(str(csv_path))} "
        f"| awk -F, '{{print \"t=\"$1\" value=\"$2}}'"
    )
    return _sh(cmd)


def resolve(incident: Incident, root_cause: str, action: str) -> tuple[float, dict]:
    """Resolve the incident by calling the correct action to resolve the incident."""
    breakdown = {"completion": 0.5, "root_cause": 0.0, "action": 0.0}
    breakdown["root_cause"] = 1.5 if root_cause == incident.root_cause else -1.0
    breakdown["action"] = 1.0 if action == incident.correct_action else -0.5
    return sum(breakdown.values()), breakdown


class IncidentEnv:
    MAX_STEPS = 10

    def reset(self, seed: int):
        # generate_incident wipes any prior workdir at WORKDIR_ROOT/ep-<seed>/
        # before regenerating — idempotent, so a crashed episode doesn't leak
        # state into the next reset.
        self.incident = generate_incident(seed, WORKDIR_ROOT)
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
