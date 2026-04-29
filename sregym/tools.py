import shlex
import subprocess

from .generator import Incident

TOOL_TIMEOUT_S = 5


def _sh(cmd: str) -> str:
    """Run a shell command, return stdout with trailing newline trimmed.

    shell=True is intentional — pipes (`tail | grep`, `tail | awk`) are part of
    the tool surface. The trade-off: every interpolated arg must pass through
    shlex.quote so adversarial agent input can't break out of its slot.
    """
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=TOOL_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return "ERROR: tool timed out (>5s)"
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
