import json
from datetime import datetime, timezone
from pathlib import Path

from .trace import read_trace, read_trace_header


# --- replay (programmatic determinism check) ----------------------------


def replay_trace(trace: dict, env) -> list[tuple[int, str, object, object]]:
    """Replay a trace through the env and return the list of divergences."""
    env.reset(trace["seed"])
    diffs: list[tuple[int, str, object, object]] = []
    n = len(trace["steps"])
    for i, recorded in enumerate(trace["steps"]):
        result = env.step(recorded["action"])
        t = recorded["t"]
        if result.observation != recorded["observation"]:
            diffs.append((t, "observation", result.observation, recorded["observation"]))
        if result.reward != recorded["reward"]:
            diffs.append((t, "reward", result.reward, recorded["reward"]))
        if result.info != recorded.get("info", {}):
            diffs.append((t, "info", result.info, recorded["info"]))
        ended = result.terminated or result.truncated
        is_last = i == n - 1
        if ended != is_last:
            diffs.append((t, "termination",
                          f"terminated={result.terminated} truncated={result.truncated}",
                          f"trace_ends_here={is_last}"))
    return diffs


def replay_audit(env) -> None:
    """Replay every trace under traces/ and summarize. Exits non-zero on any failure."""
    paths = sorted(Path("traces").rglob("*.jsonl"))
    if not paths:
        raise SystemExit("audit: no traces found under traces/")

    failed: list[Path] = []
    for path in paths:
        trace = read_trace(path)
        diffs = replay_trace(trace, env)
        if not diffs:
            print(f"PASS  {path}")
        else:
            t, kind, _, _ = diffs[0]
            extra = f" (+{len(diffs) - 1} more)" if len(diffs) > 1 else ""
            print(f"FAIL  {path}  first MISMATCH: t={t} field={kind}{extra}")
            failed.append(path)

    print()
    print(f"audited {len(paths)} trace(s): {len(paths) - len(failed)} passed, {len(failed)} failed")
    if failed:
        print("  re-run `python main.py replay <path>` on a failing trace for full divergence detail")
        raise SystemExit(1)


def first_diff_index(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def excerpt(s: str, idx: int, ctx: int = 40) -> str:
    start, end = max(0, idx - ctx), min(len(s), idx + ctx)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(s) else ""
    return f"{prefix}{s[start:end]}{suffix}"


# --- pretty-print (human inspection) ------------------------------------


def print_trace(path: Path, full: bool) -> None:
    trace = read_trace(path)
    started = datetime.fromtimestamp(trace["started_at"], tz=timezone.utc).isoformat()
    duration = trace["ended_at"] - trace["started_at"]

    print(f"path: {path}")
    print(f"run_id={trace['run_id']}  seed={trace['seed']:06d}  agent={trace['agent_name']}")
    print(f"started_at={started}  duration={duration:.2f}s  steps={len(trace['steps'])}")
    print(f"task_meta={json.dumps(trace['task_meta'])}")
    print(f"ground_truth={json.dumps(trace['ground_truth'])}")
    print(f"total_reward={trace['total_reward']:+.3f}  breakdown={json.dumps(trace['reward_breakdown'])}")
    print()

    for step in trace["steps"]:
        print(f"  t={step['t']:>2}  reward={step['reward']:+.3f}  latency={step['latency_ms']}ms")
        print(f"    action: {json.dumps(step['action'])}")
        obs = step["observation"]
        if full:
            print("    observation:")
            for line in (obs.splitlines() or [""]):
                print(f"      {line}")
        else:
            preview = obs.replace("\n", " | ")
            if len(preview) > 100:
                preview = preview[:100] + f"... (+{len(obs) - 100} chars, use --full)"
            print(f"    observation: {preview}")
        if step.get("info"):
            print(f"    info: {json.dumps(step['info'])}")


def print_summary(path: Path) -> None:
    h = read_trace_header(path)
    task = h.get("task_meta", {}).get("incident_type", "?")
    print(
        f"{path}  seed={h['seed']:06d}  reward={h['total_reward']:+6.2f}  "
        f"task={task}  agent={h['agent_name']}  "
        f"ground_truth={json.dumps(h['ground_truth'])}"
    )
