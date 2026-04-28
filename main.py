import argparse
import glob
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from episode_trace import read_trace, read_trace_header, write_trace


def cmd_rollout(args: argparse.Namespace) -> None:
    from agent import run_episode  # lazy: avoids loading anthropic for read-only commands
    from gym import IncidentEnv

    env = IncidentEnv()
    run_id = uuid.uuid4().hex[:12]
    for seed in range(args.episodes):
        trace = run_episode(env, seed, run_id=run_id)
        path = write_trace(trace)
        print(
            f"seed={seed} reward={trace.total_reward:+.2f} "
            f"steps={len(trace.steps)} run_id={trace.run_id} -> {path}"
        )


def cmd_replay(args: argparse.Namespace) -> None:
    from gym import IncidentEnv  # lazy: keeps `show` startup free of env deps

    if args.path == "audit":
        _replay_audit(IncidentEnv())
        return

    path = Path(args.path)
    trace = read_trace(path)
    diffs = _replay_one(trace, IncidentEnv())

    if not diffs:
        print(f"OK: env replay matches trace  ({len(trace['steps'])} steps, seed={trace['seed']}, run_id={trace['run_id']})")
        return

    print(f"NONDETERMINISTIC: {len(diffs)} divergence(s) in {len(trace['steps'])} steps")
    print(f"  path: {path}")
    print(f"  seed: {trace['seed']}  run_id: {trace['run_id']}")
    for t, kind, got, want in diffs[:5]:
        print()
        print(f"  MISMATCH  t={t}  field={kind}")
        if kind == "observation" and isinstance(got, str) and isinstance(want, str):
            idx = _first_diff_index(got, want)
            print(f"    first diff at char {idx} (lengths: trace={len(want)}, env={len(got)})")
            print(f"    trace: {_excerpt(want, idx)!r}")
            print(f"    env:   {_excerpt(got, idx)!r}")
        else:
            print(f"    trace: {want!r}")
            print(f"    env:   {got!r}")
    if len(diffs) > 5:
        print(f"\n  ... and {len(diffs) - 5} more divergence(s)")
    raise SystemExit(1)


def _replay_one(trace: dict, env) -> list[tuple[int, str, object, object]]:
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


def _replay_audit(env) -> None:
    """Replay every trace under traces/ and summarize. Exits non-zero on any failure."""
    paths = sorted(Path("traces").rglob("*.jsonl"))
    if not paths:
        raise SystemExit("audit: no traces found under traces/")

    failed: list[Path] = []
    for path in paths:
        trace = read_trace(path)
        diffs = _replay_one(trace, env)
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


def _first_diff_index(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _excerpt(s: str, idx: int, ctx: int = 40) -> str:
    start, end = max(0, idx - ctx), min(len(s), idx + ctx)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(s) else ""
    return f"{prefix}{s[start:end]}{suffix}"


def cmd_show(args: argparse.Namespace) -> None:
    if args.filter and args.path:
        raise SystemExit("show: pass either a path or --filter, not both")
    if args.filter:
        paths = sorted(Path(p) for p in glob.glob(args.filter, recursive=True))
        if not paths:
            raise SystemExit(f"show: no traces match {args.filter!r}")
        for path in paths:
            _print_summary(path)
        return
    if not args.path:
        raise SystemExit("show: provide a PATH or --filter GLOB")
    _print_trace(Path(args.path), full=args.full)


def _print_trace(path: Path, full: bool) -> None:
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


def _print_summary(path: Path) -> None:
    h = read_trace_header(path)
    task = h.get("task_meta", {}).get("incident_type", "?")
    print(
        f"{path}  seed={h['seed']:06d}  reward={h['total_reward']:+6.2f}  "
        f"task={task}  agent={h['agent_name']}  "
        f"ground_truth={json.dumps(h['ground_truth'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="rl_gym")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rollout = sub.add_parser("rollout", help="run episodes and write traces")
    p_rollout.add_argument("--episodes", type=int, default=5)
    p_rollout.set_defaults(func=cmd_rollout)

    p_replay = sub.add_parser("replay", help="re-run env from trace seed and verify byte-identical output")
    p_replay.add_argument("path", help="trace file to replay, or 'audit' to replay every trace under traces/")
    p_replay.set_defaults(func=cmd_replay)

    p_show = sub.add_parser("show", help="pretty-print a trace or summarize many")
    p_show.add_argument("path", nargs="?", help="trace file to print")
    p_show.add_argument("--full", action="store_true", help="print complete observations (no 100-char preview)")
    p_show.add_argument("--filter", metavar="GLOB", help="glob pattern; print summary lines for each match")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
