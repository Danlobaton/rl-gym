import argparse
import glob
import time
import uuid
from pathlib import Path


def cmd_rollout(args: argparse.Namespace) -> None:
    import asyncio
    from contextlib import nullcontext

    from .agent import run_episode  # lazy: avoids loading anthropic for read-only commands
    from .rubric import default_rubric
    from .trace import write_trace

    env_url = args.env_url
    if env_url:
        from .client import EnvClient
        env_cm = EnvClient(env_url)
    else:
        from .env import IncidentEnv
        env_cm = nullcontext(IncidentEnv())

    # One rubric across the whole rollout — caches the AsyncAnthropic client so
    # we don't spin up a fresh httpx connection pool per episode.
    rubric = default_rubric()
    run_id = uuid.uuid4().hex[:12]
    run_started_at = time.time()
    with env_cm as env:
        for seed in range(args.episodes):
            trace = asyncio.run(run_episode(
                env,
                seed,
                run_id=run_id,
                run_started_at=run_started_at,
                env_url=env_url,
                rubric=rubric,
            ))
            path = write_trace(trace)
            judged = trace.diagnostics.get("judge_ran", False)
            judge_note = f"  judge={trace.diagnostics.get('judge_raw_score'):.2f}" if judged else ""
            print(
                f"seed={seed} reward={trace.total_reward:+.2f} "
                f"steps={len(trace.steps)} run_id={trace.run_id}{judge_note} -> {path}"
            )


def cmd_replay(args: argparse.Namespace) -> None:
    from .env import IncidentEnv  # lazy: keeps `show` startup free of env deps
    from .replay import replay_trace, replay_audit, first_diff_index, excerpt
    from .trace import read_trace

    if args.path == "audit":
        replay_audit(IncidentEnv())
        return

    path = Path(args.path)
    trace = read_trace(path)
    diffs = replay_trace(trace, IncidentEnv())

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
            idx = first_diff_index(got, want)
            print(f"    first diff at char {idx} (lengths: trace={len(want)}, env={len(got)})")
            print(f"    trace: {excerpt(want, idx)!r}")
            print(f"    env:   {excerpt(got, idx)!r}")
        else:
            print(f"    trace: {want!r}")
            print(f"    env:   {got!r}")
    if len(diffs) > 5:
        print(f"\n  ... and {len(diffs) - 5} more divergence(s)")
    raise SystemExit(1)


def cmd_show(args: argparse.Namespace) -> None:
    from .replay import print_trace, print_summary

    if args.filter and args.path:
        raise SystemExit("show: pass either a path or --filter, not both")
    if args.filter:
        paths = sorted(Path(p) for p in glob.glob(args.filter, recursive=True))
        if not paths:
            raise SystemExit(f"show: no traces match {args.filter!r}")
        for path in paths:
            print_summary(path)
        return
    if not args.path:
        raise SystemExit("show: provide a PATH or --filter GLOB")
    print_trace(Path(args.path), full=args.full)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rl_gym")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rollout = sub.add_parser("rollout", help="run episodes and write traces")
    p_rollout.add_argument("--episodes", type=int, default=5)
    p_rollout.add_argument(
        "--env-url",
        metavar="URL",
        help="HTTP env server URL (e.g. http://localhost:8000); omit for in-process env",
    )
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
