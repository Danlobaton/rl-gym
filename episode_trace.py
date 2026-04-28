from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import json

@dataclass
class TraceStep:
    t: int # step index
    action: dict
    observation: str
    reward: float
    info: dict = field(default_factory=dict)
    latency_ms: int = 0

@dataclass
class Trace:
    schema_version: str # "1.0" — bump on incompatible change
    run_id: str # shared across all episodes in one rollout invocation
    agent_name: str
    agent_config: dict
    seed: int
    task_meta: dict
    steps: list[TraceStep]
    total_reward: float
    reward_breakdown: dict
    ground_truth: dict
    started_at: float
    ended_at: float


def write_trace(trace: Trace, root: Path | str = "traces") -> Path:
    """Write a trace as JSONL under hive-style `dt=YYYY-MM-DD/` partition.

    Filename: run_id={run_id}_seed={seed:06d}.jsonl. UTC partitions so
    rollouts in different timezones land in the same bucket.
    """
    dt = datetime.fromtimestamp(trace.started_at, tz=timezone.utc).strftime("%Y-%m-%d")
    partition = Path(root) / f"dt={dt}"
    partition.mkdir(parents=True, exist_ok=True)
    path = partition / f"run_id={trace.run_id}_seed={trace.seed:06d}.jsonl"

    record = asdict(trace)
    steps = record.pop("steps")
    with path.open("w") as f:
        f.write(json.dumps(record) + "\n")
        for step in steps:
            f.write(json.dumps(step) + "\n")
    return path


def read_trace(path: Path | str) -> dict:
    """Read a trace JSONL file. First line is metadata, rest are steps."""
    with Path(path).open() as f:
        lines = [json.loads(line) for line in f if line.strip()]
    header, *steps = lines
    return {**header, "steps": steps}


def read_trace_header(path: Path | str) -> dict:
    """Read only the metadata header line — cheap for summary scans across many traces."""
    with Path(path).open() as f:
        return json.loads(f.readline())
