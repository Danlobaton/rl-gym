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
    # --- Token-level fields (schema 1.3+). None when the inference backend
    # can't supply them (e.g., Anthropic API). For RL training, all three
    # MUST be populated — see inference/vllm.py.
    prompt_tokens: list[int] | None = None     # token IDs the model saw (post chat-template)
    output_tokens: list[int] | None = None     # token IDs the model produced
    output_logprobs: list[float] | None = None # log P(token_t | prefix) under the policy

@dataclass
class Trace:
    schema_version: str # "1.3" — bump on incompatible change
    run_id: str # shared across all episodes in one rollout invocation
    run_started_at: float # UTC unix timestamp; shared across the run, drives partition path
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
    diagnostics: dict = field(default_factory=dict)  # rubric metadata: judge sampling, etc.
    # --- Trajectory-level token-fidelity metadata (schema 1.3+). Pinning these
    # lets a future trainer detect tokenizer/weight drift across runs — silent
    # drift is the most common source of mysteriously-broken RL training.
    policy_model: str = ""               # exact model identifier (e.g. "meta-llama/Llama-3.1-8B-Instruct")
    policy_model_revision: str = ""      # git/HF commit SHA — pins the weights
    tokenizer_hash: str = ""             # fingerprint of the tokenizer's vocab + special tokens
    sampling_params: dict = field(default_factory=dict)  # {temperature, top_p, max_tokens}


def write_trace(trace: Trace, root: Path | str = "traces") -> Path:
    """Write a trace as JSONL under hive-style `dt=YYYY-MM-DD/ts=HH-MM-SSZ/` partition.

    Filename: run_id={run_id}_seed={seed:06d}.jsonl. UTC partitions so
    rollouts in different timezones land in the same bucket; the nested
    `ts=` partition disambiguates multiple runs on the same day.
    """
    when = datetime.fromtimestamp(trace.run_started_at, tz=timezone.utc)
    partition = Path(root) / f"dt={when:%Y-%m-%d}" / f"ts={when:%H-%M-%S}Z"
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
