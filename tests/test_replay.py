"""Tests for replay — the determinism enforcer.

Replay catches the bug class that breaks every downstream system silently
(eval harnesses, trainers, reward analyses). These tests lock that contract
in: any change that makes the env nondeterministic should fail one of these.
"""
import tempfile
import time
from dataclasses import asdict

from episode_trace import Trace, TraceStep, read_trace, write_trace
from gym import IncidentEnv
from main import _excerpt, _first_diff_index, _replay_one


def make_trace(seed: int = 7) -> Trace:
    """Build a trace by stepping the env, so recorded values match what env produces."""
    env = IncidentEnv()
    env.reset(seed)
    actions = [
        {"tool": "tail_logs", "args": {"service": "web"}},
        {
            "tool": "resolve",
            "args": {
                "root_cause": env.incident.root_cause,
                "action": env.incident.correct_action,
            },
        },
    ]
    steps: list[TraceStep] = []
    for t, action in enumerate(actions):
        r = env.step(action)
        steps.append(TraceStep(
            t=t, action=action, observation=r.observation,
            reward=r.reward, info=r.info, latency_ms=0,
        ))
        if r.terminated or r.truncated:
            break
    return Trace(
        schema_version="1.0", run_id="test", agent_name="dummy", agent_config={},
        seed=seed, task_meta={"incident_type": env.incident.incident_type},
        steps=steps, total_reward=sum(s.reward for s in steps),
        reward_breakdown=steps[-1].info.get("breakdown", {}),
        ground_truth=steps[-1].info.get("ground_truth", {}),
        started_at=time.time(), ended_at=time.time(),
    )


# --- core contract: a clean trace replays byte-identically ----------------


def test_clean_trace_replays_with_zero_diffs():
    trace = asdict(make_trace())
    assert _replay_one(trace, IncidentEnv()) == []


def test_jsonl_roundtrip_preserves_replay_correctness():
    """Writing to disk and reading back must not perturb determinism."""
    with tempfile.TemporaryDirectory() as td:
        path = write_trace(make_trace(), td)
        loaded = read_trace(path)
    assert _replay_one(loaded, IncidentEnv()) == []


def test_empty_step_list_replays_cleanly():
    trace = asdict(make_trace())
    trace["steps"] = []
    assert _replay_one(trace, IncidentEnv()) == []


# --- divergence detection: each field type ---------------------------------


def test_observation_mismatch_is_detected():
    trace = asdict(make_trace())
    trace["steps"][0]["observation"] += "TAMPERED"
    diffs = _replay_one(trace, IncidentEnv())
    assert len(diffs) == 1
    _, kind, env_val, trace_val = diffs[0]
    assert kind == "observation"
    assert "TAMPERED" in trace_val
    assert "TAMPERED" not in env_val


def test_reward_mismatch_is_detected():
    trace = asdict(make_trace())
    trace["steps"][0]["reward"] = 999.0
    diffs = _replay_one(trace, IncidentEnv())
    assert len(diffs) == 1
    _, kind, env_val, trace_val = diffs[0]
    assert kind == "reward"
    assert trace_val == 999.0
    assert env_val != 999.0


def test_info_mismatch_is_detected():
    trace = asdict(make_trace())
    trace["steps"][-1]["info"] = {"breakdown": {"completion": -1}}
    info_diffs = [d for d in _replay_one(trace, IncidentEnv()) if d[1] == "info"]
    assert len(info_diffs) == 1


def test_premature_termination_is_detected():
    """If the trace ends mid-episode, env will report not-terminated — a divergence."""
    trace = asdict(make_trace())
    trace["steps"] = trace["steps"][:-1]  # drop the resolve step
    termination_diffs = [d for d in _replay_one(trace, IncidentEnv()) if d[1] == "termination"]
    assert len(termination_diffs) == 1


# --- the high-value case: state derivation chain --------------------------


def test_seed_mutation_cascades_into_divergences():
    """Changing the seed makes env load a different incident — the canonical
    'silent corruption' case replay exists to catch."""
    trace = asdict(make_trace(seed=7))
    trace["seed"] = 99
    assert len(_replay_one(trace, IncidentEnv())) >= 1


# --- helpers --------------------------------------------------------------


def test_first_diff_index():
    assert _first_diff_index("abc", "abc") == 3      # identical
    assert _first_diff_index("xbc", "abc") == 0      # diff at start
    assert _first_diff_index("abd", "abc") == 2      # diff at end of overlap
    assert _first_diff_index("abc", "abcdef") == 3   # one is a prefix of the other
    assert _first_diff_index("", "") == 0            # both empty
    assert _first_diff_index("", "abc") == 0         # one empty


def test_excerpt_at_middle_has_ellipses_on_both_sides():
    s = "x" * 200
    out = _excerpt(s, 100, ctx=20)
    assert out.startswith("...")
    assert out.endswith("...")
    assert "x" * 40 in out


def test_excerpt_at_string_start_has_no_leading_ellipsis():
    out = _excerpt("abcdefghij", 0, ctx=3)
    assert not out.startswith("...")
    assert out.endswith("...")


def test_excerpt_at_string_end_has_no_trailing_ellipsis():
    out = _excerpt("abcdefghij", 10, ctx=3)
    assert out.startswith("...")
    assert not out.endswith("...")
