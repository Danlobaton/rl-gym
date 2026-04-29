"""Tests for the Rubric — episode scoring as a separate concern from the env.

Parity tests verify the rubric's totals match the formula the env used to
implement: total = -0.02 * non_resolve_steps + 0.5 if resolved + (1.5/-1.0) * rc + (1.0/-0.5) * action.

Judge tests verify the sampling decision is deterministic on run_id and the
diagnostics dict correctly records whether the judge ran.
"""
import asyncio

from sregym.rubric import (
    DEFAULT_WEIGHTS,
    Rubric,
    StepPenalty,
    Completion,
    RootCauseMatch,
    ActionMatch,
    default_rubric,
    _parse_judge_response,
)


def _step(tool: str, **args) -> dict:
    return {"action": {"tool": tool, "args": args}}


def _resolve(rc: str, act: str, rationale: str = "") -> dict:
    args = {"root_cause": rc, "action": act}
    if rationale:
        args["rationale"] = rationale
    return _step("resolve", **args)


def _readonly(tool: str = "tail_logs") -> dict:
    return _step(tool, service="x")


def _score(rubric, steps, gt, **kw):
    """Helper: tests are sync but Rubric.score is async."""
    return asyncio.run(rubric.score(steps, gt, **kw))


# --- formula parity: every combination matches the old env-baked total ---


def test_perfect_resolve_no_investigation():
    """1 resolve step, both correct. Old formula: 0.5 + 1.5 + 1.0 = 3.0"""
    steps = [_resolve("oom_killed", "increase_memory_limit")]
    gt = {"root_cause": "oom_killed", "correct_action": "increase_memory_limit"}
    rubric = default_rubric(judge_sample_rate=0)
    total, breakdown, diags = _score(rubric, steps, gt)
    assert total == 3.0
    assert breakdown == {"step_penalty": 0.0, "completion": 0.5, "root_cause": 1.5, "action": 1.0}
    assert diags["judge_ran"] is False
    assert diags["judge_sampled"] is False


def test_step_penalty_accumulates():
    """K read-only steps + correct resolve: total = 3.0 - 0.02*K, for K = 0..6"""
    gt = {"root_cause": "X", "correct_action": "Y"}
    rubric = default_rubric(judge_sample_rate=0)
    for k in range(7):
        steps = [_readonly() for _ in range(k)] + [_resolve("X", "Y")]
        total, breakdown, _ = _score(rubric, steps, gt)
        expected = 3.0 - 0.02 * k
        assert abs(total - expected) < 1e-9, f"k={k}: got {total}, expected {expected}"
        assert abs(breakdown["step_penalty"] - (-0.02 * k)) < 1e-9


def test_wrong_root_cause_only():
    """Resolve with wrong rc but right action. Old formula: 0.5 - 1.0 + 1.0 = 0.5"""
    steps = [_resolve("wrong", "Y")]
    gt = {"root_cause": "X", "correct_action": "Y"}
    total, breakdown, _ = _score(default_rubric(judge_sample_rate=0), steps, gt)
    assert total == 0.5
    assert breakdown["root_cause"] == -1.0
    assert breakdown["action"] == 1.0


def test_wrong_action_only():
    """Right rc, wrong action. Old formula: 0.5 + 1.5 - 0.5 = 1.5"""
    steps = [_resolve("X", "wrong")]
    gt = {"root_cause": "X", "correct_action": "Y"}
    total, _, _ = _score(default_rubric(judge_sample_rate=0), steps, gt)
    assert total == 1.5


def test_both_wrong():
    """Both wrong. Old formula: 0.5 - 1.0 - 0.5 = -1.0"""
    steps = [_resolve("nope", "nope")]
    gt = {"root_cause": "X", "correct_action": "Y"}
    total, _, _ = _score(default_rubric(judge_sample_rate=0), steps, gt)
    assert total == -1.0


def test_truncated_no_resolve():
    """Episode truncated without ever resolving — only step_penalty fires."""
    steps = [_readonly() for _ in range(3)]
    total, breakdown, _ = _score(default_rubric(judge_sample_rate=0), steps, {})
    assert abs(total - (-0.06)) < 1e-9
    assert breakdown["completion"] == 0.0
    assert breakdown["root_cause"] == 0.0
    assert breakdown["action"] == 0.0


def test_breakdown_sums_to_total():
    """Total is exactly the sum of breakdown values when judge isn't sampled."""
    steps = [_readonly(), _readonly(), _resolve("X", "Y")]
    gt = {"root_cause": "X", "correct_action": "Y"}
    total, breakdown, _ = _score(default_rubric(judge_sample_rate=0), steps, gt)
    assert abs(total - sum(breakdown.values())) < 1e-9


# --- judge sampling: deterministic, breakdown gets no fake zeros --------


def test_judge_sample_rate_zero_never_judges():
    rubric = Rubric(terms=[], judge_sample_rate=0)
    assert not rubric._should_judge("any_run_id")


def test_judge_empty_run_id_never_judges():
    """Anonymous run_id (empty) opts out of sampling regardless of rate."""
    rubric = Rubric(terms=[], judge_sample_rate=1.0)
    assert not rubric._should_judge("")


def test_judge_sampling_is_deterministic_on_run_id():
    """Same run_id + same rate → same decision, every time, every process."""
    rubric = Rubric(terms=[], judge_sample_rate=0.5)
    decisions = {rubric._should_judge("abc-123") for _ in range(20)}
    assert len(decisions) == 1  # always the same answer


def test_judge_sampling_distinguishes_run_ids():
    """Different run_ids produce different sampling decisions in expectation."""
    rubric = Rubric(terms=[], judge_sample_rate=0.5)
    decisions = {rubric._should_judge(f"run-{i}") for i in range(50)}
    # With 50 distinct ids and rate 0.5, both outcomes should appear.
    assert decisions == {True, False}


def test_unsampled_breakdown_omits_rationale_quality():
    """When the trace isn't sampled, breakdown has no rationale_quality key
    (don't fake a zero — would conflate 'not judged' with 'judged poorly')."""
    steps = [_resolve("X", "Y", rationale="anything")]
    gt = {"root_cause": "X", "correct_action": "Y"}
    rubric = default_rubric(judge_sample_rate=0)  # forced off
    _, breakdown, diags = _score(rubric, steps, gt, run_id="some-run")
    assert "rationale_quality" not in breakdown
    assert diags["judge_ran"] is False


def test_diagnostics_records_sampling_decision():
    """Diagnostics tells the analyst whether the judge sampled, ran, and why if not."""
    rubric = default_rubric(judge_sample_rate=0)
    _, _, diags = _score(rubric, [_readonly()], {}, run_id="any")
    assert "judge_sampled" in diags
    assert "judge_ran" in diags


# --- weights / extensibility -------------------------------------------


def test_default_weights_includes_rationale_quality():
    """The judge weight is exposed in the module-level constant (1.0 by default)."""
    assert DEFAULT_WEIGHTS["rationale_quality"] == 1.0


def test_judge_weight_defaults_from_default_weights():
    """A fresh Rubric pulls its judge_weight from DEFAULT_WEIGHTS unless overridden."""
    r = Rubric(terms=[])
    assert r.judge_weight == DEFAULT_WEIGHTS["rationale_quality"]


def test_judge_weight_can_be_overridden():
    r = Rubric(terms=[], judge_weight=2.5)
    assert r.judge_weight == 2.5


def test_custom_rubric_can_omit_terms():
    """A rubric with only StepPenalty produces just that one term."""
    steps = [_readonly(), _resolve("X", "Y")]
    rubric = Rubric(terms=[StepPenalty()], judge_sample_rate=0)
    total, breakdown, _ = _score(rubric, steps, {"root_cause": "X", "correct_action": "Y"})
    assert breakdown == {"step_penalty": -0.02}
    assert total == -0.02


def test_custom_rubric_can_reweight():
    steps = [_resolve("wrong", "Y")]
    gt = {"root_cause": "X", "correct_action": "Y"}
    harsh = Rubric(
        terms=[Completion(), RootCauseMatch(correct=1.5, wrong=-5.0), ActionMatch()],
        judge_sample_rate=0,
    )
    total, breakdown, _ = _score(harsh, steps, gt)
    assert breakdown["root_cause"] == -5.0
    assert total == 0.5 - 5.0 + 1.0  # -3.5


def test_custom_rubric_can_add_a_new_term():
    """Verifies extensibility: novel terms compose into Rubric without changes."""
    from dataclasses import dataclass

    @dataclass
    class FixedBonus:
        name: str = "bonus"
        amount: float = 5.0
        def score(self, steps, ground_truth):
            return self.amount

    rubric = Rubric(terms=[StepPenalty(), FixedBonus()], judge_sample_rate=0)
    steps = [_readonly()]
    total, breakdown, _ = _score(rubric, steps, {})
    assert breakdown == {"step_penalty": -0.02, "bonus": 5.0}
    assert abs(total - 4.98) < 1e-9


# --- judge response parser ----------------------------------------------


def test_parse_judge_response_well_formed():
    text = "SCORE: 0.85\nREASONING: cites the OOMKilled metric peak at t=37."
    score, reasoning = _parse_judge_response(text)
    assert score == 0.85
    assert "OOMKilled" in reasoning


def test_parse_judge_response_clamps_out_of_range():
    score, _ = _parse_judge_response("SCORE: 1.5\nREASONING: x")
    assert score == 1.0
    score, _ = _parse_judge_response("SCORE: -0.3\nREASONING: x")
    assert score == 0.0


def test_parse_judge_response_handles_extra_whitespace():
    text = "  SCORE:    0.7   \n  REASONING:   spaces are fine  "
    score, reasoning = _parse_judge_response(text)
    assert score == 0.7
    assert reasoning == "spaces are fine"


def test_parse_judge_response_returns_none_when_score_missing():
    score, reasoning = _parse_judge_response("REASONING: no score here")
    assert score is None
    assert reasoning == "no score here"


def test_parse_judge_response_handles_unparseable_score():
    score, _ = _parse_judge_response("SCORE: not_a_number\nREASONING: x")
    assert score is None
