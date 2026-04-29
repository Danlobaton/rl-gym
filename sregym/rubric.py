"""Episode scoring as a separate concern from environment dynamics.

The env owns state and transitions; the rubric owns *what a completed
episode is worth*. Separating these lets reward shaping change (different
weights, new terms, curriculum schedules) without touching the env.

A rubric is a list of ScoringTerms; each term reads the completed episode
(steps + ground truth) and emits one float. The rubric returns their sum
plus a per-term breakdown for downstream analysis.

WHY DID I DO THIS! Great question young padawan!

- Now I can A/B test different reward shaping strategies
- I can score one trace with multiple rubrics and see what changes
- I can also test the rubric itself in isolation from the env

It was a win/win!
"""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


# Module-level weights for terms that aren't simple sync ScoringTerms.
# Existing terms keep their hard-coded defaults (per the "don't change existing
# terms" constraint); new terms read from here so reward shaping can be tweaked
# in one place.
DEFAULT_WEIGHTS: dict[str, float] = {
    "rationale_quality": 1.0,
}


# Module-level prompt so it can be iterated on without touching scoring logic.
JUDGE_PROMPT = """You are evaluating an SRE agent's diagnostic rationale for an incident.

The actual root cause was: {root_cause}

Available evidence the agent could have read (logs and one metric from the affected service):

LOGS for {affected_service}:
{log_content}

METRIC ({metric_name}):
{metric_content}

The agent's rationale for diagnosing the incident was:
{rationale}

Score the rationale on whether it cites SPECIFIC evidence from the logs or metrics that supports the declared root cause. A high score (close to 1.0) requires concrete references to log lines or metric values; a low score (close to 0.0) for vague, unsupported, or irrelevant rationales.

Output EXACTLY two lines, in this format:
SCORE: <float between 0.0 and 1.0>
REASONING: <one short sentence>
"""


class ScoringTerm(Protocol):
    name: str
    def score(self, steps: list[dict], ground_truth: dict) -> float: ...


def _resolve_step(steps: list[dict]) -> dict | None:
    """The first resolve action in the episode, or None if the agent never resolved."""
    for s in steps:
        if s.get("action", {}).get("tool") == "resolve":
            return s
    return None


@dataclass
class StepPenalty:
    """Small fixed cost per non-resolve step. Encourages efficient investigation."""
    name: str = "step_penalty"
    cost: float = -0.02

    def score(self, steps: list[dict], ground_truth: dict) -> float:
        non_resolve = sum(1 for s in steps if s.get("action", {}).get("tool") != "resolve")
        return self.cost * non_resolve


@dataclass
class Completion:
    """Constant reward for closing the loop with a resolve call (regardless of correctness)."""
    name: str = "completion"
    weight: float = 0.5

    def score(self, steps: list[dict], ground_truth: dict) -> float:
        return self.weight if _resolve_step(steps) is not None else 0.0


@dataclass
class RootCauseMatch:
    """Reward correct root-cause diagnosis, penalize wrong, zero if never resolved."""
    name: str = "root_cause"
    correct: float = 1.5
    wrong: float = -1.0

    def score(self, steps: list[dict], ground_truth: dict) -> float:
        rs = _resolve_step(steps)
        if rs is None:
            return 0.0
        proposed = rs.get("action", {}).get("args", {}).get("root_cause", "")
        return self.correct if proposed == ground_truth.get("root_cause", "") else self.wrong


@dataclass
class ActionMatch:
    """Reward correct remediation action, penalize wrong, zero if never resolved."""
    name: str = "action"
    correct: float = 1.0
    wrong: float = -0.5

    def score(self, steps: list[dict], ground_truth: dict) -> float:
        rs = _resolve_step(steps)
        if rs is None:
            return 0.0
        proposed = rs.get("action", {}).get("args", {}).get("action", "")
        return self.correct if proposed == ground_truth.get("correct_action", "") else self.wrong


def _parse_judge_response(text: str) -> tuple[float | None, str]:
    """Parse the judge's two-line `SCORE: X / REASONING: ...` output.

    Robust to: case differences, extra whitespace, surrounding noise the model
    sometimes emits. Returns (None, "") if SCORE can't be extracted.
    """
    score: float | None = None
    reasoning = ""
    for raw in text.splitlines():
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("SCORE:"):
            try:
                v = float(line.split(":", 1)[1].strip().split()[0])
                score = max(0.0, min(1.0, v))  # clamp into [0, 1]
            except (ValueError, IndexError):
                pass
        elif upper.startswith("REASONING:"):
            try:
                reasoning = line.split(":", 1)[1].strip()
            except IndexError:
                pass
    return score, reasoning


@dataclass
class Rubric:
    """Compose ScoringTerms into a single (total, breakdown, diagnostics) score.

    The synchronous terms in `terms` always run. The async LLM-judge runs on a
    fraction of episodes, deterministic on `run_id`, and is wired in here rather
    than as a sync ScoringTerm because the judge needs an HTTP client, a
    workdir, and is too expensive to fire on every trace.
    """
    terms: list[ScoringTerm] = field(default_factory=list)
    judge_sample_rate: float = 0.1
    judge_model: str = "claude-opus-4-7"
    judge_weight: float = field(
        default_factory=lambda: DEFAULT_WEIGHTS["rationale_quality"]
    )
    judge_max_tokens: int = 200

    def __post_init__(self):
        # AsyncAnthropic is cached on the rubric instance so 100 episodes don't
        # spin up 100 clients. Created lazily — tests that disable the judge
        # never instantiate it.
        self._client = None

    async def score(
        self,
        steps: list[dict],
        ground_truth: dict,
        run_id: str = "",
        workdir: Path | str | None = None,
    ) -> tuple[float, dict[str, float], dict]:
        breakdown = {term.name: term.score(steps, ground_truth) for term in self.terms}
        diagnostics: dict = {"judge_sampled": False, "judge_ran": False}

        if self._should_judge(run_id):
            diagnostics["judge_sampled"] = True
            judge_result = await self._run_judge(steps, ground_truth, workdir)
            if judge_result is not None:
                raw_score, reasoning = judge_result
                # Breakdown stores the WEIGHTED contribution (consistent with
                # other terms); raw 0–1 score lives in diagnostics.
                breakdown["rationale_quality"] = raw_score * self.judge_weight
                diagnostics["judge_ran"] = True
                diagnostics["judge_model"] = self.judge_model
                diagnostics["judge_raw_score"] = raw_score
                diagnostics["judge_reasoning"] = reasoning
            else:
                diagnostics["judge_skip_reason"] = "evidence_unavailable_or_call_failed"

        total = sum(breakdown.values())
        return total, breakdown, diagnostics

    def _should_judge(self, run_id: str) -> bool:
        """Deterministic: same (run_id, sample_rate) → same decision, every process."""
        if not run_id or self.judge_sample_rate <= 0:
            return False
        # SHA-256, not Python hash() — the latter is process-randomized via PYTHONHASHSEED
        # and would break the "same trace always gets the same decision" property.
        h = int(hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8], 16)
        return (h / 0xFFFFFFFF) < self.judge_sample_rate

    async def _run_judge(
        self,
        steps: list[dict],
        ground_truth: dict,
        workdir: Path | str | None,
    ) -> tuple[float, str] | None:
        """Read evidence from workdir, call the judge model, parse the response.
        Returns None on any failure (missing workdir, parse error, API error)."""
        if workdir is None:
            return None
        wd = Path(workdir)
        if not wd.exists():
            return None
        affected = ground_truth.get("affected_service", "")
        log_path = wd / "logs" / f"{affected}.log"
        metrics_dir = wd / "metrics"
        if not log_path.exists() or not metrics_dir.exists():
            return None
        try:
            log_content = log_path.read_text()[:3000]  # truncate for prompt size
            metric_files = sorted(metrics_dir.glob("*.csv"))
            if not metric_files:
                return None
            metric_path = metric_files[0]
            metric_content = metric_path.read_text()[:1500]
        except OSError:
            return None

        rationale = ""
        for s in steps:
            if s.get("action", {}).get("tool") == "resolve":
                rationale = s.get("action", {}).get("args", {}).get("rationale", "") or ""
                break

        prompt = JUDGE_PROMPT.format(
            root_cause=ground_truth.get("root_cause", "unknown"),
            affected_service=affected or "(unknown service)",
            log_content=log_content,
            metric_name=metric_path.name,
            metric_content=metric_content,
            rationale=rationale or "(no rationale provided)",
        )

        try:
            client = self._get_client()
            resp = await client.messages.create(
                model=self.judge_model,
                max_tokens=self.judge_max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            block = resp.content[0]
            text = getattr(block, "text", "")
        except Exception:
            return None

        score, reasoning = _parse_judge_response(text)
        if score is None:
            return None
        return score, reasoning

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic()
        return self._client


def default_rubric(judge_sample_rate: float = 0.1) -> Rubric:
    """Matches the prior env-baked scoring formula exactly for the sync terms,
    plus an LLM-as-judge `rationale_quality` term that fires on a sampled subset.

    total (when judge fires) = step_penalty + completion + root_cause + action
                              + rationale_quality * judge_weight
    """
    return Rubric(
        terms=[StepPenalty(), Completion(), RootCauseMatch(), ActionMatch()],
        judge_sample_rate=judge_sample_rate,
    )
