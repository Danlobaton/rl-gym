from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
import json, re, time

from inference import InferenceClient

from .trace import Trace, TraceStep
from .env import StepResult
from .rubric import Rubric, default_rubric


class EnvProtocol(Protocol):
    def reset(self, seed: int) -> tuple[str, dict]: ...
    def step(self, action: dict) -> "StepResult": ...

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


SYSTEM_PROMPT = """You are an SRE incident triage agent. Investigate the alert
using read-only tools, then call resolve(root_cause, action).

Valid root_cause values: db_pool_exhausted, oom_killed, bad_deploy, missing_config
Valid actions: scale_db_pool, increase_memory_limit, rollback_deploy, restore_config

Respond ONLY with a JSON object: {"tool": "...", "args": {...}}
"""


def _default_inference() -> InferenceClient:
    """Backwards-compat default: Anthropic with the original demo settings.

    Production / RL-training rollouts inject a VllmClient explicitly so the
    trace captures token IDs + logprobs.
    """
    from inference.anthropic_client import AnthropicClient
    return AnthropicClient()


async def run_episode(
    env: EnvProtocol,
    seed: int,
    run_id: str,
    run_started_at: float,
    env_url: str | None = None,
    rubric: Rubric | None = None,
    inference: InferenceClient | None = None,
) -> Trace:
    if inference is None:
        inference = _default_inference()
    if rubric is None:
        rubric = default_rubric()

    started = time.time()
    obs, reset_info = env.reset(seed)
    # Single-history shape that every backend understands; AnthropicClient
    # strips the system message and remaps to its native format internally.
    history: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs},
    ]
    steps: list[TraceStep] = []
    final_info: dict = {}

    t = 0
    while True:
        step_started = time.time()

        resp = await inference.chat(history)
        text = resp.text
        history.append({"role": "assistant", "content": text})

        action = parse_action(text)
        result = env.step(action)
        latency_ms = int((time.time() - step_started) * 1000)

        steps.append(TraceStep(
            t=t,
            action=action,
            observation=result.observation,
            reward=result.reward,
            info=result.info,
            latency_ms=latency_ms,
            # Token fields flow through unchanged: None for closed APIs,
            # populated for vLLM/SGLang/TGI. The trainer reads these directly.
            prompt_tokens=resp.prompt_tokens,
            output_tokens=resp.output_tokens,
            output_logprobs=resp.output_logprobs,
        ))
        t += 1

        if result.terminated or result.truncated:
            final_info = result.info
            break

        history.append({"role": "user", "content": result.observation})

    # Episode-level scoring: rubric reads the completed trajectory + ground truth.
    # Augment ground_truth with affected_service so the LLM-judge can locate the
    # right log file in the workdir; pulled from reset_info, not env terminal info,
    # so this is purely additive and doesn't affect replay's per-step info checks.
    ground_truth = dict(final_info.get("ground_truth", {}))
    if reset_info.get("affected_service"):
        ground_truth["affected_service"] = reset_info["affected_service"]

    # Workdir is in-process-only — EnvClient (HTTP) doesn't expose .incident,
    # so getattr returns None and the judge skips gracefully.
    workdir = getattr(getattr(env, "incident", None), "workdir", None)

    total_reward, breakdown, diagnostics = await rubric.score(
        [asdict(s) for s in steps],
        ground_truth,
        run_id=run_id,
        workdir=workdir,
    )

    # Pull tokenizer fingerprint + sampling params from the inference client.
    # Both are best-effort — only vLLM-style backends populate them; closed
    # APIs leave them empty so the trace honestly says "this isn't trainable."
    tokenizer_hash = ""
    sampling_params: dict = {}
    if hasattr(inference, "tokenizer_hash"):
        try:
            tokenizer_hash = inference.tokenizer_hash()
        except Exception:
            tokenizer_hash = ""
    if hasattr(inference, "temperature") and hasattr(inference, "max_tokens"):
        sampling_params = {
            "temperature": getattr(inference, "temperature", None),
            "max_tokens": getattr(inference, "max_tokens", None),
        }

    return Trace(
        schema_version="1.3",
        run_id=run_id,
        run_started_at=run_started_at,
        agent_name=getattr(inference, "model", "unknown"),
        agent_config={"model": getattr(inference, "model", "unknown"), "env_url": env_url},
        seed=seed,
        task_meta={"incident_type": reset_info.get("incident_type", "")},
        steps=steps,
        total_reward=total_reward,
        reward_breakdown=breakdown,
        ground_truth=ground_truth,
        started_at=started,
        ended_at=time.time(),
        diagnostics=diagnostics,
        policy_model=getattr(inference, "model", ""),
        policy_model_revision="",  # populated when vLLM exposes it; out of Phase 0 scope
        tokenizer_hash=tokenizer_hash,
        sampling_params=sampling_params,
    )


def parse_action(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    print(f"[parse_error] could not parse model output:\n---\n{text}\n---", flush=True)
    return {"tool": "resolve", "args": {"root_cause": "unknown", "action": "unknown"}}
