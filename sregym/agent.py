from pathlib import Path
from typing import Literal, Protocol

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock
from dotenv import load_dotenv
import json, re, os, time

from .trace import Trace, TraceStep
from .env import StepResult


class EnvProtocol(Protocol):
    def reset(self, seed: int) -> tuple[str, dict]: ...
    def step(self, action: dict) -> "StepResult": ...

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _message(role: Literal["user", "assistant"], content: str) -> MessageParam:
    return {"role": role, "content": content}

SYSTEM_PROMPT = """You are an SRE incident triage agent. Investigate the alert
using read-only tools, then call resolve(root_cause, action).

Valid root_cause values: db_pool_exhausted, oom_killed, bad_deploy, missing_config
Valid actions: scale_db_pool, increase_memory_limit, rollback_deploy, restore_config

Respond ONLY with a JSON object: {"tool": "...", "args": {...}}
"""


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 512


def run_episode(
    env: EnvProtocol,
    seed: int,
    run_id: str,
    run_started_at: float,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    env_url: str | None = None,
) -> Trace:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Put it in a .env file in the project root "
            "or run: export ANTHROPIC_API_KEY=..."
        )
    client = Anthropic(api_key=api_key)

    started = time.time()
    obs, reset_info = env.reset(seed)
    history: list[MessageParam] = [_message("user", obs)]
    steps: list[TraceStep] = []
    total_reward = 0.0
    final_info: dict = {}

    t = 0
    while True:
        step_started = time.time()

        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=history,
        )
        block0 = resp.content[0]
        if not isinstance(block0, TextBlock):
            raise TypeError(f"expected model text block, got {type(block0).__name__}")
        text = block0.text
        history.append(_message("assistant", text))

        action = parse_action(text)
        result = env.step(action)
        latency_ms = int((time.time() - step_started) * 1000)

        steps.append(TraceStep(
            t=t, action=action, observation=result.observation,
            reward=result.reward, info=result.info, latency_ms=latency_ms,
        ))
        total_reward += result.reward
        t += 1

        if result.terminated or result.truncated:
            final_info = result.info
            break

        history.append(_message("user", result.observation))

    return Trace(
        schema_version="1.1",
        run_id=run_id,
        run_started_at=run_started_at,
        agent_name=model,
        agent_config={"model": model, "max_tokens": max_tokens, "env_url": env_url},
        seed=seed,
        task_meta={"incident_type": reset_info.get("incident_type", "")},
        steps=steps,
        total_reward=total_reward,
        reward_breakdown=final_info.get("breakdown", {}),
        ground_truth=final_info.get("ground_truth", {}),
        started_at=started,
        ended_at=time.time(),
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
