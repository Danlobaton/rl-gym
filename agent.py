from pathlib import Path
from typing import Literal

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock
from dotenv import load_dotenv
import json, re, os, time

from gym import IncidentEnv
from episode_trace import Trace, TraceStep

load_dotenv(Path(__file__).resolve().parent / ".env")


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
    env: IncidentEnv,
    seed: int,
    run_id: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
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
        schema_version="1.0",
        run_id=run_id,
        agent_name=model,
        agent_config={"model": model, "max_tokens": max_tokens},
        seed=seed,
        task_meta={"incident_type": env.incident.incident_type},
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
