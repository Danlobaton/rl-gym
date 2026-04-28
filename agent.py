from typing import Literal

from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlock
import json, re, os

from gym import IncidentEnv


def _message(role: Literal["user", "assistant"], content: str) -> MessageParam:
    return {"role": role, "content": content}

SYSTEM_PROMPT = """You are an SRE incident triage agent. Investigate the alert
using read-only tools, then call resolve(root_cause, action).

Valid root_cause values: db_pool_exhausted, oom_killed, bad_deploy, missing_config
Valid actions: scale_db_pool, increase_memory_limit, rollback_deploy, restore_config

Respond ONLY with a JSON object: {"tool": "...", "args": {...}}
"""

def run_episode(env: IncidentEnv, seed: int) -> float:
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    obs, info = env.reset(seed)
    history: list[MessageParam] = [_message("user", obs)]
    total_reward = 0.0

    while True:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=history,
        )
        block0 = resp.content[0]
        if not isinstance(block0, TextBlock):
            raise TypeError(f"expected model text block, got {type(block0).__name__}")
        text = block0.text
        history.append(_message("assistant", text))

        # parse JSON action (robustly)
        action = parse_action(text)
        result = env.step(action)
        total_reward += result.reward

        if result.terminated or result.truncated:
            return total_reward

        history.append(_message("user", result.observation))

def parse_action(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    return {"tool": "resolve", "args": {"root_cause": "unknown", "action": "unknown"}}