"""Shared types across inference backends."""
from dataclasses import dataclass
from typing import Protocol


@dataclass
class InferenceResponse:
    """One model call's output. Token fields are optional because closed-API
    backends (Anthropic, OpenAI) can't expose them — only open-weight servers
    (vLLM, SGLang, TGI) return token IDs and per-token logprobs.

    For RL training, you NEED prompt_tokens + output_tokens + output_logprobs.
    For demo / eval rollouts against a closed model, text alone is enough.
    """
    text: str
    prompt_tokens: list[int] | None = None
    output_tokens: list[int] | None = None
    output_logprobs: list[float] | None = None  # log P(token_t | prefix) under the policy


class InferenceClient(Protocol):
    """Async interface every backend must implement.

    `messages` follows OpenAI's chat-completion shape: list of {role, content}
    where role ∈ {system, user, assistant}. Backends are responsible for
    translating to their native format if it differs.
    """
    model: str
    async def chat(self, messages: list[dict]) -> InferenceResponse: ...
