"""Anthropic backend — closed-API, no logprobs available.

Useful for demo / eval rollouts where you want to interact with Claude. Not
useful for RL training — you can't sample logprobs and you can't update model
weights via the API. Use VllmClient for trainable rollouts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .types import InferenceResponse

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types import MessageParam, TextBlock


@dataclass
class AnthropicClient:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 512
    _client: AsyncAnthropic = field(init=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic()

    async def chat(self, messages: list[dict[str, Any]]) -> InferenceResponse:
        # Anthropic's API treats system as a top-level field, not a message.
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        msgs: list[MessageParam] = [m for m in messages if m["role"] != "system"]  # type: ignore[misc]

        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system="\n".join(system_parts) if system_parts else "",
            messages=msgs,
        )
        from anthropic.types import TextBlock
        block0 = resp.content[0] if resp.content else None
        text = block0.text if isinstance(block0, TextBlock) else ""
        # No prompt_tokens / output_tokens / output_logprobs — Anthropic's API
        # doesn't return them. The trace will record None for these fields,
        # which signals "this trajectory is not RL-trainable."
        return InferenceResponse(text=text)
