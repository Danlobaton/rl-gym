"""vLLM client using the native /generate endpoint.

Trade-off vs the OpenAI-compat /v1/completions path:
- token IDs are first-class fields in vLLM's RequestOutput — no `"token_id:N"`
  string parsing dance and no `--return-tokens-as-token-ids` server flag.
- the request accepts pre-tokenized prompts directly via `prompt_token_ids`.
- BUT: vLLM's default `vllm.entrypoints.api_server` /generate handler returns
  only `{"text": [...]}`. To get token_ids + logprobs in the response, the
  operator needs to either (a) run a small custom FastAPI wrapper around
  AsyncLLMEngine that serializes the full RequestOutput (~30 lines), or
  (b) use a community RL fork of vLLM that extends the response shape.
  Frontier RL stacks typically do (a) — the upstream demo server is not
  intended for training.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import httpx

from .types import InferenceResponse


class _TokenizerLike(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
    ) -> str | list[int] | list[str] | list[list[int]] | Any: ...

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def get_vocab(self) -> dict[str, int]: ...

    @property
    def special_tokens_map(self) -> dict[str, str]: ...


@dataclass
class VllmClient:
    """Async client against vLLM's native /generate endpoint.

    `model` is both the request param AND the local tokenizer source — they
    must match exactly, or the prompt the model sees diverges from the prompt
    we recorded. The tokenizer_hash() method emits a fingerprint for the trace.
    """
    base_url: str
    model: str
    max_tokens: int = 512
    temperature: float = 1.0
    timeout: float = 120.0
    _client: httpx.AsyncClient = field(init=False)
    _tokenizer: _TokenizerLike | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=self.timeout)

    @property
    def tokenizer(self) -> _TokenizerLike:
        """Lazy-load — `transformers` is a heavy import we only pay for if
        a vLLM rollout actually runs."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        assert self._tokenizer is not None
        return self._tokenizer

    async def chat(self, messages: list[dict]) -> InferenceResponse:
        # Apply chat template client-side so the server has zero opportunity
        # to re-tokenize. add_generation_prompt appends the assistant-turn
        # prefix the model expects.
        prompt = cast(str, self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))
        prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        r = await self._client.post(
            f"{self.base_url.rstrip('/')}/generate",
            json={
                # vLLM's native API accepts pre-tokenized prompts as a
                # first-class field, unlike /v1/completions where this is a
                # non-standard extension.
                "prompt_token_ids": prompt_token_ids,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "logprobs": 1,            # logprob of chosen token (sufficient for PPO)
                "stream": False,
            },
        )
        r.raise_for_status()
        data = r.json()

        # Expected response shape (RequestOutput serialized):
        #   {"outputs": [{"text": str, "token_ids": list[int], "logprobs": list[dict]}]}
        # Default upstream vllm.entrypoints.api_server returns only {"text": [...]};
        # if you see this error, your server isn't serializing the full RequestOutput.
        if "outputs" not in data:
            raise ValueError(
                "vLLM /generate response missing 'outputs' field. The default "
                "vllm.entrypoints.api_server returns only {'text': [...]} — RL "
                "trajectories require a server that serializes the full "
                "RequestOutput (outputs[i].token_ids + outputs[i].logprobs)."
            )

        out = data["outputs"][0]
        text = out.get("text", "")

        # Token IDs are the load-bearing field. If they're missing or malformed,
        # the trajectory is structurally untrainable — refuse to emit it rather
        # than recording a trace the trainer can't use.
        output_tokens = out.get("token_ids")
        if not isinstance(output_tokens, list) or not all(isinstance(x, int) for x in output_tokens):
            raise ValueError(
                f"vLLM /generate response missing or malformed 'token_ids' in "
                f"outputs[0]: got {type(output_tokens).__name__}. Expected list[int]."
            )

        # vLLM's logprobs field shape: List[Dict[int, Logprob]] — one dict per
        # generated token, mapping {token_id: Logprob}. JSON serialization
        # stringifies int dict keys. The dict typically contains only the
        # chosen token (when logprobs=1) but may have top-K alternates when
        # logprobs>1.
        raw_logprobs = out.get("logprobs") or []
        output_logprobs = _extract_chosen_logprobs(output_tokens, raw_logprobs)

        # Length equality is the alignment invariant the trainer relies on.
        # Refusing to record misaligned data is better than recording corrupted
        # gradients into a year-long training run.
        if len(output_tokens) != len(output_logprobs):
            raise ValueError(
                f"vLLM response misalignment: {len(output_tokens)} tokens vs "
                f"{len(output_logprobs)} logprobs. Refusing to emit a trajectory "
                "with non-positionally-aligned token/logprob arrays."
            )

        return InferenceResponse(
            text=text,
            prompt_tokens=prompt_token_ids,
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
        )

    def tokenizer_hash(self) -> str:
        """Stable fingerprint of the tokenizer's vocab + special tokens.

        Pin this in the trace so a year-old trajectory's `output_tokens`
        re-decode to the same text on the trainer's side — silent tokenizer
        drift across HF Hub revisions is the single most common source of
        mysteriously-broken RL training.
        """
        return _hash_tokenizer(self.tokenizer)

    async def aclose(self) -> None:
        await self._client.aclose()


def _extract_chosen_logprobs(
    output_tokens: list[int],
    raw_logprobs: list[Any],
) -> list[float]:
    """For each generated token, look up its logprob in vLLM's per-position dict.

    JSON serialization of `Dict[int, Logprob]` produces string keys; the
    Logprob value can be either a dict like `{"logprob": -0.5, ...}` or a
    bare float depending on serializer. Handle both. Missing entries coerce
    to 0.0 to keep the array shape stable for the trainer.
    """
    out: list[float] = []
    for i, tok_id in enumerate(output_tokens):
        if i >= len(raw_logprobs):
            out.append(0.0)
            continue
        entry = raw_logprobs[i]
        if entry is None or not isinstance(entry, dict):
            out.append(0.0)
            continue
        # Try string key first (JSON-roundtripped), fall back to int key.
        chosen = entry.get(str(tok_id))
        if chosen is None:
            chosen = entry.get(tok_id)
        if chosen is None:
            out.append(0.0)
            continue
        if isinstance(chosen, dict):
            out.append(float(chosen.get("logprob", 0.0)))
        else:
            out.append(float(chosen))
    return out


def _hash_tokenizer(tokenizer: _TokenizerLike) -> str:
    parts = [
        repr(sorted(tokenizer.get_vocab().items())),
        repr(tokenizer.special_tokens_map),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
