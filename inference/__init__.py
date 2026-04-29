"""Inference backends for the agent loop.

Separated from `sregym/` because inference is its own concern with its own
lifecycle: backends swap (vLLM today, SGLang tomorrow, trainer-internal in
production) without touching env code. Every backend implements the same
`InferenceClient` protocol so the agent loop is backend-agnostic.
"""
from .types import InferenceClient, InferenceResponse

__all__ = ["InferenceClient", "InferenceResponse"]
