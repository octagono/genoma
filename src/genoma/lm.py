"""Shared LM construction: one place for model IDs, kwargs, adapters.

dspy.LM(model, **kwargs) with cache_control_injection_points for Anthropic
provider caching, and ChatAdapter(use_native_function_calling=True,
parallel_tool_calls=True) for ReActV2 native tool calling.

Also hosts a client-side **token-bucket rate limiter**: under high-volume
parallel scoring, waiting for a 429 and backing off can still trip provider
Tier limits (TPM/RPM). The bucket estimates input tokens per request *before*
sending and delays host-side when the current minute approaches 80% of the
configured budget — preventing the 429s from ever occurring. Disabled by
default (rpm/tpm = 0); enabled via ``LM_RPM``/``LM_TPM`` env or config.
"""

from __future__ import annotations

import os
import threading
import time
from functools import lru_cache

import dspy

from genoma import config as cfg


class TokenBucket:
    """Thread-safe token-bucket rate limiter (token-based, not fixed-window).

    Refills ``capacity`` tokens over ``refill_seconds``; ``acquire(n)`` blocks
    until ``n`` tokens are available (or raises if it would exceed the cap).
    ``rpm``/``tpm`` of 0 mean unlimited (no limiting).
    """

    def __init__(self, *, capacity: float = 0.0, refill_seconds: float = 60.0) -> None:
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._refill_seconds = max(1e-6, float(refill_seconds))
        self._rate = self._capacity / self._refill_seconds  # tokens per second
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float) -> None:
        if self._capacity <= 0:
            return  # unlimited
        with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                deficit = n - self._tokens
                sleep_for = deficit / self._rate
                # Cap the wait so a huge burst can't stall a thread forever.
                time.sleep(min(sleep_for, 5.0))


def _estimate_input_tokens(messages) -> int:
    """Rough input-token estimate (4 chars/token) for rate-limit budgeting."""
    total = 0
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else str(m)
        if isinstance(content, list):  # multimodal parts
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(part.get("text", "")) // 4
        else:
            total += len(str(content)) // 4
    return max(1, total)


# Module-level singleton so all LM instances share ONE budget per process.
# Reads cfg.LM_TPM (env LM_TPM); 0 = unlimited (no rate limiting).
_bucket = TokenBucket(capacity=float(cfg.LM_TPM), refill_seconds=60.0)


class RateLimitedLM(dspy.LM):
    """dspy.LM wrapper that token-budgets requests client-side.

    Acquires estimated input tokens before delegating to the underlying LM;
    the bucket delays host-side when the configured per-minute TPM budget is
    near exhaustion, so provider 429s / Tier-limit blocks are avoided rather
    than retried. No-op when the budget is 0 (unlimited).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._bucket = _bucket

    @property
    def supports_response_schema(self) -> bool:
        """False for providers whose API rejects ``json_schema`` response_format.

        litellm reports ``supports_response_schema=True`` for DeepSeek, but the
        live API answers 400 "This response_format type is unavailable now" to
        ``json_schema`` while accepting plain ``json_object``. Returning False
        here makes dspy's JSON/BAML adapter fall back to
        ``{"type": "json_object"}`` (the prompt already contains "json", which
        the json_object mode requires).
        """
        if self.model.startswith("deepseek/") or "/deepseek/" in self.model:
            return False
        return super().supports_response_schema

    def __call__(self, *args, **kwargs) -> dspy.LMOutput:
        if self._bucket._capacity > 0:
            messages = kwargs.get("messages")
            if messages is None and args:
                messages = args[0]
            self._bucket.acquire(_estimate_input_tokens(messages))
        return super().__call__(*args, **kwargs)


@lru_cache(maxsize=8)
def make_lm(model: str, **kwargs) -> dspy.LM:
    """Build a dspy.LM, merging model-specific defaults + env overrides."""
    merged = {"cache": True, "num_retries": 3, "timeout": 60}
    if "anthropic/" in model:
        merged.update(cfg.PRIMARY_LM_KWARGS)
    if model.startswith("deepseek/") or "/deepseek/" in model:
        # DeepSeek v4-pro defaults to a reasoning mode that occasionally emits
        # the final answer ONLY in reasoning_content (content stays empty),
        # which breaks dspy's adapter. Disable thinking: the answer then lands
        # reliably in content and output tokens drop.
        merged["thinking"] = {"type": "disabled"}
    merged.update(kwargs)
    api_key = os.environ.get("OPENROUTER_API_KEY") if "openrouter/" in model else None
    if api_key:
        merged["api_key"] = api_key
    return RateLimitedLM(model, **merged)


def primary_lm() -> dspy.LM:
    return make_lm(cfg.PRIMARY_LM)


def sub_lm() -> dspy.LM:
    return make_lm(cfg.SUB_LM)


def reflection_lm() -> dspy.LM:
    return make_lm(cfg.REFLECTION_LM, **cfg.REFLECTION_LM_KWARGS)


def panel_lm(model_slug: str) -> dspy.LM:
    """A panel member model (heterogeneous consensus panel)."""
    return make_lm(model_slug, temperature=cfg.PANEL_TEMPERATURE)


def native_tool_adapter() -> dspy.ChatAdapter:
    """ChatAdapter with native function calling + parallel tool calls (ReActV2)."""
    return dspy.ChatAdapter(
        use_native_function_calling=True,
        parallel_tool_calls=True,
    )


def default_adapter() -> dspy.JSONAdapter:
    """Structured-output adapter for signatures with pydantic models.

    Uses BAMLAdapter (`dspy.adapters.baml_adapter`, not a top-level `dspy.`
    export in 3.3.1) because it renders Pydantic schemas far more
    token-efficiently than JSONAdapter — smaller schema prompts, especially
    for list-typed outputs. It is a JSONAdapter subclass, so native function
    calling + structured outputs + json_repair fallback are inherited.
    """
    from dspy.adapters.baml_adapter import BAMLAdapter

    return BAMLAdapter()


def configure_defaults() -> None:
    """Global DSPy defaults: primary LM + BAML adapter for structured output.

    BAMLAdapter enforces the pydantic output schema (e.g. `epcr` ge=0 le=1)
    with ~2.4x smaller schema prompts than JSONAdapter on list-typed outputs.
    `dspy.context(lm=...)` overrides per-block for panels and debate rounds.

    Also hardens the DSPy disk cache: ``restrict_pickle`` confines pickle
    deserialization to a known-safe allowlist (dspy.core.types.* — the
    LMResponse/LMOutput/LMTextPart/LMUsage objects the cache stores) plus the
    library-trusted prefixes (litellm.types.*, openai.types.*, numpy helpers).
    An out-of-allowlist entry raises DeserializationError and is deleted
    (degrades to a cache miss, never a crash) — the cache stores LM responses
    verbatim.
    """
    from dspy.clients import configure_cache
    from dspy.core import LMOutput, LMResponse, LMTextPart, LMUsage

    configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=str(cfg.CACHE_DIR),
        restrict_pickle=True,
        safe_types=[LMResponse, LMOutput, LMTextPart, LMUsage],
    )
    dspy.configure(lm=primary_lm(), adapter=default_adapter())
