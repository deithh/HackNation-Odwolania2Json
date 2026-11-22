"""OpenAI-compatible LLM client for open-source models (defaults loaded from config/pipeline.json, overridable via .env)."""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

from ..config import PROJECT_ROOT, PipelineConfig
from ..utils.text import extract_json_block

load_dotenv(PROJECT_ROOT / ".env")

RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, httpx.TransportError)
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


def _default_api_key() -> str:
    return os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""


def _is_local_url(url: str) -> bool:
    lowered = url.lower()
    return any(host in lowered for host in ("localhost", "127.0.0.1", "0.0.0.0"))


def _compute_backoff(attempt: int, retry_after_header: str | None = None, *, cap: float = 30.0) -> float:
    base = float(min(2 ** attempt, cap))
    if retry_after_header:
        try:
            base = max(base, min(float(retry_after_header), cap))
        except ValueError:
            pass
    return base + random.uniform(0.0, 0.75)


@dataclass(frozen=True)
class LLMConfig:
    """Immutable connection and model configuration (JSON defaults + optional `.env` overrides)."""

    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or "https://api.regolo.ai/v1")
    api_key: str = field(default_factory=_default_api_key)
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL") or "qwen3.5-9b")
    timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "300")))
    retries: int = 4
    thinking_kwarg: str = field(default_factory=lambda: os.getenv("LLM_THINKING_KWARG") or "auto")

    @classmethod
    def from_pipeline_config(cls, config: PipelineConfig) -> "LLMConfig":
        llm_cfg = config.llm or {}
        return cls(
            base_url=os.getenv("LLM_BASE_URL") or str(llm_cfg.get("base_url") or "https://api.regolo.ai/v1"),
            api_key=_default_api_key(),
            model=os.getenv("LLM_MODEL") or str(llm_cfg.get("model") or "qwen3.5-9b"),
            timeout=float(os.getenv("LLM_TIMEOUT") or llm_cfg.get("timeout", 300.0)),
            retries=int(llm_cfg.get("retries", 4)),
            thinking_kwarg=str(os.getenv("LLM_THINKING_KWARG") or llm_cfg.get("thinking_kwarg", "auto")),
        )


class OpenAICompatibleClient:
    """Resilient OpenAI-compatible chat client with structured-output fallback, backoff+jitter, and Retry-After."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client: OpenAI | None = None

    def _get_sdk_client(self) -> OpenAI:
        if self._client is None:
            api_key = self.config.api_key
            if not api_key:
                if _is_local_url(self.config.base_url):
                    api_key = "EMPTY"
                else:
                    raise RuntimeError("LLM_API_KEY (or OPENROUTER_API_KEY) is not set (add it to .env)")
            # Disable hidden SDK retries so our unified loop controls backoff, jitter, and Retry-After.
            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=api_key,
                timeout=self.config.timeout,
                max_retries=0,
            )
        return self._client

    def list_models(self) -> list[str]:
        return [m.id for m in self._get_sdk_client().models.list().data]

    def _should_send_thinking_kwarg(self, thinking: bool) -> bool:
        mode = self.config.thinking_kwarg.strip().lower()
        if mode in ("1", "true", "yes"):
            return True
        if mode in ("0", "false", "no"):
            return False
        if thinking:
            return True
        return _is_local_url(self.config.base_url) and "qwen3" in self.config.model.lower()

    def chat(
        self,
        messages: list[dict[str, Any]] | str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
        thinking: bool = False,
        retries: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Executes a chat completion with json_schema -> json_object -> plain degradation and resilient retries."""
        msg_list: list[dict[str, Any]] = (
            [{"role": "user", "content": messages}] if isinstance(messages, str) else list(messages)
        )
        if system:
            msg_list = [{"role": "system", "content": system}, *msg_list]

        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        if self._should_send_thinking_kwarg(thinking):
            extra_body.setdefault("chat_template_kwargs", {"enable_thinking": thinking})
        elif "openrouter.ai" in self.config.base_url.lower() and not thinking:
            extra_body.setdefault("reasoning", {"effort": "none"})

        expect_json = json_schema is not None or json_mode
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema, "strict": True},
            }
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        max_retries = self.config.retries if retries is None else retries
        sdk = self._get_sdk_client()
        attempt = 0
        while True:
            call_kwargs = dict(kwargs)
            if extra_body:
                call_kwargs["extra_body"] = extra_body
            try:
                response = sdk.chat.completions.create(
                    model=self.config.model,
                    messages=msg_list,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **call_kwargs,
                )
                choices = getattr(response, "choices", None) or []
                content = (choices[0].message.content or "") if choices else ""
                extracted = extract_json_block(content, expect_json=expect_json)
                if not extracted and attempt < max_retries:
                    time.sleep(_compute_backoff(attempt))
                    attempt += 1
                    continue
                return extracted
            except (BadRequestError, NotFoundError) as exc:
                msg = str(exc).lower()
                if "chat_template_kwargs" in extra_body and ("chat_template" in msg or "extra" in msg or "unknown" in msg):
                    extra_body.pop("chat_template_kwargs", None)
                    continue
                if "reasoning" in extra_body and ("reasoning" in msg or "effort" in msg or "no endpoints" in msg):
                    extra_body.pop("reasoning", None)
                    continue
                rf = kwargs.get("response_format")
                if isinstance(rf, dict):
                    rf_type = rf.get("type")
                    if rf_type == "json_schema":
                        kwargs["response_format"] = {"type": "json_object"}
                        continue
                    if rf_type == "json_object":
                        kwargs.pop("response_format", None)
                        continue
                raise
            except RETRYABLE_ERRORS as exc:
                if attempt >= max_retries:
                    raise
                retry_after = None
                resp = getattr(exc, "response", None)
                if resp is not None:
                    retry_after = resp.headers.get("retry-after")
                time.sleep(_compute_backoff(attempt, retry_after))
                attempt += 1
            except APIStatusError as exc:
                if exc.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                    retry_after = exc.response.headers.get("retry-after") if exc.response is not None else None
                    time.sleep(_compute_backoff(attempt, retry_after))
                    attempt += 1
                    continue
                raise
