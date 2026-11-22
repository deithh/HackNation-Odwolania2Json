"""Remote Decider classifier (systemone / Decisions API: single-call choice + per-class noul)."""
from __future__ import annotations

import os
import random
import time
from collections.abc import Mapping
from typing import Any

import httpx
from dotenv import load_dotenv

from ..config import PROJECT_ROOT, DocumentTypeRegistry, PromptTemplates
from ..models import ClassificationResult
from ..utils.text import trim_document_text
from .base import OTHER, DocumentClassifier

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DECIDER_URL = "https://api.regolo.ai/v1/systemone"
DEFAULT_DECIDER_MODEL = "clef-flash"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


class DeciderRemoteError(RuntimeError):
    """Raised when the remote Decisions API fails after retries."""


class DeciderClient:
    """HTTP client for the Decisions API endpoint (JSON defaults + optional `.env` overrides)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
        connect_timeout: float = 20.0,
        retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ):
        self._api_key = api_key
        self._url = url
        self._model = model
        self._timeout = timeout
        self._connect_timeout = connect_timeout
        self._retries = retries
        self._transport = transport

    def ask_many(self, state: str | dict | list, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Evaluates multiple typed questions (`choice`, `noul`, `score`, `locate`) in a single API call."""
        api_key = self._api_key or os.getenv("DECIDER_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise DeciderRemoteError("DECIDER_API_KEY / LLM_API_KEY is not set (add it to .env)")
        url = os.getenv("DECIDER_REMOTE_URL") or self._url or DEFAULT_DECIDER_URL
        model = os.getenv("DECIDER_REMOTE_MODEL") or self._model or DEFAULT_DECIDER_MODEL
        body = {"model": model, "state": state, "questions": dict(questions)}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        client_timeout = httpx.Timeout(self._timeout, connect=self._connect_timeout)

        with httpx.Client(timeout=client_timeout, transport=self._transport) as client:
            for attempt in range(self._retries + 1):
                try:
                    resp = client.post(url, headers=headers, json=body)
                except httpx.TransportError as exc:
                    if attempt < self._retries:
                        time.sleep(min(2 ** attempt, 30) + random.uniform(0.0, 0.75))
                        continue
                    raise DeciderRemoteError(
                        f"Decider transport error after {self._retries + 1} attempts: {exc}"
                    ) from exc
                if resp.status_code in RETRY_STATUS and attempt < self._retries:
                    delay = float(min(2 ** attempt, 30))
                    retry_after = resp.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = max(delay, min(float(retry_after), 30.0))
                        except ValueError:
                            pass
                    time.sleep(delay + random.uniform(0.0, 0.75))
                    continue
                if resp.status_code != 200:
                    raise DeciderRemoteError(f"Decider returned {resp.status_code}: {resp.text[:500]}")
                try:
                    payload = resp.json()
                    answers = payload["answers"]
                except (ValueError, KeyError, TypeError) as exc:
                    if attempt < self._retries:
                        time.sleep(min(2 ** attempt, 30) + random.uniform(0.0, 0.75))
                        continue
                    raise DeciderRemoteError(f"Invalid decider JSON response: {exc}") from exc
                return answers
        raise DeciderRemoteError("unreachable")


class DeciderClassifier(DocumentClassifier):
    """Multi-question document classifier using `choice` + per-class binary `noul` verification."""

    def __init__(
        self,
        *,
        client: DeciderClient | None = None,
        max_chars: int = 12000,
        min_choice_prob: float = 0.55,
        min_noul_prob: float = 0.50,
        other_description: str | None = None,
        question: str | None = None,
    ):
        prompts = PromptTemplates.load() if (not other_description or not question) else None
        self._client = client or DeciderClient()
        self._max_chars = max_chars
        self._min_choice_prob = min_choice_prob
        self._min_noul_prob = min_noul_prob
        self._other_description = other_description or (prompts.other_description if prompts else "")
        self._question = question or (prompts.decider_question if prompts else "")

    def classify(self, text: str, registry: DocumentTypeRegistry) -> ClassificationResult:
        state_text = trim_document_text(text, self._max_chars)
        if not state_text:
            return ClassificationResult(
                doc_type=OTHER,
                method="decider",
                probability=1.0,
                probabilities={**{tid: 0.0 for tid in registry.keys()}, OTHER: 1.0},
                noul_scores={tid: 0.0 for tid in registry.keys()},
                reason="empty document text",
            )

        options = {tid: dt.classification_criteria for tid, dt in registry.items()}
        options[OTHER] = self._other_description

        questions: dict[str, dict[str, Any]] = {
            "doc_type": {
                "type": "choice",
                "instructions": self._question,
                "criteria": options,
            }
        }
        for tid, dt in registry.items():
            false_means = (
                dt.classification_exclude
                or f"Dokument NIE jest typu '{dt.name}' — jest innym dokumentem lub tylko wspomina o '{dt.name}'."
            )
            questions[f"is_{tid}"] = {
                "type": "noul",
                "instructions": f"Czy ten dokument faktycznie JEST dokumentem typu: {dt.name}?",
                "criteria": {"true": dt.classification_include, "false": false_means},
            }

        answers = self._client.ask_many(state_text, questions)
        choice_ans = answers["doc_type"]
        raw_choice: str = choice_ans["choice"]
        probabilities: dict[str, float] = {k: float(v) for k, v in choice_ans["probabilities"].items()}
        noul_scores = {tid: float(answers[f"is_{tid}"]["noul"]) for tid in registry.keys()}

        chosen = raw_choice
        reason = f"choice={raw_choice} ({probabilities.get(raw_choice, 0.0):.3f})"
        if raw_choice != OTHER:
            p_choice = probabilities.get(raw_choice, 0.0)
            p_noul = noul_scores.get(raw_choice, 0.0)
            if p_choice < self._min_choice_prob or p_noul < self._min_noul_prob:
                chosen = OTHER
                reason = (
                    f"rejected {raw_choice}: p_choice={p_choice:.3f} (min {self._min_choice_prob}), "
                    f"p_noul={p_noul:.3f} (min {self._min_noul_prob})"
                )
            else:
                reason = f"accepted {raw_choice}: p_choice={p_choice:.3f}, p_noul={p_noul:.3f}"

        return ClassificationResult(
            doc_type=chosen,
            method="decider",
            probability=probabilities.get(chosen, 0.0),
            probabilities=probabilities,
            noul_scores=noul_scores,
            raw_choice=raw_choice,
            reason=reason,
        )
