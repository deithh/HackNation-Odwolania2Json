"""LLM-based document classifier using prompts & settings loaded from JSON configs."""
from __future__ import annotations

import json

from ..config import DocumentTypeRegistry, PromptTemplates
from ..extraction.client import OpenAICompatibleClient
from ..models import ClassificationResult
from ..utils.text import trim_document_text
from .base import OTHER, DocumentClassifier


class LLMClassifier(DocumentClassifier):
    """Classifies documents via structured JSON output from an OpenAI-compatible open-source LLM."""

    def __init__(
        self,
        client: OpenAICompatibleClient,
        *,
        max_chars: int = 12000,
        min_confidence: float = 0.60,
        temperature: float = 0.0,
        max_tokens: int = 256,
        system_prompt_template: str | None = None,
        other_description: str | None = None,
    ):
        prompts = PromptTemplates.load() if (not system_prompt_template or not other_description) else None
        self._client = client
        self._max_chars = max_chars
        self._min_confidence = min_confidence
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_template = system_prompt_template or (prompts.llm_classify_system_prompt if prompts else "")
        self._other_description = other_description or (prompts.other_description if prompts else "")

    def classify(self, text: str, registry: DocumentTypeRegistry) -> ClassificationResult:
        trimmed = trim_document_text(text, self._max_chars)
        if not trimmed:
            return ClassificationResult(
                doc_type=OTHER,
                method="llm",
                probability=1.0,
                reason="empty document text",
            )

        allowed = [*registry.keys(), OTHER]
        types_block = "\n".join(
            f'- "{tid}" ({dt.name}): {dt.classification_criteria}'
            for tid, dt in registry.items()
        )
        system = self._system_template.format(types_block=types_block, other_desc=self._other_description)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["doc_type", "confidence", "reason"],
            "properties": {
                "doc_type": {"type": "string", "enum": allowed},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reason": {"type": "string"},
            },
        }

        parsed = None
        for json_attempt in range(2):
            raw = self._client.chat(
                f"Treść dokumentu:\n<<<\n{trimmed}\n>>>",
                system=system,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                json_schema=schema,
            )
            try:
                candidate = json.loads(raw)
                if isinstance(candidate, dict):
                    parsed = candidate
                    break
            except json.JSONDecodeError:
                if json_attempt == 0:
                    continue
                raise
        if not isinstance(parsed, dict):
            raise ValueError("LLM classifier did not return a JSON object")
        raw_choice = parsed.get("doc_type", OTHER)
        if raw_choice not in allowed:
            raw_choice = OTHER
        confidence = float(parsed.get("confidence", 0.0))
        reason = str(parsed.get("reason", ""))

        chosen = raw_choice
        if raw_choice != OTHER and confidence < self._min_confidence:
            chosen = OTHER
            reason = f"rejected {raw_choice} (confidence={confidence:.2f} < {self._min_confidence}): {reason}"

        return ClassificationResult(
            doc_type=chosen,
            method="llm",
            probability=confidence,
            raw_choice=raw_choice,
            reason=reason,
        )
