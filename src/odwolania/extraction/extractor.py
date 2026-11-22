"""Structured document extraction orchestrator: prompt -> LLM -> schema post-processing."""
from __future__ import annotations

import json
from typing import Any

from ..config import DocumentType, PipelineConfig
from ..models import ExtractionResult, ExtractionTrace
from .client import LLMConfig, OpenAICompatibleClient
from .postprocessor import SchemaPostprocessor
from .prompt import PromptBuilder


class ExtractionError(RuntimeError):
    """Raised when the LLM fails to return a valid JSON object."""


class DocumentExtractor:
    """Extracts and normalizes structured JSON for any configured `DocumentType`."""

    def __init__(
        self,
        client: OpenAICompatibleClient | None = None,
        *,
        llm_config: LLMConfig | None = None,
        pipeline_config: PipelineConfig | None = None,
        prompt_builder: PromptBuilder | None = None,
        postprocessor: SchemaPostprocessor | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int | None = None,
    ):
        ext_cfg: dict[str, Any] = (pipeline_config.extraction if pipeline_config else None) or {}
        self._client = client or OpenAICompatibleClient(
            llm_config or (LLMConfig.from_pipeline_config(pipeline_config) if pipeline_config else None)
        )
        self._prompt_builder = prompt_builder or PromptBuilder(
            prompts=pipeline_config.prompts if pipeline_config else None
        )
        self._postprocessor = postprocessor or SchemaPostprocessor(
            pipeline_config.postprocessing if pipeline_config else None
        )
        self._temperature = float(ext_cfg.get("temperature", 0.0) if temperature is None else temperature)
        self._max_tokens = int(ext_cfg.get("max_tokens", 3000) if max_tokens is None else max_tokens)
        self._retries = ext_cfg.get("retries") if retries is None else retries

    def extract(
        self,
        text: str,
        doc_type: DocumentType,
        *,
        notes_override: list[str] | None = None,
        template_override: str | None = None,
    ) -> ExtractionResult:
        """Runs LLM extraction and deterministic schema post-processing, returning data + audit trace."""
        system_prompt = self._prompt_builder.build_system_prompt(
            doc_type,
            notes_override=notes_override,
            template_override=template_override,
        )
        user_prompt = self._prompt_builder.build_user_prompt(text)
        relaxed_schema = self._prompt_builder.relax_schema_for_decoding(doc_type.schema)

        raw_output = ""
        parsed: Any = None
        last_exc: Exception | None = None
        for json_attempt in range(2):
            raw_output = self._client.chat(
                user_prompt,
                system=system_prompt,
                temperature=self._temperature,
                json_schema=relaxed_schema,
                max_tokens=self._max_tokens,
                retries=self._retries,
            )
            try:
                parsed = json.loads(raw_output)
                if not isinstance(parsed, dict):
                    raise ExtractionError(f"model returned {type(parsed).__name__}, expected JSON object")
                last_exc = None
                break
            except (json.JSONDecodeError, ExtractionError) as exc:
                last_exc = exc
                if json_attempt == 0:
                    continue
        trace = ExtractionTrace(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output=raw_output,
        )
        if last_exc is not None:
            if isinstance(last_exc, ExtractionError):
                raise last_exc
            raise ExtractionError(
                f"model returned invalid JSON ({last_exc}); output starts: {raw_output[:200]!r}"
            ) from last_exc

        trace.raw_parsed = parsed
        cleaned = self._postprocessor.process(parsed, doc_type, source_text=text)
        return ExtractionResult(data=cleaned, trace=trace)
