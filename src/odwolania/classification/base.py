"""Abstract document classifier interface, automatic fallback wrapper, and JSON-config-driven factory."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..config import DocumentTypeRegistry, PromptTemplates
from ..extraction.client import LLMConfig, OpenAICompatibleClient
from ..models import ClassificationResult

OTHER = "other"


class DocumentClassifier(ABC):
    """Base contract for document-type classifiers returning a `ClassificationResult`."""

    @abstractmethod
    def classify(self, text: str, registry: DocumentTypeRegistry) -> ClassificationResult:
        """Assigns one registered `type_id` or `OTHER` ('other') to the document text."""


class FallbackClassifier(DocumentClassifier):
    """Runs `primary` classifier and falls back to `fallback` (e.g. `RuleClassifier`) only if primary fails."""

    def __init__(self, primary: DocumentClassifier, fallback: DocumentClassifier):
        self._primary = primary
        self._fallback = fallback

    def classify(self, text: str, registry: DocumentTypeRegistry) -> ClassificationResult:
        try:
            return self._primary.classify(text, registry)
        except Exception as exc:
            res = self._fallback.classify(text, registry)
            res.method = f"{res.method}(fallback)"
            res.details["fallback_from_error"] = f"{type(exc).__name__}: {exc}"
            res.reason = f"fallback after {type(exc).__name__}: {res.reason}"
            return res


class ClassifierFactory:
    """Instantiates the classifier configured in `config/pipeline.json` and `config/prompts.json`."""

    @staticmethod
    def create(
        cls_cfg: dict[str, Any] | None = None,
        *,
        prompts: PromptTemplates | None = None,
        llm_client: OpenAICompatibleClient | None = None,
        llm_config: LLMConfig | None = None,
    ) -> DocumentClassifier:
        cfg = cls_cfg or {}
        tpls = prompts or PromptTemplates.load()
        method = str(cfg.get("method", "decider")).lower()
        use_fallback = bool(cfg.get("fallback_to_rules", True))

        if method == "decider":
            from .decider import DeciderClassifier, DeciderClient

            dec_cfg = cfg.get("decider") or {}
            client = DeciderClient(
                url=dec_cfg.get("url"),
                model=dec_cfg.get("model"),
                timeout=float(dec_cfg.get("timeout", 120.0)),
                connect_timeout=float(dec_cfg.get("connect_timeout", 20.0)),
                retries=int(dec_cfg.get("retries", 3)),
            )
            primary: DocumentClassifier = DeciderClassifier(
                client=client,
                max_chars=int(cfg.get("max_chars", 12000)),
                min_choice_prob=float(cfg.get("min_choice_prob", 0.55)),
                min_noul_prob=float(cfg.get("min_noul_prob", 0.50)),
                other_description=cfg.get("other_description") or tpls.other_description,
                question=cfg.get("decider_question") or tpls.decider_question,
            )
        elif method == "llm":
            from .llm import LLMClassifier

            llm_cls_cfg = cfg.get("llm") or {}
            client = llm_client or OpenAICompatibleClient(llm_config)
            primary = LLMClassifier(
                client=client,
                max_chars=int(cfg.get("max_chars", 12000)),
                min_confidence=float(cfg.get("min_llm_confidence", 0.60)),
                temperature=float(llm_cls_cfg.get("temperature", 0.0)),
                max_tokens=int(llm_cls_cfg.get("max_tokens", 256)),
                system_prompt_template=tpls.llm_classify_system_prompt,
                other_description=cfg.get("other_description") or tpls.other_description,
            )
        elif method == "rules":
            from .rules import RuleClassifier

            return RuleClassifier()
        else:
            raise ValueError(f"unknown classification method: {method!r} (use 'decider', 'llm', or 'rules')")

        if use_fallback:
            from .rules import RuleClassifier

            return FallbackClassifier(primary=primary, fallback=RuleClassifier())
        return primary
