"""JSON-driven pipeline configuration, prompt templates, and document type registry.

Everything configurable lives in JSON (`config/pipeline.json`, `config/prompts.json`, `config/types/*.json`);
secrets and runtime overrides (API keys, optional endpoint/model overrides) live in `.env`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CONFIG = PROJECT_ROOT / "config" / "pipeline.json"
DEFAULT_PROMPTS_FILE = PROJECT_ROOT / "config" / "prompts.json"
DEFAULT_DOC_TYPES_DIR = PROJECT_ROOT / "config" / "types"


def resolve_path(path: str | Path, *, root: Path = PROJECT_ROOT) -> Path:
    """Resolves relative paths against `root` (defaults to the repository root)."""
    p = Path(path)
    return p if p.is_absolute() else root / p


@dataclass(frozen=True)
class PromptTemplates:
    """All prompt templates loaded from `config/prompts.json`."""

    extraction_system_prompt: str = ""
    llm_classify_system_prompt: str = ""
    decider_question: str = ""
    other_description: str = ""
    pdf_infer_system_prompt: str = ""
    gepa_reflection_prompt: str = ""

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PromptTemplates":
        file_path = resolve_path(path) if path else DEFAULT_PROMPTS_FILE
        raw = json.loads(file_path.read_text(encoding="utf-8-sig"))
        return cls(
            extraction_system_prompt=str(raw.get("extraction_system_prompt", "")),
            llm_classify_system_prompt=str(raw.get("llm_classify_system_prompt", "")),
            decider_question=str(raw.get("decider_question", "")),
            other_description=str(raw.get("other_description", "")),
            pdf_infer_system_prompt=str(raw.get("pdf_infer_system_prompt", "")),
            gepa_reflection_prompt=str(raw.get("gepa_reflection_prompt", "")),
        )


@dataclass(frozen=True)
class DocumentType:
    """Typed wrapper around a single `config/types/<type>.json` definition."""

    type_id: str
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.raw.get("nazwa") or self.type_id)

    @property
    def description(self) -> str:
        return str(self.raw.get("opis", ""))

    @property
    def classification_cfg(self) -> dict[str, Any]:
        return dict(self.raw.get("classification") or {})

    @property
    def priority(self) -> int:
        return int(self.classification_cfg.get("priority", 99))

    @property
    def classification_include(self) -> str:
        return str(self.classification_cfg.get("include") or self.description or self.name)

    @property
    def classification_exclude(self) -> str:
        return str(self.classification_cfg.get("exclude") or "")

    @property
    def classification_criteria(self) -> str:
        inc = self.classification_include
        exc = self.classification_exclude
        return f"{inc} {exc}".strip() if exc else inc

    @property
    def extraction_cfg(self) -> dict[str, Any]:
        return dict(self.raw.get("extraction") or {})

    @property
    def extraction_pages(self) -> Any:
        return self.extraction_cfg.get("pages")

    @property
    def extraction_notes(self) -> list[str]:
        return list(self.extraction_cfg.get("uwagi") or [])

    @property
    def lookup_bindings(self) -> list[dict[str, Any]]:
        return list(self.extraction_cfg.get("lookup_bindings") or [])

    @property
    def consistency_rules(self) -> list[dict[str, Any]]:
        return list(self.extraction_cfg.get("consistency_rules") or [])

    @property
    def extraction_lookups(self) -> dict[str, Any]:
        """All lookup tables in `extraction` (everything except control keys)."""
        reserved = {"pages", "uwagi", "lookup_bindings", "consistency_rules"}
        return {k: v for k, v in self.extraction_cfg.items() if k not in reserved}

    @property
    def schema(self) -> dict[str, Any]:
        return dict(self.raw.get("schema") or {})


class DocumentTypeRegistry:
    """Loads and exposes all document types defined in `config/types/*.json`, ordered by priority."""

    def __init__(self, types: dict[str, DocumentType]):
        ordered = sorted(types.items(), key=lambda kv: kv[1].priority)
        self._types: dict[str, DocumentType] = dict(ordered)

    @classmethod
    def from_dir(cls, types_dir: str | Path | None = None) -> "DocumentTypeRegistry":
        folder = resolve_path(types_dir) if types_dir else DEFAULT_DOC_TYPES_DIR
        if not folder.is_dir():
            raise FileNotFoundError(f"document types folder not found: {folder}")
        loaded: dict[str, DocumentType] = {}
        for path in sorted(folder.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            tid = raw["type_id"]
            loaded[tid] = DocumentType(type_id=tid, raw=raw)
        return cls(loaded)

    def get(self, type_id: str) -> DocumentType | None:
        return self._types.get(type_id)

    def items(self):
        return self._types.items()

    def values(self):
        return self._types.values()

    def keys(self):
        return self._types.keys()

    def __contains__(self, type_id: str) -> bool:
        return type_id in self._types

    def __len__(self) -> int:
        return len(self._types)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration loaded from `config/pipeline.json` and `config/prompts.json`."""

    runtime: dict[str, Any] = field(default_factory=dict)
    page_analysis: dict[str, Any] = field(default_factory=dict)
    ocr: dict[str, Any] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    classification: dict[str, Any] = field(default_factory=dict)
    extraction: dict[str, Any] = field(default_factory=dict)
    postprocessing: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    optimization: dict[str, Any] = field(default_factory=dict)
    web: dict[str, Any] = field(default_factory=dict)
    prompts: PromptTemplates = field(default_factory=PromptTemplates)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PipelineConfig":
        cfg_path = resolve_path(path) if path else DEFAULT_PIPELINE_CONFIG
        raw = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        runtime = dict(raw.get("runtime") or {})
        # Backward-compatible fallbacks if cache_dir/types_dir sit at top level
        if "cache_dir" not in runtime and "cache_dir" in raw:
            runtime["cache_dir"] = raw["cache_dir"]
        if "types_dir" not in runtime and "types_dir" in raw:
            runtime["types_dir"] = raw["types_dir"]
        prompts_file = runtime.get("prompts_file", "config/prompts.json")
        prompts = PromptTemplates.load(prompts_file)
        return cls(
            runtime=runtime,
            page_analysis=dict(raw.get("page_analysis") or {}),
            ocr=dict(raw.get("ocr") or {}),
            llm=dict(raw.get("llm") or {}),
            classification=dict(raw.get("classification") or {}),
            extraction=dict(raw.get("extraction") or {}),
            postprocessing=dict(raw.get("postprocessing") or {}),
            evaluation=dict(raw.get("evaluation") or {}),
            optimization=dict(raw.get("optimization") or {}),
            web=dict(raw.get("web") or {}),
            prompts=prompts,
        )

    @property
    def input_dir(self) -> str:
        return str(self.runtime.get("input_dir", "input"))

    @property
    def output_dir(self) -> str:
        return str(self.runtime.get("output_dir", "output"))

    @property
    def default_workers(self) -> int:
        return int(self.runtime.get("default_workers", 4))

    @property
    def cache_dir(self) -> str:
        return str(self.runtime.get("cache_dir", "cache"))

    @property
    def types_dir(self) -> str:
        return str(self.runtime.get("types_dir", "config/types"))

    @property
    def prompts_file(self) -> str:
        return str(self.runtime.get("prompts_file", "config/prompts.json"))

    @property
    def classification_pages(self) -> Any:
        return self.page_analysis.get("classification_pages", "1-3,last")

    @property
    def ocr_engine_name(self) -> str:
        return str(self.ocr.get("engine", "rapid"))

    @property
    def ocr_force(self) -> bool:
        return bool(self.ocr.get("force", False))

    @property
    def render_width_px(self) -> int:
        return int(self.ocr.get("render_width_px", 1654))

    def with_force_ocr(self, force: bool = True) -> "PipelineConfig":
        ocr_copy = dict(self.ocr)
        ocr_copy["force"] = force
        return PipelineConfig(
            runtime=dict(self.runtime),
            page_analysis=dict(self.page_analysis),
            ocr=ocr_copy,
            llm=dict(self.llm),
            classification=dict(self.classification),
            extraction=dict(self.extraction),
            postprocessing=dict(self.postprocessing),
            evaluation=dict(self.evaluation),
            optimization=dict(self.optimization),
            web=dict(self.web),
            prompts=self.prompts,
        )
