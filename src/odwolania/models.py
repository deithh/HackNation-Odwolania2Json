"""Core data models shared across the OCR, classification, extraction, and pipeline layers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PageText:
    """Extracted text and layout statistics for a single PDF page (1-based index)."""

    index: int
    source: str                 # pdf | ocr | none
    text: str = ""
    kind: str = ""              # text | scan | ocr_layer | blank
    chars: int = 0
    image_coverage: float = 0.0
    text_quality: float = 0.0
    invisible_share: float = 0.0
    skew_angle: float = 0.0
    spans: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DocumentText:
    """Aggregated text and extraction strategy for a PDF document."""

    file: str
    sha1: str
    strategy: str               # text | hybrid | ocr | error
    strategy_reason: str
    n_pages: int = 0
    pages: list[PageText] = field(default_factory=list)
    requested: list[int] = field(default_factory=list)

    def text_for(self, pages: list[int] | None = None) -> str:
        """Returns joined text for the requested 1-based page indices."""
        wanted = pages if pages is not None else self.requested
        selected = [p for p in self.pages if not wanted or p.index in wanted]
        return "\n".join(p.text for p in selected if p.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationResult:
    """Outcome of document-type classification with confidence and audit metadata."""

    doc_type: str
    method: str
    probability: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    noul_scores: dict[str, float] = field(default_factory=dict)
    raw_choice: str | None = None
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "doc_type": self.doc_type,
            "method": self.method,
            "reason": self.reason,
        }
        if self.probability is not None:
            data["probability"] = self.probability
        if self.probabilities:
            data["probabilities"] = self.probabilities
        if self.noul_scores:
            data["noul_scores"] = self.noul_scores
        if self.raw_choice is not None:
            data["raw_choice"] = self.raw_choice
        if self.details:
            data.update(self.details)
        return data


@dataclass
class ExtractionTrace:
    """Audit trace captured during LLM extraction for full pipeline transparency."""

    system_prompt: str = ""
    user_prompt: str = ""
    raw_output: str = ""
    raw_parsed: dict[str, Any] | None = None


@dataclass
class ExtractionResult:
    """Structured extraction output together with its audit trace."""

    data: dict[str, Any]
    trace: ExtractionTrace = field(default_factory=ExtractionTrace)


@dataclass
class ProcessingRecord:
    """Per-file summary returned by the pipeline."""

    file: str
    doc_type: str | None = None
    output: str | None = None
    error: str | None = None
    traceback: str | None = None
    strategy: str | None = None
    strategy_reason: str | None = None
    n_pages: int | None = None
    seconds: float = 0.0
    classification: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["traceback"] is None:
            del data["traceback"]
        return data
