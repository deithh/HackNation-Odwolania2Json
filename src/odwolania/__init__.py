"""Odwołania2JSON — config-driven PDF to structured JSON pipeline for KIO public-procurement appeals."""
from .config import DocumentType, DocumentTypeRegistry, PipelineConfig, PromptTemplates
from .evaluation import EvalBucket, EvalCounts, F1Evaluator
from .models import (
    ClassificationResult,
    DocumentText,
    ExtractionResult,
    ExtractionTrace,
    PageText,
    ProcessingRecord,
)
from .optimization import GepaTypeOptimizer, OptimizationResult
from .pipeline import DocumentPipeline
from .type_tools import TypeConfigScaffolder, TypeConfigValidator, ValidationIssue, ValidationReport

__all__ = [
    "ClassificationResult",
    "DocumentPipeline",
    "DocumentText",
    "DocumentType",
    "DocumentTypeRegistry",
    "EvalBucket",
    "EvalCounts",
    "ExtractionResult",
    "ExtractionTrace",
    "F1Evaluator",
    "GepaTypeOptimizer",
    "OptimizationResult",
    "PageText",
    "PipelineConfig",
    "ProcessingRecord",
    "PromptTemplates",
    "TypeConfigScaffolder",
    "TypeConfigValidator",
    "ValidationIssue",
    "ValidationReport",
]
