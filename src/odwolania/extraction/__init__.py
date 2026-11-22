"""Structured LLM extraction, prompt building, source grounding, and schema post-processing."""
from .client import LLMConfig, OpenAICompatibleClient
from .extractor import DocumentExtractor, ExtractionError
from .grounding import SourceTextIndex
from .postprocessor import ConsistencyRuleRegistry, NormalizerRegistry, SchemaPostprocessor
from .prompt import PromptBuilder

__all__ = [
    "ConsistencyRuleRegistry",
    "DocumentExtractor",
    "ExtractionError",
    "LLMConfig",
    "NormalizerRegistry",
    "OpenAICompatibleClient",
    "PromptBuilder",
    "SchemaPostprocessor",
    "SourceTextIndex",
]
