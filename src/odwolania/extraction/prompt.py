"""Generic prompt construction loaded from `config/prompts.json` and `config/types/*.json`."""
from __future__ import annotations

import copy
import json
from typing import Any

from ..config import DocumentType, PromptTemplates


def _strip_keys(node: Any, drop_keys: set[str], *, drop_x_extensions: bool = False) -> Any:
    if isinstance(node, dict):
        return {
            k: _strip_keys(v, drop_keys, drop_x_extensions=drop_x_extensions)
            for k, v in node.items()
            if k not in drop_keys and not (drop_x_extensions and k.startswith("x-"))
        }
    if isinstance(node, list):
        return [_strip_keys(item, drop_keys, drop_x_extensions=drop_x_extensions) for item in node]
    return node


class PromptBuilder:
    """Builds extraction prompts and relaxed decoding schemas from JSON configs (`prompts.json` + `types/*.json`)."""

    def __init__(self, template: str | None = None, *, prompts: PromptTemplates | None = None):
        if template is not None:
            self._template = template
        elif prompts is not None and prompts.extraction_system_prompt:
            self._template = prompts.extraction_system_prompt
        else:
            self._template = PromptTemplates.load().extraction_system_prompt

    def build_system_prompt(
        self,
        doc_type: DocumentType,
        *,
        notes_override: list[str] | None = None,
        template_override: str | None = None,
    ) -> str:
        notes = doc_type.extraction_notes if notes_override is None else notes_override
        lookups = doc_type.extraction_lookups
        clean_schema = _strip_keys(copy.deepcopy(doc_type.schema), set(), drop_x_extensions=True)
        tpl = template_override if template_override is not None else self._template
        return tpl.format(
            name=doc_type.name,
            description=doc_type.description,
            schema=json.dumps(clean_schema, ensure_ascii=False),
            notes=("\nExtraction notes:\n" + "\n".join(f"- {n}" for n in notes) + "\n") if notes else "",
            lookups=("\nLookup tables:\n" + json.dumps(lookups, ensure_ascii=False, indent=1) + "\n") if lookups else "",
        )

    @staticmethod
    def build_user_prompt(text: str) -> str:
        return f"Document text:\n<<<\n{text}\n>>>"

    @staticmethod
    def relax_schema_for_decoding(schema: dict[str, Any]) -> dict[str, Any]:
        """Strips strict `pattern`/`format` regexes and `x-*` hints from constrained-decoding grammar."""
        return _strip_keys(copy.deepcopy(schema), {"pattern", "format"}, drop_x_extensions=True)
