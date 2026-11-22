"""Zero-code document-type management: config validation, scaffolding, and sample-based schema inference."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .classification.rules import FEATURE_ALIASES, FeatureExtractor
from .config import DEFAULT_DOC_TYPES_DIR, PipelineConfig, resolve_path
from .extraction.client import LLMConfig, OpenAICompatibleClient
from .extraction.postprocessor import ConsistencyRuleRegistry, NormalizerRegistry
from .ocr import DocumentReader
from .utils.pages import resolve_pages
from .utils.text import parse_polish_date, trim_document_text

_TYPE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,48}$")
_VALID_OPS = {">=", "<=", "==", ">", "<"}
_VALID_FEATURES = set(FEATURE_ALIASES.keys()) | set(FEATURE_ALIASES.values()) | set(FeatureExtractor.extract("").keys())

_CONSISTENCY_REQUIRED_KEYS: dict[str, set[str]] = {
    "clear_object_when": {"list_field", "when_field", "equals", "target_field"},
    "fallback_from_single_item": {
        "list_field",
        "object_field",
        "when_field",
        "equals",
        "target_field",
        "source_list",
        "source_field",
    },
}


@dataclass
class ValidationIssue:
    file: str
    severity: str       # error | warning
    message: str


@dataclass
class ValidationReport:
    checked_files: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors


class TypeConfigValidator:
    """Validates `config/types/*.json` files against all structural, regex, lookup, and rule contracts."""

    def validate_dict(self, data: Any, *, source_name: str = "<memory>") -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if not isinstance(data, dict):
            return [ValidationIssue(source_name, "error", "root JSON value must be an object")]

        for req in ("type_id", "nazwa", "opis", "classification", "extraction", "schema"):
            if req not in data:
                issues.append(ValidationIssue(source_name, "error", f"missing required top-level key {req!r}"))

        tid = data.get("type_id")
        if not isinstance(tid, str) or not _TYPE_ID_RE.fullmatch(tid):
            issues.append(
                ValidationIssue(
                    source_name,
                    "error",
                    f"invalid type_id {tid!r} (expected lowercase snake_case matching ^[a-z][a-z0-9_]{{1,48}}$)",
                )
            )

        self._validate_classification(data.get("classification"), source_name, issues)
        self._validate_extraction(data.get("extraction"), source_name, issues)
        self._validate_schema(data.get("schema"), source_name, issues)
        return issues

    def validate_file(self, path: str | Path) -> ValidationReport:
        p = Path(path)
        report = ValidationReport(checked_files=1)
        try:
            raw = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            report.issues.append(ValidationIssue(p.name, "error", f"failed to parse JSON: {exc}"))
            return report
        report.issues.extend(self.validate_dict(raw, source_name=p.name))
        return report

    def validate_dir(self, types_dir: str | Path | None = None) -> ValidationReport:
        folder = resolve_path(types_dir) if types_dir else DEFAULT_DOC_TYPES_DIR
        report = ValidationReport()
        if not folder.is_dir():
            report.issues.append(ValidationIssue(str(folder), "error", f"directory not found: {folder}"))
            return report

        seen_ids: dict[str, str] = {}
        files = sorted(folder.glob("*.json"))
        if not files:
            report.issues.append(ValidationIssue(str(folder), "warning", "no *.json type configs found"))
            return report

        for path in files:
            report.checked_files += 1
            try:
                raw = json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                report.issues.append(ValidationIssue(path.name, "error", f"failed to parse JSON: {exc}"))
                continue
            report.issues.extend(self.validate_dict(raw, source_name=path.name))
            tid = raw.get("type_id") if isinstance(raw, dict) else None
            if isinstance(tid, str):
                if tid in seen_ids:
                    report.issues.append(
                        ValidationIssue(path.name, "error", f"duplicate type_id {tid!r} (already in {seen_ids[tid]})")
                    )
                else:
                    seen_ids[tid] = path.name
        return report

    @staticmethod
    def _validate_classification(cls_cfg: Any, src: str, issues: list[ValidationIssue]) -> None:
        if not isinstance(cls_cfg, dict):
            issues.append(ValidationIssue(src, "error", "'classification' must be an object"))
            return
        prio = cls_cfg.get("priority")
        if not isinstance(prio, int) or isinstance(prio, bool) or prio < 1:
            issues.append(ValidationIssue(src, "error", "'classification.priority' must be a positive integer"))
        rule = cls_cfg.get("rule")
        if rule is not None:
            if not isinstance(rule, list):
                issues.append(ValidationIssue(src, "error", "'classification.rule' must be a list"))
            else:
                for idx, cond in enumerate(rule):
                    if not isinstance(cond, list) or len(cond) != 3:
                        issues.append(ValidationIssue(src, "error", f"classification.rule[{idx}] must be [feature, op, value]"))
                        continue
                    feat, op, val = cond
                    if feat not in _VALID_FEATURES:
                        issues.append(ValidationIssue(src, "error", f"unknown rule feature {feat!r}"))
                    if op not in _VALID_OPS:
                        issues.append(ValidationIssue(src, "error", f"invalid rule operator {op!r}"))
                    if not isinstance(val, (int, float)) or isinstance(val, bool):
                        issues.append(ValidationIssue(src, "error", f"rule threshold for {feat!r} must be numeric"))

    @staticmethod
    def _validate_extraction(ext_cfg: Any, src: str, issues: list[ValidationIssue]) -> None:
        if not isinstance(ext_cfg, dict):
            issues.append(ValidationIssue(src, "error", "'extraction' must be an object"))
            return
        if "pages" in ext_cfg and ext_cfg["pages"] is not None:
            try:
                resolve_pages(ext_cfg["pages"], 10)
            except Exception as exc:
                issues.append(ValidationIssue(src, "error", f"invalid extraction.pages: {exc}"))
        uwagi = ext_cfg.get("uwagi")
        if uwagi is not None and (not isinstance(uwagi, list) or not all(isinstance(u, str) for u in uwagi)):
            issues.append(ValidationIssue(src, "error", "'extraction.uwagi' must be a list of strings"))

        reserved = {"pages", "uwagi", "lookup_bindings", "consistency_rules"}
        lookups = {k: v for k, v in ext_cfg.items() if k not in reserved}

        for idx, binding in enumerate(ext_cfg.get("lookup_bindings") or []):
            if not isinstance(binding, dict):
                issues.append(ValidationIssue(src, "error", f"lookup_bindings[{idx}] must be an object"))
                continue
            table = binding.get("table")
            if table not in lookups or not isinstance(lookups[table], dict):
                issues.append(ValidationIssue(src, "error", f"lookup_bindings[{idx}] references unknown table {table!r}"))
            for req in ("id_field", "label_field"):
                if not isinstance(binding.get(req), str) or not binding[req]:
                    issues.append(ValidationIssue(src, "error", f"lookup_bindings[{idx}] missing non-empty {req!r}"))

        for idx, rule_spec in enumerate(ext_cfg.get("consistency_rules") or []):
            if not isinstance(rule_spec, dict):
                issues.append(ValidationIssue(src, "error", f"consistency_rules[{idx}] must be an object"))
                continue
            rname = str(rule_spec.get("rule") or "")
            if ConsistencyRuleRegistry.get(rname) is None:
                issues.append(
                    ValidationIssue(
                        src,
                        "error",
                        f"consistency_rules[{idx}] uses unregistered rule {rname!r} "
                        f"(registered: {', '.join(ConsistencyRuleRegistry.names())})",
                    )
                )
            else:
                missing = _CONSISTENCY_REQUIRED_KEYS.get(rname, set()) - set(rule_spec.keys())
                if missing:
                    issues.append(
                        ValidationIssue(src, "error", f"consistency_rules[{idx}] ({rname}) missing keys: {sorted(missing)}")
                    )

    def _validate_schema(self, schema: Any, src: str, issues: list[ValidationIssue]) -> None:
        if not isinstance(schema, dict):
            issues.append(ValidationIssue(src, "error", "'schema' must be a JSON Schema object"))
            return
        if schema.get("type") != "object" or not isinstance(schema.get("properties"), dict):
            issues.append(ValidationIssue(src, "error", "'schema' must have type='object' and a 'properties' map"))
            return
        self._walk_schema(schema, schema, "$", src, issues)

    def _walk_schema(
        self,
        node: Any,
        root: dict[str, Any],
        path: str,
        src: str,
        issues: list[ValidationIssue],
    ) -> None:
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if isinstance(ref, str):
            if not ref.startswith("#/"):
                issues.append(ValidationIssue(src, "error", f"{path}: unsupported external $ref {ref!r}"))
            else:
                target: Any = root
                try:
                    for part in ref[2:].split("/"):
                        target = target[part]
                except Exception:
                    issues.append(ValidationIssue(src, "error", f"{path}: broken $ref {ref!r}"))

        pattern = node.get("pattern")
        if isinstance(pattern, str):
            try:
                re.compile(pattern)
            except re.error as exc:
                issues.append(ValidationIssue(src, "error", f"{path}: invalid regex pattern {pattern!r} ({exc})"))

        norm = node.get("x-normalizer")
        if isinstance(norm, str) and NormalizerRegistry.get(norm) is None:
            issues.append(
                ValidationIssue(
                    src,
                    "error",
                    f"{path}: unknown x-normalizer {norm!r} (registered: {', '.join(NormalizerRegistry.names())})",
                )
            )

        props = node.get("properties")
        if isinstance(props, dict):
            for k, sub in props.items():
                self._walk_schema(sub, root, f"{path}.{k}", src, issues)
        if "items" in node:
            self._walk_schema(node["items"], root, f"{path}[]", src, issues)
        defs = node.get("$defs")
        if isinstance(defs, dict):
            for k, sub in defs.items():
                self._walk_schema(sub, root, f"{path}.$defs.{k}", src, issues)


class TypeConfigScaffolder:
    """Creates new `config/types/<type_id>.json` files from templates, sample JSONs, or sample PDFs."""

    def __init__(self, validator: TypeConfigValidator | None = None):
        self._validator = validator or TypeConfigValidator()

    @staticmethod
    def build_template(
        type_id: str,
        *,
        name: str | None = None,
        description: str = "",
        priority: int = 10,
        pages: str = "1-3,last",
    ) -> dict[str, Any]:
        display_name = name or type_id.replace("_", " ").capitalize()
        desc = description or f"Dokument typu: {display_name}."
        root_key = "".join(part.capitalize() for part in type_id.split("_"))
        return {
            "type_id": type_id,
            "nazwa": display_name,
            "opis": desc,
            "classification": {
                "priority": priority,
                "include": desc,
                "exclude": f"NIE jest innym typem dokumentu, który jedynie wspomina o '{display_name}'.",
            },
            "extraction": {
                "pages": pages,
                "uwagi": [
                    "Uzupełnij tutaj ewentualne uwagi domenowe dla tego typu dokumentu.",
                ],
            },
            "schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": type_id,
                "title": display_name,
                "type": "object",
                "additionalProperties": False,
                "required": [root_key],
                "properties": {
                    root_key: {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["data", "miejscowosc", "podmiot"],
                        "properties": {
                            "data": {
                                "type": ["string", "null"],
                                "format": "date",
                                "pattern": r"^\d{4}-\d{2}-\d{2}$",
                                "description": "Data dokumentu z nagłówka (ISO YYYY-MM-DD).",
                            },
                            "miejscowosc": {
                                "type": ["string", "null"],
                                "description": "Miejscowość z linii nagłówkowej dokumentu.",
                            },
                            "podmiot": {
                                "type": ["string", "null"],
                                "x-normalizer": "entity_name",
                                "description": "Główny podmiot występujący w dokumencie.",
                            },
                        },
                    }
                },
            },
        }

    @classmethod
    def infer_from_json_sample(
        cls,
        sample_data: dict[str, Any],
        type_id: str,
        *,
        name: str | None = None,
        description: str = "",
        priority: int = 10,
        pages: str = "1-3,last",
    ) -> dict[str, Any]:
        """Deterministically infers a complete `config/types/<type_id>.json` from an example output JSON."""
        if not isinstance(sample_data, dict) or not sample_data:
            raise ValueError("sample JSON must be a non-empty JSON object")
        display_name = name or type_id.replace("_", " ").capitalize()
        desc = description or f"Dokument typu: {display_name}."
        schema = cls._infer_schema_node(sample_data, key=None)
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": type_id,
            "title": display_name,
            **schema,
        }
        return {
            "type_id": type_id,
            "nazwa": display_name,
            "opis": desc,
            "classification": {
                "priority": priority,
                "include": desc,
                "exclude": f"NIE jest innym typem dokumentu, który jedynie wspomina o '{display_name}'.",
            },
            "extraction": {
                "pages": pages,
                "uwagi": [],
            },
            "schema": schema,
        }

    @classmethod
    def _infer_schema_node(cls, value: Any, *, key: str | None) -> dict[str, Any]:
        if isinstance(value, dict):
            props = {k: cls._infer_schema_node(v, key=k) for k, v in value.items()}
            return {
                "type": "object",
                "additionalProperties": False,
                "required": list(props.keys()),
                "properties": props,
            }
        if isinstance(value, list):
            if value and all(isinstance(x, dict) for x in value):
                merged: dict[str, Any] = {}
                for item in value:
                    for k, v in item.items():
                        if k not in merged or merged[k] is None:
                            merged[k] = v
                item_schema = cls._infer_schema_node(merged, key=key)
            elif value:
                item_schema = cls._infer_schema_node(value[0], key=key)
            else:
                item_schema = {"type": "string"}
            return {"type": "array", "items": item_schema}
        if isinstance(value, bool):
            return {"type": ["boolean", "null"], "description": cls._default_desc(key)}
        if isinstance(value, int):
            return {"type": ["integer", "null"], "description": cls._default_desc(key)}
        if isinstance(value, float):
            return {"type": ["number", "null"], "description": cls._default_desc(key)}
        if isinstance(value, str):
            return cls._infer_string_leaf(value, key=key)
        return {"type": ["string", "null"], "description": cls._default_desc(key)}

    @classmethod
    def _infer_string_leaf(cls, value: str, *, key: str | None) -> dict[str, Any]:
        node: dict[str, Any] = {"type": ["string", "null"], "description": cls._default_desc(key)}
        low_key = (key or "").lower()
        s = value.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) or (parse_polish_date(s) and "data" in low_key):
            node["format"] = "date"
            node["pattern"] = r"^\d{4}-\d{2}-\d{2}$"
        elif re.fullmatch(r"(?:PL)?\d{26}", s, flags=re.I):
            node["pattern"] = r"^(?:PL)?\d{26}$"
        elif re.fullmatch(r"\d{4}/BZP\d{8}(/\d{2})?", s, flags=re.I):
            node["pattern"] = r"^\d{4}/BZP\d{8}(/\d{2})?$"
        elif re.fullmatch(r"\d{6}-\d{4}", s):
            node["pattern"] = r"^\d{6}-\d{4}$"
        elif re.fullmatch(r"\d{11}", s) and "pesel" in low_key:
            node["pattern"] = r"^\d{11}$"
        elif re.fullmatch(r"\d{10}", s) and ("krs" in low_key or "nip" in low_key):
            node["pattern"] = r"^\d{10}$"
            if "krs" in low_key:
                node["x-normalizer"] = "krs"
        elif re.fullmatch(r"\d{9}|\d{14}", s) and "regon" in low_key:
            node["pattern"] = r"^(\d{9}|\d{14})$"
        elif re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", s):
            node["pattern"] = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
        elif "imie" in low_key or "nazwisko" in low_key or "reprezentant" in low_key:
            node["x-normalizer"] = "person_name"
        elif "nazwa" in low_key:
            node["x-normalizer"] = "entity_name"
        return node

    @staticmethod
    def _default_desc(key: str | None) -> str:
        return f"Wartość pola '{key}'." if key else "Wartość pola."

    def infer_from_pdf_sample(
        self,
        pdf_path: str | Path,
        type_id: str,
        *,
        name: str | None = None,
        priority: int = 10,
        pipeline_config: PipelineConfig | None = None,
        llm_config: LLMConfig | None = None,
    ) -> dict[str, Any]:
        """Reads a sample PDF (native text or local RapidOCR) and asks the LLM to draft a type config."""
        cfg = pipeline_config or PipelineConfig.load()
        reader = DocumentReader(cfg)
        doc = reader.read(pdf_path, pages="1-3,last")
        if doc.strategy == "error":
            raise RuntimeError(f"cannot read sample PDF: {doc.strategy_reason}")
        text = trim_document_text(doc.text_for(), int(cfg.classification.get("max_chars", 12000)))
        client = OpenAICompatibleClient(llm_config or LLMConfig.from_pipeline_config(cfg))
        user_prompt = (
            f"type_id: {type_id}\n"
            f"sugerowana nazwa: {name or type_id.replace('_', ' ').capitalize()}\n"
            f"priority: {priority}\n\n"
            f"Treść przykładowego dokumentu:\n<<<\n{text}\n>>>"
        )
        raw = client.chat(
            user_prompt,
            system=cfg.prompts.pdf_infer_system_prompt,
            temperature=float(cfg.extraction.get("temperature", 0.0)),
            max_tokens=int(cfg.extraction.get("max_tokens", 3000)),
            json_mode=True,
        )
        draft = json.loads(raw)
        if not isinstance(draft, dict):
            raise RuntimeError("LLM did not return a JSON object for the inferred type config")
        draft["type_id"] = type_id
        draft.setdefault("classification", {})["priority"] = priority
        return draft

    def save(
        self,
        config_dict: dict[str, Any],
        *,
        types_dir: str | Path | None = None,
        overwrite: bool = False,
    ) -> tuple[Path, ValidationReport]:
        folder = resolve_path(types_dir) if types_dir else DEFAULT_DOC_TYPES_DIR
        folder.mkdir(parents=True, exist_ok=True)
        tid = config_dict["type_id"]
        target = folder / f"{tid}.json"
        if target.exists() and not overwrite:
            raise FileExistsError(f"type config already exists: {target} (pass --overwrite to replace)")
        issues = self._validator.validate_dict(config_dict, source_name=target.name)
        report = ValidationReport(checked_files=1, issues=issues)
        target.write_text(json.dumps(config_dict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target, report
