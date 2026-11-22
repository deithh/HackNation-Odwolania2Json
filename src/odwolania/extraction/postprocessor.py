"""100% schema- and config-driven JSON post-processor with extensible normalizer & rule registries.

Contains ZERO hardcoded document types or domain-specific field names:
- String normalization is driven by `x-normalizer` hints (registered in `NormalizerRegistry`)
  with automatic inference from standard JSON Schema `pattern` / `format` keywords.
- Code <-> label lookup synchronization is driven by `extraction.lookup_bindings`.
- Cross-field entity consistency is driven by `extraction.consistency_rules` (`ConsistencyRuleRegistry`).
"""
from __future__ import annotations

import copy
import re
from typing import Any, Callable

from ..config import DocumentType
from ..utils.text import is_placeholder_digits, parse_amount, parse_polish_date
from .grounding import BZP_FLEX_RE, TED_ALT_RE, TED_OJS_RE, SourceTextIndex

_NULL_STRINGS = {"", "null", "none", "brak", "brak danych", "n/a", "nd", "nie dotyczy", "-", "–", "—"}
_TITLE_PREFIX_RE = re.compile(
    r"^(?:(?:prof\.|dr\s+hab\.|dr|mgr|inż\.|r\.\s*pr\.|radca\s+prawny|adw\.|adwokat|"
    r"apl\.\s*radc\.|apl\.\s*adw\.|mec\.|mecenas|pan|pani)\s+)+",
    flags=re.I,
)
_EMAIL_ONLY_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_OCR_DIACRITIC_FIX = str.maketrans({"Ž": "Ż", "ž": "ż", "Š": "Ś", "š": "ś", "Č": "Ć", "č": "ć"})

StringNormalizerFn = Callable[[str, dict[str, Any], SourceTextIndex | None], str | None]
ConsistencyRuleFn = Callable[[dict[str, Any], dict[str, Any]], None]


class NormalizerRegistry:
    """Extensible registry of named string normalizers referenced via `x-normalizer` or schema patterns."""

    _registry: dict[str, StringNormalizerFn] = {}

    @classmethod
    def register(cls, name: str, fn: StringNormalizerFn) -> None:
        cls._registry[name.lower()] = fn

    @classmethod
    def get(cls, name: str) -> StringNormalizerFn | None:
        return cls._registry.get(name.lower())

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._registry)


class ConsistencyRuleRegistry:
    """Extensible registry of declarative post-processing consistency rules."""

    _registry: dict[str, ConsistencyRuleFn] = {}

    @classmethod
    def register(cls, name: str, fn: ConsistencyRuleFn) -> None:
        cls._registry[name.lower()] = fn

    @classmethod
    def get(cls, name: str) -> ConsistencyRuleFn | None:
        return cls._registry.get(name.lower())

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._registry)


# ---------------------------------------------------------------------------
# Built-in generic string normalizers (domain-agnostic)
# ---------------------------------------------------------------------------

def _norm_krs(s: str, _node: dict[str, Any], _src: SourceTextIndex | None) -> str | None:
    digits = re.sub(r"\D", "", s)
    if 1 <= len(digits) <= 9:
        digits = digits.zfill(10)
    return digits


def _norm_digits(s: str, _node: dict[str, Any], _src: SourceTextIndex | None) -> str | None:
    return re.sub(r"\D", "", s)


def _norm_iban(s: str, _node: dict[str, Any], _src: SourceTextIndex | None) -> str | None:
    compact = re.sub(r"[\s\-–]", "", s.strip().upper())
    m_cc = re.match(r"^([A-Z]{2})([A-Z0-9]+)$", compact)
    if m_cc:
        cc, rest = m_cc.group(1), m_cc.group(2)
        if cc == "PL":
            return f"PL{re.sub(r'\D', '', rest)}"
        return f"{cc}{re.sub(r'[^A-Z0-9]', '', rest)}"
    return re.sub(r"\D", "", compact)


def _norm_bzp(s: str, _node: dict[str, Any], _src: SourceTextIndex | None) -> str | None:
    match = BZP_FLEX_RE.search(s)
    if match:
        return f"{match.group(1)}/BZP{match.group(2)}" + (f"/{match.group(3)}" if match.group(3) else "")
    return re.sub(r"\s+", "", s).upper()


def _norm_ted(s: str, node: dict[str, Any], src: SourceTextIndex | None) -> str | None:
    m_alt = TED_ALT_RE.search(s)
    m_ojs = TED_OJS_RE.search(s) if not m_alt else None
    if m_alt:
        return f"{m_alt.group(3)}-{m_alt.group(1)}"
    if m_ojs:
        return f"{m_ojs.group(2)}-{m_ojs.group(1)}"
    compact = re.sub(r"\s+", "", s)
    pattern = str(node.get("pattern", r"^\d{6}-\d{4}$"))
    if not re.fullmatch(pattern, compact) and src is not None:
        resolved = src.resolve_ted(compact)
        if resolved is not None:
            return resolved
    return compact


def _norm_email(s: str, _node: dict[str, Any], _src: SourceTextIndex | None) -> str | None:
    return re.sub(r"^mailto:", "", s, flags=re.I).strip().rstrip(".,;:")


def _norm_person_name(s: str, _node: dict[str, Any], _src: SourceTextIndex | None) -> str | None:
    return _TITLE_PREFIX_RE.sub("", s.translate(_OCR_DIACRITIC_FIX)).strip(" ,;-")


def _norm_entity_name(s: str, node: dict[str, Any], _src: SourceTextIndex | None) -> str | None:
    if SchemaPostprocessor._allows_null(node) and _EMAIL_ONLY_RE.fullmatch(s):
        return None
    return s.translate(_OCR_DIACRITIC_FIX)


def _norm_unquoted_text(s: str, _node: dict[str, Any], _src: SourceTextIndex | None) -> str | None:
    cleaned = s.translate(_OCR_DIACRITIC_FIX)
    if (cleaned.startswith("„") and cleaned.endswith("”")) or (cleaned.startswith('"') and cleaned.endswith('"')):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _norm_plain_text(s: str, _node: dict[str, Any], _src: SourceTextIndex | None) -> str | None:
    return s.translate(_OCR_DIACRITIC_FIX)


NormalizerRegistry.register("krs", _norm_krs)
NormalizerRegistry.register("digits", _norm_digits)
NormalizerRegistry.register("nip", _norm_digits)
NormalizerRegistry.register("regon", _norm_digits)
NormalizerRegistry.register("pesel", _norm_digits)
NormalizerRegistry.register("nrb", _norm_digits)
NormalizerRegistry.register("iban", _norm_iban)
NormalizerRegistry.register("iban_pl", _norm_iban)
NormalizerRegistry.register("bzp", _norm_bzp)
NormalizerRegistry.register("ted", _norm_ted)
NormalizerRegistry.register("email", _norm_email)
NormalizerRegistry.register("person_name", _norm_person_name)
NormalizerRegistry.register("entity_name", _norm_entity_name)
NormalizerRegistry.register("unquoted_text", _norm_unquoted_text)
NormalizerRegistry.register("plain_text", _norm_plain_text)


# ---------------------------------------------------------------------------
# Built-in declarative consistency rules (domain-agnostic)
# ---------------------------------------------------------------------------

def _rule_clear_object_when(data: dict[str, Any], spec: dict[str, Any]) -> None:
    """For each dict in `list_field`, if `item[when_field] == equals`, nullifies all keys in `item[target_field]`."""
    items = data.get(spec["list_field"])
    if not isinstance(items, list):
        return
    when_field = spec["when_field"]
    expected = spec["equals"]
    target_field = spec["target_field"]
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get(when_field) == expected:
            target = item.get(target_field)
            if isinstance(target, dict):
                for k in list(target.keys()):
                    target[k] = None


def _rule_fallback_from_single_item(data: dict[str, Any], spec: dict[str, Any]) -> None:
    """Fills an empty `target_field` on `item[object_field]` from a unique matching entry in `item[source_list]`."""
    items = data.get(spec["list_field"])
    if not isinstance(items, list):
        return
    obj_field = spec["object_field"]
    when_field = spec["when_field"]
    expected = spec["equals"]
    target_field = spec["target_field"]
    source_list = spec["source_list"]
    source_field = spec["source_field"]
    must_contain_in = spec.get("must_contain_in")

    for item in items:
        if not isinstance(item, dict):
            continue
        obj = item.get(obj_field)
        sources = item.get(source_list)
        if not isinstance(obj, dict) or obj.get(when_field) != expected or obj.get(target_field):
            continue
        if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], dict):
            continue
        candidate = sources[0].get(source_field)
        if not isinstance(candidate, str) or not candidate:
            continue
        if must_contain_in:
            container = obj.get(must_contain_in)
            if not isinstance(container, str) or candidate.lower() not in container.lower():
                continue
        obj[target_field] = candidate


ConsistencyRuleRegistry.register("clear_object_when", _rule_clear_object_when)
ConsistencyRuleRegistry.register("fallback_from_single_item", _rule_fallback_from_single_item)


# ---------------------------------------------------------------------------
# Generic schema-driven post-processor
# ---------------------------------------------------------------------------

class SchemaPostprocessor:
    """Normalizes, validates, and grounds extracted JSON according to a `DocumentType`'s schema and JSON config."""

    def __init__(self, postprocessing_cfg: dict[str, Any] | None = None):
        cfg = postprocessing_cfg or {}
        raw_nulls = cfg.get("null_strings")
        self._null_strings: set[str] = (
            {str(x).lower() for x in raw_nulls} if isinstance(raw_nulls, list) else set(_NULL_STRINGS)
        )
        self._ted_year_min = int(cfg.get("ted_year_min", 2000))
        self._ted_year_max = int(cfg.get("ted_year_max", 2035))
        self._nrb_max_dist = int(cfg.get("nrb_max_edit_distance", 4))
        self._nrb_reject_dist = int(cfg.get("nrb_reject_distance", 6))

    def process(self, data: dict[str, Any], doc_type: DocumentType, *, source_text: str | None = None) -> dict[str, Any]:
        schema = doc_type.schema
        if not schema or not isinstance(data, dict):
            return data
        src = SourceTextIndex(source_text) if source_text else None
        cleaned = self._walk(copy.deepcopy(data), schema, schema, src=src)
        if isinstance(cleaned, dict):
            self._apply_lookup_bindings(cleaned, doc_type.extraction_lookups, doc_type.lookup_bindings)
            self._apply_consistency_rules(cleaned, doc_type.consistency_rules)
        return cleaned

    @staticmethod
    def _resolve_ref(node: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
        ref = node.get("$ref")
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return node
        target: Any = root
        for part in ref[2:].split("/"):
            target = target[part]
        merged = dict(target)
        for k, v in node.items():
            if k != "$ref":
                merged[k] = v
        return merged

    @staticmethod
    def _allows_null(node: dict[str, Any]) -> bool:
        t = node.get("type")
        if t == "null" or (isinstance(t, list) and "null" in t):
            return True
        enum = node.get("enum")
        return isinstance(enum, list) and None in enum

    @staticmethod
    def _allowed_types(node: dict[str, Any]) -> set[str]:
        t = node.get("type")
        if isinstance(t, str):
            return {t}
        if isinstance(t, list):
            return set(t)
        return set()

    @staticmethod
    def _infer_normalizer(node: dict[str, Any]) -> str | None:
        explicit = node.get("x-normalizer")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip().lower()
        pattern = str(node.get("pattern", ""))
        if pattern in (r"^\d{10}$", r"^(\d{9}|\d{14})$", r"^\d{11}$", r"^\d{26}$"):
            return "digits"
        if pattern in (r"^(?:PL)?\d{26}$", r"^(?:[A-Z]{2}[A-Z0-9]{13,32}|\d{26})$"):
            return "iban"
        if "/BZP" in pattern:
            return "bzp"
        if pattern == r"^\d{6}-\d{4}$":
            return "ted"
        if "@" in pattern:
            return "email"
        if not pattern:
            if "bez cudzysłow" in str(node.get("description") or "").lower():
                return "unquoted_text"
            return "plain_text"
        return None

    def _normalize_string(
        self,
        val: str,
        node: dict[str, Any],
        src: SourceTextIndex | None,
    ) -> str | None:
        s: str | None = re.sub(r"\s+", " ", val).strip()
        allows_null = self._allows_null(node)
        if allows_null and s.lower() in self._null_strings:
            return None

        pattern = str(node.get("pattern", ""))
        fmt = str(node.get("format", ""))

        if fmt == "date" or pattern == r"^\d{4}-\d{2}-\d{2}$":
            parsed = parse_polish_date(s)
            if parsed is not None:
                return parsed
            return None if allows_null else s

        norm_name = self._infer_normalizer(node)
        if norm_name:
            fn = NormalizerRegistry.get(norm_name)
            if fn is not None:
                s = fn(s, node, src)
                if s is None:
                    return None

        if pattern and not re.fullmatch(pattern, s):
            return None if allows_null else s

        if allows_null:
            s = self._ground_identifier(s, pattern, norm_name, src)
            if s is None:
                return None

        enum = node.get("enum")
        if isinstance(enum, list) and s not in enum:
            return None if None in enum else s
        return s

    def _ground_identifier(
        self,
        value: str,
        pattern: str,
        norm_name: str | None,
        src: SourceTextIndex | None,
    ) -> str | None:
        if pattern in (r"^\d{10}$", r"^(\d{9}|\d{14})$", r"^\d{11}$"):
            if is_placeholder_digits(value):
                return None
            if src is not None and not src.has_digits(value, allow_krs_unpadded=(norm_name == "krs")):
                return None
        elif pattern in (r"^\d{26}$", r"^(?:PL)?\d{26}$", r"^(?:[A-Z]{2}[A-Z0-9]{13,32}|\d{26})$"):
            digits = re.sub(r"\D", "", value)
            if is_placeholder_digits(digits):
                return None
            if src is not None:
                return src.ground_nrb(
                    value,
                    allow_pl=("PL" in pattern or "[A-Z]{2}" in pattern),
                    max_dist=self._nrb_max_dist,
                    reject_dist=self._nrb_reject_dist,
                )
        elif "/BZP" in pattern:
            match = BZP_FLEX_RE.fullmatch(value)
            if not match or is_placeholder_digits(match.group(2)):
                return None
            if src is not None and not src.has_bzp(value):
                return None
        elif pattern == r"^\d{6}-\d{4}$":
            match = re.fullmatch(r"(\d{6})-(\d{4})", value)
            if (
                not match
                or is_placeholder_digits(match.group(1))
                or not (self._ted_year_min <= int(match.group(2)) <= self._ted_year_max)
            ):
                return None
            if src is not None and not src.has_ted(
                value, year_min=self._ted_year_min, year_max=self._ted_year_max
            ):
                return None
        return value

    def _walk(
        self,
        val: Any,
        node: dict[str, Any],
        root: dict[str, Any],
        *,
        src: SourceTextIndex | None,
    ) -> Any:
        node = self._resolve_ref(node, root)
        types = self._allowed_types(node)

        if val is None:
            return None

        if "object" in types and isinstance(val, dict):
            props = node.get("properties") or {}
            out: dict[str, Any] = {}
            for k, v in val.items():
                if k in props:
                    out[k] = self._walk(v, props[k], root, src=src)
                elif node.get("additionalProperties", True) is not False:
                    out[k] = v
            for req in node.get("required") or []:
                if req not in out:
                    prop_schema = self._resolve_ref(props.get(req, {}), root)
                    if self._allows_null(prop_schema):
                        out[req] = None
            return out

        if "array" in types and isinstance(val, list):
            item_schema = node.get("items") or {}
            return [self._walk(item, item_schema, root, src=src) for item in val]

        if "integer" in types and not isinstance(val, bool):
            return self._normalize_integer(val, node)

        if "number" in types and not isinstance(val, bool):
            num_f = parse_amount(val)
            if num_f is None:
                return None if self._allows_null(node) else val
            if "minimum" in node and num_f < node["minimum"]:
                return None if self._allows_null(node) else num_f
            return num_f

        if "string" in types:
            if isinstance(val, int) and not isinstance(val, bool):
                val = str(val)
            if isinstance(val, str):
                return self._normalize_string(val, node, src)

        return val

    def _normalize_integer(self, val: Any, node: dict[str, Any]) -> Any:
        num: int | None
        if isinstance(val, int):
            num = val
        elif isinstance(val, float) and val.is_integer():
            num = int(val)
        elif isinstance(val, str):
            s = val.strip()
            if self._allows_null(node) and s.lower() in self._null_strings:
                return None
            num = int(s) if re.fullmatch(r"-?\d+", s) else None
        else:
            num = None

        if num is None:
            return None if self._allows_null(node) else val
        if "minimum" in node and num < node["minimum"]:
            return None if self._allows_null(node) else num
        if "maximum" in node and num > node["maximum"]:
            return None if self._allows_null(node) else num
        enum = node.get("enum")
        if isinstance(enum, list) and num not in enum:
            return None if None in enum else num
        return num

    @staticmethod
    def _apply_lookup_bindings(
        data: dict[str, Any],
        lookups: dict[str, Any],
        bindings: list[dict[str, Any]],
    ) -> None:
        """Synchronizes any configured `(id_field, label_field)` pair 1:1 with its lookup table."""
        for binding in bindings:
            table = lookups.get(binding.get("table", ""))
            id_field = binding.get("id_field")
            label_field = binding.get("label_field")
            if not isinstance(table, dict) or not id_field or not label_field:
                continue
            stack: list[Any] = [data]
            while stack:
                obj = stack.pop()
                if isinstance(obj, dict):
                    if id_field in obj and label_field in obj:
                        pid = obj.get(id_field)
                        if pid is not None and str(pid) in table and table[str(pid)] is not None:
                            obj[label_field] = table[str(pid)]
                        elif pid is None and isinstance(obj.get(label_field), str):
                            norm = re.sub(r"[\s/]+", " ", obj[label_field].lower()).replace(" i ", " ").strip()
                            for k, v in table.items():
                                if isinstance(v, str):
                                    v_norm = re.sub(r"[\s/]+", " ", v.lower()).replace(" i ", " ").strip()
                                    if norm == v_norm:
                                        obj[id_field] = int(k) if k.isdigit() else k
                                        obj[label_field] = v
                                        break
                    for value in obj.values():
                        if isinstance(value, (dict, list)):
                            stack.append(value)
                elif isinstance(obj, list):
                    stack.extend(x for x in obj if isinstance(x, (dict, list)))

    @staticmethod
    def _apply_consistency_rules(data: dict[str, Any], rules: list[dict[str, Any]]) -> None:
        for spec in rules:
            rule_name = str(spec.get("rule") or "")
            fn = ConsistencyRuleRegistry.get(rule_name)
            if fn is not None:
                fn(data, spec)
