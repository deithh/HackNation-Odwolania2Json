"""Deterministic format-feature classifier (zero-LLM fallback driven by `config/types/*.json`)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from ..config import DocumentTypeRegistry
from ..models import ClassificationResult
from ..utils.text import fix_ocr_digits, nrb_checksum_ok
from .base import OTHER, DocumentClassifier

_NB = r"(?<!\d)(?<![\-/.,]\d)"
_NA = r"(?!\d)(?![\-/.,]\d)"
_UP, _LO = "A-ZĄĆĘŁŃÓŚŹŻ", "a-ząćęłńóśźż"

NRB_RE = re.compile(r"(?<![\d])(?:[A-Z]{2} ?)?\d{2}(?: ?\d{4}){6}(?!\d)")
AMOUNT_RE = re.compile(r"(?<![\d.,])(?:\d{1,3}(?:[ .]\d{3})+|\d+)[,.]\d{2}(?![\d.,])")
DATE_RE = re.compile(r"(?<!\d)(?:\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2} [^\W\d_]{3,} \d{4})(?!\d)")
ID_9_10_RE = re.compile(_NB + r"(?:\d{10}|\d{9})" + _NA)
ID_11_RE = re.compile(_NB + r"\d{11}" + _NA)
EMAIL_RE = re.compile(r"[\w.\-]+@[\w\-]+\.[\w.\-]+")
POSTAL_RE = re.compile(r"(?<![\d\-])\d{2}-\d{3}(?![\d\-])")
ART_RE = re.compile(r"\bart\. ?\d+")
PARA_RE = re.compile(r"§ ?\d+")
BZP_RE = re.compile(r"\d{4}/[A-Z]{3} ?\d{8}")
TED_RE = re.compile(r"(?<!\d)\d{4}/S ?\d{3}-\d{6}|(?<![\d/\-])\d{6}-\d{4}(?![\d\-])")
SIGNATURE_LINE_RE = re.compile(r"^\s*[.…_]{8,}\s*$")
NAME_LINE_RE = re.compile(
    rf"^(?:[{_LO}]{{1,4}}\. ?){{0,2}}[{_UP}][{_LO}]+(?:-[{_UP}][{_LO}]+)? [{_UP}][{_LO}]+(?:-[{_UP}][{_LO}]+)?\s*[,(]?"
)
NAME_FULL_RE = re.compile(
    rf"^(?:[{_LO}]{{1,4}}\. ?){{0,2}}[{_UP}][{_LO}]+(?:-[{_UP}][{_LO}]+)?(?: [{_UP}][{_LO}]+(?:-[{_UP}][{_LO}]+)?){{1,2}}$"
)
KEY_VALUE_RE = re.compile(r"^[^\W\d][^:\n]{1,40}:\s*\S")
ENUM_RE = re.compile(r"^\s*(?:\d{1,2}[.)]|[a-z][.)]|[–\-•])\s+\S")
QUOTE_RE = re.compile(r"„[^”]{10,}”")

FEATURE_ALIASES: dict[str, str] = {
    "nrb": "account_numbers",
    "nrb_suma_ok": "account_checksum_ok",
    "kwoty": "amounts",
    "daty": "dates",
    "id_10_9": "id_9_10",
    "email": "emails",
    "kod_poczt": "postal_codes",
    "art": "article_refs",
    "paragraf": "section_refs",
    "linie_podpisu": "signature_lines",
    "podpisy": "signatures",
    "linie_nazwisk": "name_lines",
    "kv_pct": "key_value_pct",
    "wyliczenia": "enumerations",
    "cytaty": "quotes",
    "znaki_k": "chars_k",
}

_OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}


@dataclass
class RuleCheck:
    type_id: str
    rule_ok: bool
    failed_conditions: list[str]
    envelope_violations: list[str]
    envelope_ok: bool


class FeatureExtractor:
    """Computes keyword-free format feature counts (NRB, amounts, dates, IDs, citations, etc.)."""

    @staticmethod
    def extract(text: str) -> dict[str, Any]:
        cleaned = fix_ocr_digits(text.replace("\r\n", "\n").replace("\r", "\n"), include_separators=False)
        lines = [line for line in cleaned.split("\n") if line.strip()]
        flat = re.sub(r"\s+", " ", cleaned)
        accounts = NRB_RE.findall(flat)
        no_accounts = NRB_RE.sub(" ", flat)
        sig_idx = [i for i, line in enumerate(lines) if SIGNATURE_LINE_RE.match(line)]
        signed = sum(
            1
            for i in sig_idx
            if any(NAME_LINE_RE.match(lines[j].strip()) for j in range(i + 1, min(i + 3, len(lines))))
        )
        return {
            "account_numbers": len(accounts),
            "account_checksum_ok": sum(nrb_checksum_ok(x) for x in accounts),
            "amounts": len(AMOUNT_RE.findall(no_accounts)),
            "dates": len(DATE_RE.findall(flat)),
            "id_9_10": len(ID_9_10_RE.findall(no_accounts)),
            "id_11": len(ID_11_RE.findall(no_accounts)),
            "emails": len(EMAIL_RE.findall(flat)),
            "postal_codes": len(POSTAL_RE.findall(no_accounts)),
            "article_refs": len(ART_RE.findall(flat)),
            "section_refs": len(PARA_RE.findall(flat)),
            "bzp": len(BZP_RE.findall(flat)),
            "ted": len(TED_RE.findall(flat)),
            "signature_lines": len(sig_idx),
            "signatures": signed,
            "name_lines": sum(bool(NAME_FULL_RE.match(line.strip())) for line in lines),
            "key_value_pct": round(100 * sum(bool(KEY_VALUE_RE.match(line)) for line in lines) / max(1, len(lines))),
            "enumerations": sum(bool(ENUM_RE.match(line)) for line in lines),
            "quotes": len(QUOTE_RE.findall(flat)),
            "chars_k": round(len(flat) / 1000, 1),
        }


class RuleClassifier(DocumentClassifier):
    """Evaluates deterministic entry rules defined in each `config/types/<type>.json`."""

    def __init__(self, feature_extractor: FeatureExtractor | None = None):
        self._features = feature_extractor or FeatureExtractor()

    @staticmethod
    def _lookup_feature(feats: dict[str, Any], name: str) -> Any:
        return feats.get(FEATURE_ALIASES.get(name, name))

    def _evaluate_rule(self, type_id: str, cls_cfg: dict[str, Any], feats: dict[str, Any]) -> RuleCheck:
        failed: list[str] = []
        for feat_name, op, threshold in cls_cfg.get("rule") or []:
            actual = self._lookup_feature(feats, feat_name)
            if actual is None:
                raise KeyError(f"type {type_id!r}: unknown feature {feat_name!r} in classification rule")
            if not _OPS[op](actual, threshold):
                failed.append(f"{feat_name} {op} {threshold} (actual {actual})")
        envelope = cls_cfg.get("envelope") or {}
        violations = [
            f"{f}={self._lookup_feature(feats, f)} outside [{lo}, {hi}]"
            for f, (lo, hi) in envelope.items()
            if self._lookup_feature(feats, f) is not None and not lo <= self._lookup_feature(feats, f) <= hi
        ]
        max_viol = int(cls_cfg.get("max_violations", 0))
        return RuleCheck(type_id, not failed, failed, violations, len(violations) <= max_viol)

    def classify(self, text: str, registry: DocumentTypeRegistry) -> ClassificationResult:
        feats = self._features.extract(text)
        checks = [self._evaluate_rule(tid, dt.classification_cfg, feats) for tid, dt in registry.items()]
        winner = next((c for c in checks if c.rule_ok and c.envelope_ok), None)
        if winner is not None:
            doc_type, reason = winner.type_id, f"satisfies the entry rule of '{winner.type_id}'"
        else:
            doc_type, reason = OTHER, "no type satisfies its entry rule"
        return ClassificationResult(
            doc_type=doc_type,
            method="rules",
            reason=reason,
            details={"features": feats, "checks": [asdict(c) for c in checks]},
        )
