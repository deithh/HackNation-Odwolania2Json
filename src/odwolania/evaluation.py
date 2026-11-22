"""UZP F1 evaluation engine (configurable via `config/pipeline.json`, zero mandatory extra deps)."""
from __future__ import annotations

import difflib
import itertools
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PipelineConfig

try:
    from rapidfuzz import fuzz as _rfuzz
except ImportError:
    _rfuzz = None


@dataclass
class EvalCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, other: "EvalCounts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class EvalBucket:
    total: EvalCounts = field(default_factory=EvalCounts)
    by_field: dict[str, EvalCounts] = field(default_factory=lambda: defaultdict(EvalCounts))
    mismatches: list[dict[str, Any]] = field(default_factory=list)

    def record(self, field_path: str, kind: str, gold: Any, pred: Any, file_label: str) -> None:
        setattr(self.total, kind, getattr(self.total, kind) + 1)
        fc = self.by_field[field_path]
        setattr(fc, kind, getattr(fc, kind) + 1)
        if kind in ("fp", "fn"):
            self.mismatches.append(
                {"file": file_label, "field": field_path, "kind": kind, "gold": gold, "pred": pred}
            )

    def merge(self, other: "EvalBucket") -> None:
        self.total.add(other.total)
        for key, counts in other.by_field.items():
            self.by_field[key].add(counts)
        self.mismatches.extend(other.mismatches)


class F1Evaluator:
    """Computes exact UZP field-level F1 (exact match + configurable fuzzy threshold for long strings)."""

    def __init__(
        self,
        *,
        fuzzy_threshold: float = 80.0,
        long_string_fields: set[str] | None = None,
    ):
        self.fuzzy_threshold = float(fuzzy_threshold)
        self.long_string_fields: set[str] = set(long_string_fields or set())

    @classmethod
    def from_pipeline_config(cls, config: PipelineConfig) -> "F1Evaluator":
        ev = config.evaluation or {}
        return cls(
            fuzzy_threshold=float(ev.get("fuzzy_threshold", 80.0)),
            long_string_fields=set(ev.get("long_string_fields") or []),
        )

    @staticmethod
    def _is_empty(val: Any) -> bool:
        if val is None:
            return True
        if isinstance(val, str) and not val.strip():
            return True
        return False

    @staticmethod
    def _norm_ws(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip())

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        a_norm = F1Evaluator._norm_ws(a).lower()
        b_norm = F1Evaluator._norm_ws(b).lower()
        if a_norm == b_norm:
            return 100.0
        if _rfuzz is not None:
            return float(_rfuzz.ratio(a_norm, b_norm))
        return difflib.SequenceMatcher(None, a_norm, b_norm).ratio() * 100.0

    def _values_match(self, field_path: str, gold: Any, pred: Any) -> bool:
        if type(gold) is bool or type(pred) is bool:
            return gold is pred
        if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
            return abs(float(gold) - float(pred)) < 1e-6
        if isinstance(gold, str) and isinstance(pred, str):
            g_norm, p_norm = self._norm_ws(gold), self._norm_ws(pred)
            if g_norm == p_norm:
                return True
            if field_path in self.long_string_fields:
                return self._string_similarity(g_norm, p_norm) >= self.fuzzy_threshold
            return False
        return str(gold).strip() == str(pred).strip()

    def _count_nonempty_leaves(
        self,
        obj: Any,
        path: str,
        kind: str,
        bucket: EvalBucket,
        file_label: str,
    ) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                sub = f"{path}.{k}" if path else k
                self._count_nonempty_leaves(v, sub, kind, bucket, file_label)
        elif isinstance(obj, list):
            for item in obj:
                self._count_nonempty_leaves(item, f"{path}[]", kind, bucket, file_label)
        elif not self._is_empty(obj):
            if kind == "fn":
                bucket.record(path, "fn", obj, None, file_label)
            else:
                bucket.record(path, "fp", None, obj, file_label)

    def _score_leaf(self, path: str, gold: Any, pred: Any, bucket: EvalBucket, file_label: str) -> None:
        g_empty = self._is_empty(gold)
        p_empty = self._is_empty(pred)
        if g_empty and p_empty:
            bucket.record(path, "tn", gold, pred, file_label)
        elif g_empty and not p_empty:
            bucket.record(path, "fp", gold, pred, file_label)
        elif not g_empty and p_empty:
            bucket.record(path, "fn", gold, pred, file_label)
        elif self._values_match(path, gold, pred):
            bucket.record(path, "tp", gold, pred, file_label)
        else:
            bucket.record(path, "fp", gold, pred, file_label)

    def _match_lists(
        self,
        path: str,
        gold_list: list[Any],
        pred_list: list[Any],
        bucket: EvalBucket,
        file_label: str,
    ) -> None:
        item_path = f"{path}[]"
        n_g, n_p = len(gold_list), len(pred_list)
        if n_g == 0 and n_p == 0:
            bucket.record(item_path, "tn", [], [], file_label)
            return
        if n_g == 0:
            for item in pred_list:
                self._count_nonempty_leaves(item, item_path, "fp", bucket, file_label)
            return
        if n_p == 0:
            for item in gold_list:
                self._count_nonempty_leaves(item, item_path, "fn", bucket, file_label)
            return

        pair_buckets: list[list[EvalBucket]] = []
        for g_item in gold_list:
            row: list[EvalBucket] = []
            for p_item in pred_list:
                sub = EvalBucket()
                self._compare_nodes(item_path, g_item, p_item, sub, file_label)
                row.append(sub)
            pair_buckets.append(row)

        k = min(n_g, n_p)
        matched_g: set[int] = set()
        matched_p: set[int] = set()
        if max(n_g, n_p) <= 7:
            best_score: tuple[int, int, int] | None = None
            best_pairs: list[tuple[int, int]] = []
            if n_g <= n_p:
                for perm in itertools.permutations(range(n_p), k):
                    tp = sum(pair_buckets[i][perm[i]].total.tp for i in range(k))
                    err = sum(pair_buckets[i][perm[i]].total.fp + pair_buckets[i][perm[i]].total.fn for i in range(k))
                    tn = sum(pair_buckets[i][perm[i]].total.tn for i in range(k))
                    score = (tp, -err, tn)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_pairs = [(i, perm[i]) for i in range(k)]
            else:
                for perm in itertools.permutations(range(n_g), k):
                    tp = sum(pair_buckets[perm[j]][j].total.tp for j in range(k))
                    err = sum(pair_buckets[perm[j]][j].total.fp + pair_buckets[perm[j]][j].total.fn for j in range(k))
                    tn = sum(pair_buckets[perm[j]][j].total.tn for j in range(k))
                    score = (tp, -err, tn)
                    if best_score is None or score > best_score:
                        best_score = score
                        best_pairs = [(perm[j], j) for j in range(k)]
            pairs = best_pairs
        else:
            candidates = sorted(
                (
                    (
                        pair_buckets[i][j].total.tp,
                        -(pair_buckets[i][j].total.fp + pair_buckets[i][j].total.fn),
                        pair_buckets[i][j].total.tn,
                        i,
                        j,
                    )
                    for i in range(n_g)
                    for j in range(n_p)
                ),
                reverse=True,
            )
            pairs = []
            for _, _, _, i, j in candidates:
                if i not in matched_g and j not in matched_p:
                    pairs.append((i, j))
                    matched_g.add(i)
                    matched_p.add(j)

        for i, j in pairs:
            matched_g.add(i)
            matched_p.add(j)
            bucket.merge(pair_buckets[i][j])
        for i in range(n_g):
            if i not in matched_g:
                self._count_nonempty_leaves(gold_list[i], item_path, "fn", bucket, file_label)
        for j in range(n_p):
            if j not in matched_p:
                self._count_nonempty_leaves(pred_list[j], item_path, "fp", bucket, file_label)

    def _compare_nodes(
        self,
        path: str,
        gold: Any,
        pred: Any,
        bucket: EvalBucket,
        file_label: str,
    ) -> None:
        if isinstance(gold, dict) and isinstance(pred, dict):
            if not gold and not pred:
                bucket.record(path or "<empty>", "tn", {}, {}, file_label)
                return
            if not gold and pred:
                self._count_nonempty_leaves(pred, path, "fp", bucket, file_label)
                return
            if gold and not pred:
                self._count_nonempty_leaves(gold, path, "fn", bucket, file_label)
                return
            for key in sorted(set(gold.keys()) | set(pred.keys())):
                sub = f"{path}.{key}" if path else key
                in_g, in_p = key in gold, key in pred
                if in_g and in_p:
                    self._compare_nodes(sub, gold[key], pred[key], bucket, file_label)
                elif in_g:
                    self._count_nonempty_leaves(gold[key], sub, "fn", bucket, file_label)
                else:
                    self._count_nonempty_leaves(pred[key], sub, "fp", bucket, file_label)
            return

        if isinstance(gold, list) and isinstance(pred, list):
            self._match_lists(path, gold, pred, bucket, file_label)
            return

        if isinstance(gold, (dict, list)) and pred is None:
            self._count_nonempty_leaves(gold, path, "fn", bucket, file_label)
            return
        if gold is None and isinstance(pred, (dict, list)):
            self._count_nonempty_leaves(pred, path, "fp", bucket, file_label)
            return

        self._score_leaf(path, gold, pred, bucket, file_label)

    def compare(self, gold: Any, pred: Any, *, file_label: str = "<doc>") -> EvalBucket:
        """Scores one predicted JSON object against one ground-truth JSON object."""
        bucket = EvalBucket()
        self._compare_nodes("", gold, pred, bucket, file_label)
        return bucket

    @staticmethod
    def collect_pairs(
        pred_dir: Path,
        gold_dir: Path,
        *,
        expect_empty: bool = False,
    ) -> list[tuple[str, Path | None, Path | None]]:
        pairs: list[tuple[str, Path | None, Path | None]] = []
        if expect_empty:
            for pdf in sorted(gold_dir.rglob("*.pdf")):
                rel = pdf.relative_to(gold_dir).with_suffix(".json")
                pred_path = pred_dir / rel
                pairs.append((str(rel).replace("\\", "/"), pred_path if pred_path.is_file() else None, None))
            return pairs

        pdfs = [p for p in sorted(gold_dir.rglob("*.pdf")) if not p.name.startswith(".")]
        if pdfs:
            for pdf in pdfs:
                rel_json = pdf.relative_to(gold_dir).with_suffix(".json")
                pred_path = pred_dir / rel_json
                if not pred_path.is_file() and pdf.stem == "potwierdzenie_przelewu":
                    alt = pred_dir / rel_json.with_name("przelew.json")
                    if alt.is_file():
                        pred_path = alt
                gold_path = pdf.with_suffix(".json")
                if not gold_path.is_file() and pdf.stem == "potwierdzenie_przelewu":
                    alt_g = pdf.with_name("przelew.json")
                    if alt_g.is_file():
                        gold_path = alt_g
                pairs.append(
                    (
                        str(rel_json).replace("\\", "/"),
                        pred_path if pred_path.is_file() else None,
                        gold_path if gold_path.is_file() else None,
                    )
                )
            return pairs

        for gold_path in sorted(gold_dir.rglob("*.json")):
            if gold_path.name == "manifest.json":
                continue
            rel = gold_path.relative_to(gold_dir)
            pred_path = pred_dir / rel
            if not pred_path.is_file() and gold_path.stem == "przelew":
                alt = pred_dir / rel.with_name("potwierdzenie_przelewu.json")
                if alt.is_file():
                    pred_path = alt
            pairs.append((str(rel).replace("\\", "/"), pred_path if pred_path.is_file() else None, gold_path))
        return pairs

    def evaluate_dir(
        self,
        pred_dir: str | Path,
        gold_dir: str | Path,
        *,
        expect_empty: bool = False,
    ) -> dict[str, Any]:
        """Evaluates all JSON files in `pred_dir` against `gold_dir` and returns a structured report."""
        pairs = self.collect_pairs(Path(pred_dir), Path(gold_dir), expect_empty=expect_empty)
        overall = EvalBucket()
        per_file: list[dict[str, Any]] = []

        for label, pred_path, gold_path in pairs:
            gold = {} if gold_path is None else json.loads(gold_path.read_text(encoding="utf-8"))
            if pred_path is None:
                pred: Any = {}
                missing = True
            else:
                pred = json.loads(pred_path.read_text(encoding="utf-8"))
                missing = False
            fb = self.compare(gold, pred, file_label=label)
            overall.merge(fb)
            per_file.append(
                {
                    "file": label,
                    "missing_pred": missing,
                    **fb.total.to_dict(),
                    "mismatches": len(fb.mismatches),
                }
            )

        return {
            "files": len(pairs),
            "fuzzy_threshold": self.fuzzy_threshold,
            "fuzzy_backend": "rapidfuzz" if _rfuzz is not None else "difflib",
            "overall": overall.total.to_dict(),
            "by_field": {k: v.to_dict() for k, v in sorted(overall.by_field.items())},
            "mismatches": overall.mismatches,
            "per_file": per_file,
        }
