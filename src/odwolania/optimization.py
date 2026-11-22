"""GEPA (Genetic-Pareto) reflective prompt optimizer for any configured document type.

Enables zero-code extensibility + instant quality tuning: drop a new `config/types/<type_id>.json`,
provide a folder of labeled `<sample>.pdf` + `<sample>.json` pairs, and run `optimize-type`
to automatically evolve `extraction.uwagi` (and optionally the global prompt) against exact F1.
"""
from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import DEFAULT_DOC_TYPES_DIR, DocumentType, DocumentTypeRegistry, PipelineConfig, resolve_path
from .evaluation import EvalBucket, F1Evaluator
from .extraction import DocumentExtractor, LLMConfig, OpenAICompatibleClient
from .ocr import DocumentReader
from .type_tools import TypeConfigValidator


@dataclass(frozen=True)
class DocSample:
    """One labeled PDF/JSON training or validation example."""

    rel: str
    doc_type: str
    text: str
    gold: dict[str, Any]


@dataclass
class DocTrace:
    """Single-document evaluation trace passed to GEPA's reflective dataset builder."""

    sample: DocSample
    raw_output: str
    cleaned: dict[str, Any] | None
    error: str | None
    mismatches: list[dict[str, Any]]
    tp: int
    fp: int
    fn: int
    tn: int
    f1: float


@dataclass
class OptimizationResult:
    """Summary of a GEPA optimization run."""

    type_id: str
    elapsed_seconds: float
    train_count: int
    val_count: int
    seed_train: dict[str, Any]
    seed_val: dict[str, Any]
    best_train: dict[str, Any]
    best_val: dict[str, Any]
    seed_candidate: dict[str, str]
    best_candidate: dict[str, str]
    applied_to: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_id": self.type_id,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "train_count": self.train_count,
            "val_count": self.val_count,
            "seed_train": self.seed_train,
            "seed_val": self.seed_val,
            "best_train": self.best_train,
            "best_val": self.best_val,
            "seed_candidate": self.seed_candidate,
            "best_candidate": self.best_candidate,
            "applied_to": self.applied_to,
        }


def _notes_list_to_text(notes: list[str]) -> str:
    return "\n".join(f"- {n}" for n in notes)


def _notes_text_to_list(text: str) -> list[str]:
    """Converts evolved markdown bullet text back into a clean JSON list for `extraction.uwagi`."""
    out: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ", "• ")):
            line = line[2:].strip()
        if line:
            out.append(line)
    return out


class OpenAICompatibleReflectionLM:
    """Reflection LLM caller for GEPA (supports OpenRouter reasoning + standard OpenAI-compatible endpoints)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.6,
        max_tokens: int = 16384,
        timeout: float = 240.0,
        reasoning: bool = True,
        retries: int = 4,
    ):
        self._url = base_url if base_url.endswith("/chat/completions") else base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._reasoning = reasoning
        self._retries = retries

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._reasoning and "openrouter.ai" in self._url.lower():
            payload["reasoning"] = {"enabled": True}

        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=httpx.Timeout(self._timeout, connect=20.0)) as client:
            for attempt in range(self._retries + 1):
                try:
                    resp = client.post(self._url, headers=headers, json=payload)
                except httpx.TransportError:
                    if attempt >= self._retries:
                        raise
                    time.sleep(min(2 ** attempt, 30) + random.uniform(0.0, 0.75))
                    continue
                if resp.status_code in (408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524) and attempt < self._retries:
                    delay = float(min(2 ** attempt, 30))
                    retry_after = resp.headers.get("retry-after")
                    if retry_after:
                        try:
                            delay = max(delay, min(float(retry_after), 30.0))
                        except ValueError:
                            pass
                    time.sleep(delay + random.uniform(0.0, 0.75))
                    continue
                if resp.status_code != 200:
                    raise RuntimeError(f"Reflection LM returned {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                choice = data["choices"][0]
                content = (choice["message"].get("content") or "").strip()
                if not content and attempt < self._retries and "reasoning" in payload:
                    payload["reasoning"] = {"effort": "low"}
                    continue
                return content
        raise RuntimeError("unreachable")


class GepaTypeOptimizer:
    """Runs GEPA reflective prompt evolution on any registered `DocumentType` using exact UZP F1."""

    def __init__(
        self,
        config: PipelineConfig,
        registry: DocumentTypeRegistry,
        *,
        types_dir: str | Path | None = None,
        llm_config: LLMConfig | None = None,
    ):
        self._config = config
        self._registry = registry
        self._types_dir = resolve_path(types_dir) if types_dir else resolve_path(config.types_dir)
        self._llm_config = llm_config or LLMConfig.from_pipeline_config(config)
        self._reader = DocumentReader(config)
        self._evaluator = F1Evaluator.from_pipeline_config(config)
        self._extractor = DocumentExtractor(
            client=OpenAICompatibleClient(self._llm_config),
            pipeline_config=config,
        )

    def load_samples(self, data_dir: str | Path, doc_type: DocumentType) -> list[DocSample]:
        """Loads all `(pdf, json)` pairs under `data_dir` for the given `DocumentType`."""
        root = Path(data_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"labeled data folder not found: {root}")
        samples: list[DocSample] = []
        pages_spec = doc_type.extraction_pages
        for pdf in sorted(root.rglob("*.pdf")):
            if pdf.name.startswith("."):
                continue
            gt_path = pdf.with_suffix(".json")
            if not gt_path.is_file() and pdf.stem == "potwierdzenie_przelewu":
                alt = pdf.with_name("przelew.json")
                if alt.is_file():
                    gt_path = alt
            if not gt_path.is_file():
                continue
            doc = self._reader.read(pdf, pages=pages_spec)
            if doc.strategy == "error":
                continue
            gold = json.loads(gt_path.read_text(encoding="utf-8"))
            if not isinstance(gold, dict) or not gold:
                continue
            rel = str(pdf.relative_to(root)).replace("\\", "/")
            samples.append(DocSample(rel=rel, doc_type=doc_type.type_id, text=doc.text_for(), gold=gold))
        return samples

    @staticmethod
    def split_samples(
        samples: list[DocSample],
        *,
        train_ratio: float = 0.5,
        seed: int = 42,
    ) -> tuple[list[DocSample], list[DocSample]]:
        if len(samples) < 2:
            raise ValueError(f"need at least 2 labeled (pdf, json) pairs to optimize, found {len(samples)}")
        shuffled = list(samples)
        random.Random(seed).shuffle(shuffled)
        k = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_ratio))))
        return shuffled[:k], shuffled[k:]

    def _eval_one(
        self,
        sample: DocSample,
        doc_type: DocumentType,
        candidate: dict[str, str],
    ) -> DocTrace:
        notes_override = _notes_text_to_list(candidate["notes"]) if "notes" in candidate else None
        tpl_override = candidate.get("system_prompt")
        raw = ""
        try:
            res = self._extractor.extract(
                sample.text,
                doc_type,
                notes_override=notes_override,
                template_override=tpl_override,
            )
            raw = res.trace.raw_output
            bucket = self._evaluator.compare(sample.gold, res.data, file_label=sample.rel)
            return DocTrace(
                sample=sample,
                raw_output=raw,
                cleaned=res.data,
                error=None,
                mismatches=bucket.mismatches,
                tp=bucket.total.tp,
                fp=bucket.total.fp,
                fn=bucket.total.fn,
                tn=bucket.total.tn,
                f1=bucket.total.f1,
            )
        except Exception as exc:
            bucket = self._evaluator.compare(sample.gold, {}, file_label=sample.rel)
            return DocTrace(
                sample=sample,
                raw_output=raw[:500],
                cleaned={},
                error=f"{type(exc).__name__}: {exc}",
                mismatches=bucket.mismatches,
                tp=0,
                fp=0,
                fn=bucket.total.fn,
                tn=0,
                f1=0.0,
            )

    def _score_split(
        self,
        split: list[DocSample],
        doc_type: DocumentType,
        candidate: dict[str, str],
        max_workers: int,
    ) -> dict[str, Any]:
        workers = min(max_workers, max(1, len(split)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            traces = list(pool.map(lambda s: self._eval_one(s, doc_type, candidate), split))
        agg = EvalBucket()
        for t in traces:
            agg.total.tp += t.tp
            agg.total.fp += t.fp
            agg.total.fn += t.fn
            agg.total.tn += t.tn
            agg.mismatches.extend(t.mismatches)
        return {**agg.total.to_dict(), "mismatches": agg.mismatches[:20]}

    def optimize(
        self,
        type_id: str,
        data_dir: str | Path,
        *,
        optimize_system: bool = False,
        max_metric_calls: int | None = None,
        reflection_model: str | None = None,
        reflection_base_url: str | None = None,
        train_ratio: float | None = None,
        seed: int | None = None,
        apply: bool = False,
        run_dir: str | Path | None = None,
    ) -> OptimizationResult:
        """Evolves `extraction.uwagi` for `type_id` via GEPA and optionally persists it back to JSON."""
        try:
            import gepa
            from gepa.core.adapter import EvaluationBatch, GEPAAdapter
        except ImportError as exc:
            raise ImportError(
                "GEPA optimization requires optional packages: pip install -r requirements-dev.txt"
            ) from exc

        doc_type = self._registry.get(type_id)
        if doc_type is None:
            raise KeyError(f"unknown document type {type_id!r} (available: {', '.join(self._registry.keys())})")

        opt_cfg = self._config.optimization or {}
        eff_ratio = float(opt_cfg.get("train_ratio", 0.5) if train_ratio is None else train_ratio)
        eff_seed = int(opt_cfg.get("seed", 42) if seed is None else seed)
        eff_budget = int(opt_cfg.get("max_metric_calls", 60) if max_metric_calls is None else max_metric_calls)
        eff_mb = int(opt_cfg.get("minibatch_size", 3))
        eff_workers = int(opt_cfg.get("max_workers", 4))
        eff_refl_model = reflection_model or str(opt_cfg.get("reflection_model") or self._llm_config.model)
        eff_refl_url = reflection_base_url or str(opt_cfg.get("reflection_base_url") or self._llm_config.base_url)
        refl_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or self._llm_config.api_key
        if not refl_key:
            raise RuntimeError("API key required for GEPA reflection LM (set LLM_API_KEY or OPENROUTER_API_KEY in .env)")

        samples = self.load_samples(data_dir, doc_type)
        train_set, val_set = self.split_samples(samples, train_ratio=eff_ratio, seed=eff_seed)

        seed_candidate: dict[str, str] = {"notes": _notes_list_to_text(doc_type.extraction_notes)}
        if optimize_system:
            seed_candidate["system_prompt"] = self._config.prompts.extraction_system_prompt

        optimizer = self

        class _SingleTypeAdapter(GEPAAdapter[DocSample, DocTrace, dict[str, Any]]):
            def evaluate(
                self,
                batch: list[DocSample],
                candidate: dict[str, str],
                capture_traces: bool = False,
            ) -> EvaluationBatch[DocTrace, dict[str, Any]]:
                workers = min(eff_workers, max(1, len(batch)))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    traces = list(pool.map(lambda s: optimizer._eval_one(s, doc_type, candidate), batch))
                return EvaluationBatch(
                    outputs=[t.cleaned or {} for t in traces],
                    scores=[t.f1 for t in traces],
                    trajectories=traces if capture_traces else None,
                )

            def make_reflective_dataset(
                self,
                candidate: dict[str, str],
                eval_batch: EvaluationBatch[DocTrace, dict[str, Any]],
                components_to_update: list[str],
            ) -> dict[str, list[dict[str, Any]]]:
                out: dict[str, list[dict[str, Any]]] = {c: [] for c in components_to_update}
                if not eval_batch.trajectories:
                    return out
                for tr in eval_batch.trajectories:
                    if tr.error:
                        fb = f"Error: {tr.error}. Raw: {tr.raw_output[:300]}"
                    elif not tr.mismatches:
                        fb = f"Perfect extraction (F1=1.000, TP={tr.tp}, TN={tr.tn})."
                    else:
                        lines = [f"F1={tr.f1:.3f} (TP={tr.tp}, FP={tr.fp}, FN={tr.fn}). Mismatches:"]
                        for mm in tr.mismatches[:12]:
                            lines.append(
                                f"  - [{mm['kind'].upper()}] {mm['field']}: expected={mm['gold']!r}, got={mm['pred']!r}"
                            )
                        fb = "\n".join(lines)
                    rec = {
                        "Inputs": f"type_id={type_id}, file={tr.sample.rel}\nText (excerpt):\n{tr.sample.text[:3500]}",
                        "Generated Outputs": json.dumps(tr.cleaned, ensure_ascii=False),
                        "Feedback": fb,
                    }
                    for comp in components_to_update:
                        out[comp].append(rec)
                return out

        reflection_lm = OpenAICompatibleReflectionLM(
            base_url=eff_refl_url,
            model=eff_refl_model,
            api_key=refl_key,
            temperature=float(opt_cfg.get("reflection_temperature", 0.6)),
            max_tokens=int(opt_cfg.get("reflection_max_tokens", 16384)),
            timeout=float(opt_cfg.get("reflection_timeout", 240.0)),
            reasoning=bool(opt_cfg.get("reflection_reasoning", True)),
        )

        t0 = time.perf_counter()
        seed_train = self._score_split(train_set, doc_type, seed_candidate, eff_workers)
        seed_val = self._score_split(val_set, doc_type, seed_candidate, eff_workers)

        gepa_result = gepa.optimize(
            seed_candidate=seed_candidate,
            trainset=train_set,
            valset=val_set,
            adapter=_SingleTypeAdapter(),
            reflection_lm=reflection_lm,
            reflection_prompt_template=self._config.prompts.gepa_reflection_prompt or None,
            reflection_minibatch_size=min(eff_mb, len(train_set)),
            max_metric_calls=eff_budget,
            seed=eff_seed,
            run_dir=str(run_dir) if run_dir else None,
        )
        best_candidate = dict(gepa_result.best_candidate)
        best_train = self._score_split(train_set, doc_type, best_candidate, eff_workers)
        best_val = self._score_split(val_set, doc_type, best_candidate, eff_workers)
        elapsed = time.perf_counter() - t0

        applied_to: str | None = None
        if apply and best_val["f1"] >= seed_val["f1"]:
            target_path = self._types_dir / f"{type_id}.json"
            if target_path.is_file():
                updated = json.loads(target_path.read_text(encoding="utf-8"))
                updated.setdefault("extraction", {})["uwagi"] = _notes_text_to_list(best_candidate["notes"])
                issues = TypeConfigValidator().validate_dict(updated, source_name=target_path.name)
                if not any(i.severity == "error" for i in issues):
                    target_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    applied_to = str(target_path)

        return OptimizationResult(
            type_id=type_id,
            elapsed_seconds=elapsed,
            train_count=len(train_set),
            val_count=len(val_set),
            seed_train=seed_train,
            seed_val=seed_val,
            best_train=best_train,
            best_val=best_val,
            seed_candidate=seed_candidate,
            best_candidate=best_candidate,
            applied_to=applied_to,
        )
