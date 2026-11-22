"""Stateful workspace service powering the Odwołania2JSON web GUI."""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..classification import OTHER
from ..config import DEFAULT_PIPELINE_CONFIG, PROJECT_ROOT, DocumentTypeRegistry, PipelineConfig, resolve_path
from ..extraction import LLMConfig
from ..pipeline import DocumentPipeline
from ..type_tools import TypeConfigScaffolder, TypeConfigValidator
from .evidence import EvidenceLocator


def _update_dotenv(updates: dict[str, str | None], env_path: Path) -> None:
    """Safely updates or removes keys in `.env` while preserving unrelated lines."""
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.is_file() else []
    remaining = dict(updates)
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in remaining:
            val = remaining.pop(key)
            if val is not None:
                new_lines.append(f"{key}={val}")
        else:
            new_lines.append(line)
    for key, val in remaining.items():
        if val is not None:
            new_lines.append(f"{key}={val}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


@dataclass
class WebDocument:
    """One PDF loaded in the web workspace together with its inference output, trace, and visual highlights."""

    doc_id: str
    filename: str
    rel_path: str
    pdf_path: Path
    n_pages: int
    status: str = "ready"       # ready | processing | done | error
    doc_type: str | None = None
    strategy: str | None = None
    strategy_reason: str | None = None
    seconds: float | None = None
    error: str | None = None
    data: dict[str, Any] | None = None
    raw_output: str | None = None
    raw_parsed: dict[str, Any] | None = None
    input_text: str | None = None
    classification: dict[str, Any] | None = None
    highlights: list[dict[str, Any]] = field(default_factory=list)
    page_skew_angles: dict[int, float] = field(default_factory=dict)
    render_rev: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "rel_path": self.rel_path,
            "n_pages": self.n_pages,
            "status": self.status,
            "doc_type": self.doc_type,
            "strategy": self.strategy,
            "seconds": self.seconds,
            "error": self.error,
            "highlights_count": len(self.highlights),
        }

    def detail(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "strategy_reason": self.strategy_reason,
            "data": self.data,
            "classification": self.classification,
            "highlights": self.highlights,
            "render_rev": self.render_rev,
            "page_skew_angles": self.page_skew_angles,
            "trace": {
                "input_text": self.input_text or "",
                "raw_output": self.raw_output or "",
                "raw_parsed": self.raw_parsed,
            },
        }


class WebWorkspaceService:
    """Coordinates PDF uploads, live inference, visual evidence highlighting, and zero-code type editing."""

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        types_dir: str | Path | None = None,
        llm_config: LLMConfig | None = None,
    ):
        self._config_path = config_path
        self._types_dir_override = types_dir
        self._llm_config = llm_config
        self._config = PipelineConfig.load(config_path)
        self._registry = DocumentTypeRegistry.from_dir(types_dir or self._config.types_dir)
        self._pipeline = DocumentPipeline(self._config, self._registry, llm_config=llm_config)
        self._locator = EvidenceLocator()
        self._validator = TypeConfigValidator()
        self._scaffolder = TypeConfigScaffolder(self._validator)
        self._lock = threading.RLock()
        self._docs: dict[str, WebDocument] = {}
        self._opt_jobs: dict[str, dict[str, Any]] = {}
        self._page_png_cache: dict[tuple[str, int], bytes] = {}
        self._upload_dir = resolve_path(self._config.cache_dir) / "web_uploads"
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def _effective_types_dir(self) -> Path:
        return resolve_path(self._types_dir_override or self._config.types_dir)

    def reload_types(self) -> None:
        with self._lock:
            self._config = PipelineConfig.load(self._config_path)
            self._registry = DocumentTypeRegistry.from_dir(self._effective_types_dir())
            self._pipeline = DocumentPipeline(self._config, self._registry, llm_config=self._llm_config)

    def info(self) -> dict[str, Any]:
        llm_cfg = self._llm_config or LLMConfig.from_pipeline_config(self._config)
        return {
            "model": llm_cfg.model,
            "base_url": llm_cfg.base_url,
            "ocr_engine": self._config.ocr_engine_name,
            "classification_method": self._config.classification.get("method", "decider"),
            "fallback_to_rules": bool(self._config.classification.get("fallback_to_rules", True)),
            "types_count": len(self._registry),
            "types": [{"type_id": tid, "name": dt.name} for tid, dt in self._registry.items()],
        }

    # --- Documents ---

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            return [doc.summary() for doc in self._docs.values()]

    def get_document(self, doc_id: str) -> dict[str, Any]:
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                raise KeyError(f"document not found: {doc_id}")
            return doc.detail()

    def upload_pdf(self, filename: str, pdf_bytes: bytes, *, rel_path: str | None = None) -> dict[str, Any]:
        safe_name = Path(filename or "document.pdf").name
        if not safe_name.lower().endswith(".pdf"):
            raise ValueError("only .pdf files are supported")
        doc_id = uuid.uuid4().hex[:12]
        target = self._upload_dir / f"{doc_id}_{safe_name}"
        target.write_bytes(pdf_bytes)
        n_pages = self._locator.page_count_bytes(pdf_bytes)
        doc = WebDocument(
            doc_id=doc_id,
            filename=safe_name,
            rel_path=rel_path or safe_name,
            pdf_path=target,
            n_pages=n_pages,
        )
        with self._lock:
            self._docs[doc_id] = doc
        return doc.summary()

    def load_from_input_dir(self) -> list[dict[str, Any]]:
        in_dir = resolve_path(self._config.input_dir)
        out_dir = resolve_path(self._config.output_dir)
        if not in_dir.is_dir():
            return []
        added: list[dict[str, Any]] = []
        with self._lock:
            existing_paths = {d.pdf_path.resolve() for d in self._docs.values()}
        for pdf in DocumentPipeline.iter_inputs(in_dir):
            if pdf.resolve() in existing_paths:
                continue
            rel = str(pdf.relative_to(in_dir)).replace("\\", "/")
            doc_id = uuid.uuid4().hex[:12]
            n_pages = self._locator.page_count(pdf)
            doc = WebDocument(
                doc_id=doc_id,
                filename=pdf.name,
                rel_path=rel,
                pdf_path=pdf,
                n_pages=n_pages,
            )
            # Load pre-existing output JSON if already generated via CLI
            existing_json = (out_dir / Path(rel)).with_suffix(".json")
            if existing_json.is_file():
                try:
                    parsed = json.loads(existing_json.read_text(encoding="utf-8"))
                    if isinstance(parsed, dict):
                        doc.data = parsed
                        doc.status = "done"
                        doc.doc_type = self._guess_type_from_data(parsed)
                        doc.highlights = self._locator.locate(pdf, parsed)
                except Exception:
                    pass
            with self._lock:
                self._docs[doc_id] = doc
            added.append(doc.summary())
        return added

    def _guess_type_from_data(self, data: dict[str, Any]) -> str:
        if not data:
            return OTHER
        top_keys = set(data.keys())
        for tid, dt in self._registry.items():
            req = set(dt.schema.get("required") or [])
            if req and req.issubset(top_keys):
                return tid
        return OTHER

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self._docs.pop(doc_id, None)
            for key in [k for k in self._page_png_cache if k[0] == doc_id]:
                self._page_png_cache.pop(key, None)

    def analyze_document(self, doc_id: str, *, force_ocr: bool = False) -> dict[str, Any]:
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                raise KeyError(f"document not found: {doc_id}")
            doc.status = "processing"
            doc.error = None
            pipeline = self._pipeline
            if force_ocr:
                cfg = self._config.with_force_ocr(True)
                pipeline = DocumentPipeline(cfg, self._registry, llm_config=self._llm_config)

        t0 = time.perf_counter()
        try:
            doc_text = pipeline.reader.read(doc.pdf_path)
            if doc_text.strategy == "error":
                raise RuntimeError(doc_text.strategy_reason)

            cls_res = pipeline.classifier.classify(doc_text.text_for(), pipeline.registry)
            cls_dict = cls_res.to_dict()
            doc_type = pipeline.registry.get(cls_res.doc_type)

            if doc_type is None:
                data: dict[str, Any] = {}
                input_text = doc_text.text_for()
                raw_output = "{}"
                raw_parsed: dict[str, Any] | None = {}
                highlights: list[dict[str, Any]] = []
            else:
                pages_spec = doc_type.extraction_pages
                ext_doc = pipeline.reader.read(doc.pdf_path, pages_spec) if pages_spec else doc_text
                # Ensure OCR pages carry bounding-box spans even if loaded from an older pre-spans cache
                if ext_doc.strategy in ("ocr", "hybrid"):
                    missing_spans = any(p.source == "ocr" and not p.spans for p in ext_doc.pages)
                    if missing_spans:
                        forced_reader = DocumentPipeline(
                            self._config.with_force_ocr(True),
                            self._registry,
                            llm_config=self._llm_config,
                        ).reader
                        # Compute fresh spans for overlay without altering text if already cached
                        ext_doc = forced_reader.read(doc.pdf_path, pages_spec)
                input_text = ext_doc.text_for()
                ext_res = pipeline.extractor.extract(input_text, doc_type)
                data = ext_res.data
                raw_output = ext_res.trace.raw_output
                raw_parsed = ext_res.trace.raw_parsed
                highlights = self._locator.locate(doc.pdf_path, data, doc_text=ext_doc)
                skew_map = {p.index: p.skew_angle for p in ext_doc.pages if abs(p.skew_angle) >= 0.1}
            if doc_type is None:
                skew_map = {p.index: p.skew_angle for p in doc_text.pages if abs(p.skew_angle) >= 0.1}

            elapsed = round(time.perf_counter() - t0, 2)
            with self._lock:
                doc.status = "done"
                doc.doc_type = cls_res.doc_type
                doc.strategy = doc_text.strategy
                doc.strategy_reason = doc_text.strategy_reason
                doc.seconds = elapsed
                doc.data = data
                doc.classification = cls_dict
                doc.input_text = input_text
                doc.raw_output = raw_output
                doc.raw_parsed = raw_parsed
                doc.highlights = highlights
                doc.error = None
                if skew_map != doc.page_skew_angles:
                    doc.page_skew_angles = skew_map
                    doc.render_rev += 1
                    for key in [k for k in self._page_png_cache if k[0] == doc_id]:
                        self._page_png_cache.pop(key, None)
            return doc.detail()
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 2)
            with self._lock:
                doc.status = "error"
                doc.seconds = elapsed
                doc.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            return doc.detail()

    def get_page_png(self, doc_id: str, page_index: int) -> bytes:
        with self._lock:
            doc = self._docs.get(doc_id)
            if doc is None:
                raise KeyError(f"document not found: {doc_id}")
            cached = self._page_png_cache.get((doc_id, page_index))
            if cached is not None:
                return cached
            pdf_path = doc.pdf_path
            skew_angle = doc.page_skew_angles.get(page_index, 0.0)
        width_px = int(self._config.web.get("page_render_width_px", 1200))
        png = self._locator.render_page_png(pdf_path, page_index, width_px=width_px, skew_angle=skew_angle)
        with self._lock:
            self._page_png_cache[(doc_id, page_index)] = png
        return png

    def export_all_zip(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            with self._lock:
                for doc in self._docs.values():
                    if doc.data is None:
                        continue
                    arc_name = str(Path(doc.rel_path).with_suffix(".json")).replace("\\", "/")
                    zf.writestr(arc_name, json.dumps(doc.data, ensure_ascii=False, indent=2) + "\n")
        return buf.getvalue()

    # --- Zero-code Document Types ---

    def list_types(self) -> dict[str, Any]:
        types_dir = self._effective_types_dir()
        report = self._validator.validate_dir(types_dir)
        issues_by_file: dict[str, list[dict[str, str]]] = {}
        for issue in report.issues:
            issues_by_file.setdefault(issue.file, []).append(
                {"severity": issue.severity, "message": issue.message}
            )
        items: list[dict[str, Any]] = []
        for tid, dt in self._registry.items():
            fname = f"{tid}.json"
            items.append(
                {
                    "type_id": tid,
                    "name": dt.name,
                    "description": dt.description,
                    "priority": dt.priority,
                    "pages": dt.extraction_pages,
                    "notes_count": len(dt.extraction_notes),
                    "lookup_tables": list(dt.extraction_lookups.keys()),
                    "consistency_rules": [r.get("rule") for r in dt.consistency_rules],
                    "issues": issues_by_file.get(fname, []),
                    "raw": dt.raw,
                }
            )
        # Also list optional ready-to-enable examples from config/types_examples/
        examples_dir = resolve_path("config/types_examples")
        examples: list[dict[str, Any]] = []
        if examples_dir.is_dir():
            for p in sorted(examples_dir.glob("*.json")):
                try:
                    raw = json.loads(p.read_text(encoding="utf-8-sig"))
                    examples.append(
                        {
                            "filename": p.name,
                            "type_id": raw.get("type_id", p.stem),
                            "name": raw.get("nazwa", p.stem),
                            "description": raw.get("opis", ""),
                            "active": raw.get("type_id") in self._registry,
                            "raw": raw,
                        }
                    )
                except Exception:
                    pass
        return {
            "valid": report.ok,
            "checked_files": report.checked_files,
            "types": items,
            "examples": examples,
        }

    def save_type_config(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        issues = self._validator.validate_dict(config_dict, source_name=f"{config_dict.get('type_id', 'new')}.json")
        errors = [{"severity": i.severity, "message": i.message} for i in issues if i.severity == "error"]
        if errors:
            return {"ok": False, "errors": errors}
        target, _ = self._scaffolder.save(config_dict, types_dir=self._effective_types_dir(), overwrite=True)
        self.reload_types()
        return {"ok": True, "path": str(target), "type_id": config_dict["type_id"]}

    def delete_type_config(self, type_id: str) -> dict[str, Any]:
        target = self._effective_types_dir() / f"{type_id}.json"
        if not target.is_file():
            raise FileNotFoundError(f"type config not found: {type_id}")
        target.unlink()
        self.reload_types()
        return {"ok": True, "deleted": type_id}

    def infer_type_from_upload(
        self,
        filename: str,
        payload: bytes,
        *,
        type_id: str,
        name: str | None = None,
        description: str = "",
        priority: int = 10,
    ) -> dict[str, Any]:
        suffix = Path(filename).suffix.lower()
        if suffix == ".json":
            sample_obj = json.loads(payload.decode("utf-8"))
            draft = self._scaffolder.infer_from_json_sample(
                sample_obj,
                type_id,
                name=name,
                description=description,
                priority=priority,
            )
        elif suffix == ".pdf":
            tmp = self._upload_dir / f"infer_{uuid.uuid4().hex[:8]}.pdf"
            tmp.write_bytes(payload)
            try:
                draft = self._scaffolder.infer_from_pdf_sample(
                    tmp,
                    type_id,
                    name=name,
                    priority=priority,
                    pipeline_config=self._config,
                    llm_config=self._llm_config,
                )
            finally:
                tmp.unlink(missing_ok=True)
        else:
            raise ValueError(f"unsupported sample extension {suffix!r} (expected .json or .pdf)")
        issues = self._validator.validate_dict(draft, source_name=f"{type_id}.json")
        return {
            "draft": draft,
            "valid": not any(i.severity == "error" for i in issues),
            "issues": [{"severity": i.severity, "message": i.message} for i in issues],
        }

    # --- Prompt Optimization (GEPA) ---

    def start_type_optimization(
        self,
        *,
        type_id: str,
        data_dir: str = "",
        samples: list[dict[str, Any]] | None = None,
        max_metric_calls: int = 20,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Launches a background GEPA prompt-optimization job using the currently configured LLM model."""
        with self._lock:
            if type_id not in self._registry:
                raise KeyError(f"nieznany typ dokumentu: {type_id}")
            cfg = self._config
            registry = self._registry
            types_dir = self._effective_types_dir()
            llm_cfg = self._llm_config or LLMConfig.from_pipeline_config(cfg)

        budget = max(4, min(120, int(max_metric_calls or 20)))
        job_id = uuid.uuid4().hex[:12]
        temp_dir: Path | None = None

        if samples:
            temp_dir = self._upload_dir / f"opt_{job_id}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            for item in samples:
                fname = Path(str(item.get("filename") or "")).name
                ext = Path(fname).suffix.lower()
                if not fname or ext not in (".pdf", ".json"):
                    continue
                raw_b64 = item.get("content_base64")
                if not raw_b64:
                    continue
                (temp_dir / fname).write_bytes(base64.b64decode(raw_b64))
            target_dir = temp_dir
        elif data_dir.strip():
            raw_p = Path(data_dir.strip())
            target_dir = raw_p if raw_p.is_absolute() else (PROJECT_ROOT / raw_p)
            if not target_dir.is_dir():
                raise ValueError(f"katalog z próbkami nie istnieje: {target_dir}")
        else:
            raise ValueError("wybierz pliki (.pdf + .json) lub podaj ścieżkę katalogu z próbkami")

        job: dict[str, Any] = {
            "job_id": job_id,
            "type_id": type_id,
            "model": llm_cfg.model,
            "status": "running",
            "started_at": time.time(),
            "elapsed_seconds": 0.0,
            "budget": budget,
            "apply": bool(apply),
            "result": None,
            "error": None,
        }
        with self._lock:
            self._opt_jobs[job_id] = job

        def _worker() -> None:
            t0 = time.perf_counter()
            try:
                from ..optimization import GepaTypeOptimizer, _notes_text_to_list

                optimizer = GepaTypeOptimizer(
                    cfg,
                    registry,
                    types_dir=types_dir,
                    llm_config=llm_cfg,
                )
                res = optimizer.optimize(
                    type_id=type_id,
                    data_dir=target_dir,
                    max_metric_calls=budget,
                    reflection_model=llm_cfg.model,
                    reflection_base_url=llm_cfg.base_url,
                    apply=bool(apply),
                )
                best_notes = _notes_text_to_list(res.best_candidate.get("notes", ""))
                if res.applied_to:
                    self.reload_types()
                with self._lock:
                    job["status"] = "done"
                    job["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
                    job["result"] = {
                        **res.to_dict(),
                        "best_notes": best_notes,
                    }
            except Exception as exc:
                traceback.print_exc()
                with self._lock:
                    job["status"] = "error"
                    job["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
                    job["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if temp_dir is not None:
                    shutil.rmtree(temp_dir, ignore_errors=True)

        threading.Thread(target=_worker, name=f"gepa-{job_id}", daemon=True).start()
        return dict(job)

    def get_optimization_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._opt_jobs.get(job_id)
            if job is None:
                raise KeyError(f"nie znaleziono zadania optymalizacji: {job_id}")
            snap = dict(job)
            if snap["status"] == "running":
                snap["elapsed_seconds"] = round(time.time() - float(snap["started_at"]), 1)
            return snap

    # --- Runtime & API Configuration (keys are write-only, never returned on read) ---

    def _effective_config_path(self) -> Path:
        return resolve_path(self._config_path) if self._config_path else DEFAULT_PIPELINE_CONFIG

    def get_settings(self) -> dict[str, Any]:
        with self._lock:
            cfg = self._config
            llm_cfg = LLMConfig.from_pipeline_config(cfg)
            cls_cfg = cfg.classification or {}
            dec_cfg = cls_cfg.get("decider") or {}
            has_dedicated_llm_key = bool(os.getenv("LLM_API_KEY"))
            has_openrouter_key = bool(os.getenv("OPENROUTER_API_KEY"))
            if has_dedicated_llm_key:
                llm_key_source = "dedicated"
            elif has_openrouter_key:
                llm_key_source = "default"
            else:
                llm_key_source = "none"
            return {
                "llm_base_url": llm_cfg.base_url,
                "llm_model": llm_cfg.model,
                "llm_timeout": llm_cfg.timeout,
                "llm_api_key_set": bool(llm_cfg.api_key),
                "llm_api_key_source": llm_key_source,
                "openrouter_api_key_set": has_openrouter_key,
                "classification_method": str(cls_cfg.get("method", "decider")),
                "fallback_to_rules": bool(cls_cfg.get("fallback_to_rules", True)),
                "decider_url": os.getenv("DECIDER_REMOTE_URL") or str(dec_cfg.get("url", "")),
                "decider_model": os.getenv("DECIDER_REMOTE_MODEL") or str(dec_cfg.get("model", "")),
                "ocr_deskew": bool(cfg.ocr.get("deskew", True)),
                "ocr_render_width_px": int(cfg.render_width_px),
                "default_workers": int(cfg.default_workers),
            }

    def save_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            cfg_path = self._effective_config_path()
            raw = json.loads(cfg_path.read_text(encoding="utf-8-sig"))

            llm_section = raw.setdefault("llm", {})
            env_updates: dict[str, str | None] = {}

            if "llm_base_url" in payload and str(payload["llm_base_url"]).strip():
                url = str(payload["llm_base_url"]).strip()
                llm_section["base_url"] = url
                os.environ["LLM_BASE_URL"] = url
                env_updates["LLM_BASE_URL"] = url
            if "llm_model" in payload and str(payload["llm_model"]).strip():
                model = str(payload["llm_model"]).strip()
                llm_section["model"] = model
                os.environ["LLM_MODEL"] = model
                env_updates["LLM_MODEL"] = model
            if "llm_timeout" in payload and payload["llm_timeout"] is not None:
                timeout = max(10.0, float(payload["llm_timeout"]))
                llm_section["timeout"] = timeout
                os.environ["LLM_TIMEOUT"] = str(timeout)

            # Write-only API keys: non-empty replaces stored value; empty preserves existing; explicit clear removes
            if payload.get("clear_llm_api_key"):
                os.environ.pop("LLM_API_KEY", None)
                env_updates["LLM_API_KEY"] = ""
            elif payload.get("llm_api_key"):
                key = str(payload["llm_api_key"]).strip()
                if key:
                    os.environ["LLM_API_KEY"] = key
                    env_updates["LLM_API_KEY"] = key

            if payload.get("clear_openrouter_api_key"):
                os.environ.pop("OPENROUTER_API_KEY", None)
                env_updates["OPENROUTER_API_KEY"] = ""
            elif payload.get("openrouter_api_key"):
                key = str(payload["openrouter_api_key"]).strip()
                if key:
                    os.environ["OPENROUTER_API_KEY"] = key
                    env_updates["OPENROUTER_API_KEY"] = key

            cls_section = raw.setdefault("classification", {})
            if "classification_method" in payload:
                method = str(payload["classification_method"]).strip().lower()
                if method not in ("decider", "llm", "rules"):
                    raise ValueError(f"invalid classification method: {method!r}")
                cls_section["method"] = method
            if "fallback_to_rules" in payload:
                cls_section["fallback_to_rules"] = bool(payload["fallback_to_rules"])
            dec_section = cls_section.setdefault("decider", {})
            if "decider_url" in payload and str(payload["decider_url"]).strip():
                d_url = str(payload["decider_url"]).strip()
                dec_section["url"] = d_url
                os.environ["DECIDER_REMOTE_URL"] = d_url
                env_updates["DECIDER_REMOTE_URL"] = d_url
            if "decider_model" in payload and str(payload["decider_model"]).strip():
                d_model = str(payload["decider_model"]).strip()
                dec_section["model"] = d_model
                os.environ["DECIDER_REMOTE_MODEL"] = d_model
                env_updates["DECIDER_REMOTE_MODEL"] = d_model

            ocr_section = raw.setdefault("ocr", {})
            if "ocr_deskew" in payload:
                ocr_section["deskew"] = bool(payload["ocr_deskew"])
            if "ocr_render_width_px" in payload and payload["ocr_render_width_px"] is not None:
                ocr_section["render_width_px"] = max(800, min(3200, int(payload["ocr_render_width_px"])))

            rt_section = raw.setdefault("runtime", {})
            if "default_workers" in payload and payload["default_workers"] is not None:
                rt_section["default_workers"] = max(1, min(16, int(payload["default_workers"])))

            cfg_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if env_updates:
                _update_dotenv(env_updates, PROJECT_ROOT / ".env")

            self._llm_config = None
            self.reload_types()
            return {"ok": True, "settings": self.get_settings()}
