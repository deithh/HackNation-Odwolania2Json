"""End-to-end document processing pipeline: PDF -> OCR/Text -> Classification -> Extraction -> JSON."""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TextIO

from tqdm import tqdm

from .classification import OTHER, ClassifierFactory, DocumentClassifier
from .config import DocumentTypeRegistry, PipelineConfig, resolve_path
from .extraction import DocumentExtractor, LLMConfig, OpenAICompatibleClient
from .models import ProcessingRecord
from .ocr import DocumentReader
from .utils.debug import DebugSink

SUPPORTED_EXTENSIONS = {".pdf"}

# Process-local pipeline instance for multiprocessing workers (avoids reloading ONNX models per file).
_WORKER_PIPELINE: "DocumentPipeline | None" = None
_WORKER_KEY: tuple | None = None


def _get_worker_pipeline(
    config: PipelineConfig,
    types_dir: str | None,
    llm_config: LLMConfig | None,
) -> "DocumentPipeline":
    global _WORKER_PIPELINE, _WORKER_KEY
    key = (
        json.dumps(
            {k: v for k, v in config.__dict__.items() if k != "prompts"},
            sort_keys=True,
        ),
        config.prompts,
        types_dir,
        llm_config,
    )
    if _WORKER_PIPELINE is None or _WORKER_KEY != key:
        registry = DocumentTypeRegistry.from_dir(types_dir or config.types_dir)
        _WORKER_PIPELINE = DocumentPipeline(config, registry, llm_config=llm_config)
        _WORKER_KEY = key
    return _WORKER_PIPELINE


def _process_file_worker(args: tuple) -> tuple[int, ProcessingRecord]:
    idx, path, input_dir, output_dir, config, types_dir, llm_config, classify_only, debug_dir = args
    pipeline = _get_worker_pipeline(config, types_dir, llm_config)
    record = pipeline.process_file(
        path,
        input_dir,
        output_dir,
        classify_only=classify_only,
        debug_dir=debug_dir,
    )
    return idx, record


class DocumentPipeline:
    """Coordinates PDF reading, classification, schema-driven extraction, and JSON output."""

    def __init__(
        self,
        config: PipelineConfig,
        registry: DocumentTypeRegistry,
        *,
        llm_config: LLMConfig | None = None,
        reader: DocumentReader | None = None,
        classifier: DocumentClassifier | None = None,
        extractor: DocumentExtractor | None = None,
    ):
        self.config = config
        self.registry = registry
        self._llm_config = llm_config or LLMConfig.from_pipeline_config(config)
        self._llm_client = OpenAICompatibleClient(self._llm_config)
        self.reader = reader or DocumentReader(config)
        self.classifier = classifier or ClassifierFactory.create(
            config.classification,
            prompts=config.prompts,
            llm_client=self._llm_client,
            llm_config=self._llm_config,
        )
        self.extractor = extractor or DocumentExtractor(
            client=self._llm_client,
            pipeline_config=config,
        )

    @classmethod
    def from_paths(
        cls,
        *,
        config_path: str | Path | None = None,
        types_dir: str | Path | None = None,
        force_ocr: bool = False,
        llm_config: LLMConfig | None = None,
    ) -> "DocumentPipeline":
        config = PipelineConfig.load(config_path)
        if force_ocr:
            config = config.with_force_ocr(True)
        registry = DocumentTypeRegistry.from_dir(types_dir or config.types_dir)
        return cls(config, registry, llm_config=llm_config)

    @staticmethod
    def iter_inputs(input_dir: Path) -> Iterator[Path]:
        for path in sorted(input_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith("."):
                yield path

    def process_file(
        self,
        path: Path,
        input_dir: Path,
        output_dir: Path,
        *,
        classify_only: bool = False,
        debug_dir: Path | None = None,
    ) -> ProcessingRecord:
        """Processes a single PDF. Never raises: errors are recorded in `ProcessingRecord.error`."""
        rel = path.relative_to(input_dir)
        record = ProcessingRecord(file=str(rel))
        dbg = DebugSink(debug_dir, rel)
        t0 = time.perf_counter()

        try:
            doc = self.reader.read(path)
            record.strategy = doc.strategy
            record.strategy_reason = doc.strategy_reason
            record.n_pages = doc.n_pages
            dbg.write("01_text.json", doc.to_dict())
            dbg.write("01_text.txt", doc.text_for())
            if doc.strategy == "error":
                raise RuntimeError(doc.strategy_reason)

            cls_result = self.classifier.classify(doc.text_for(), self.registry)
            cls_dict = cls_result.to_dict()
            dbg.write("02_classification.json", cls_dict)
            record.doc_type = cls_result.doc_type
            record.classification = cls_dict

            if classify_only:
                record.seconds = round(time.perf_counter() - t0, 2)
                return record

            out_path = (output_dir / rel).with_suffix(".json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            doc_type = self.registry.get(cls_result.doc_type)

            if doc_type is None:
                # Unrecognized / out-of-scope document -> write empty JSON object `{}` per UZP spec.
                dbg.write("05_output.json", {})
                out_path.write_text("{}\n", encoding="utf-8")
                record.output = str(out_path.relative_to(output_dir))
                record.seconds = round(time.perf_counter() - t0, 2)
                return record

            pages_spec = doc_type.extraction_pages
            extraction_text = self.reader.read(path, pages_spec).text_for() if pages_spec else doc.text_for()
            result = self.extractor.extract(extraction_text, doc_type)

            dbg.write("03_extraction_system_prompt.txt", result.trace.system_prompt)
            dbg.write("03_extraction_input_text.txt", extraction_text)
            dbg.write("04_extraction_raw.txt", result.trace.raw_output)
            dbg.write("05_output.json", result.data)

            out_path.write_text(
                json.dumps(result.data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            record.output = str(out_path.relative_to(output_dir))
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            record.traceback = traceback.format_exc()
            if not classify_only:
                out_path = (output_dir / rel).with_suffix(".json")
                if not out_path.exists():
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text("{}\n", encoding="utf-8")
                    record.output = str(out_path.relative_to(output_dir))

        record.seconds = round(time.perf_counter() - t0, 2)
        return record

    def process_directory(
        self,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        *,
        types_dir: str | Path | None = None,
        classify_only: bool = False,
        debug_dir: str | Path | None = None,
        workers: int | None = None,
        log: TextIO = sys.stderr,
    ) -> list[ProcessingRecord]:
        """Processes all PDFs under `input_dir` into mirrored JSON files under `output_dir`."""
        in_path = resolve_path(input_dir if input_dir is not None else self.config.input_dir)
        out_path = resolve_path(output_dir if output_dir is not None else self.config.output_dir)
        if not in_path.is_dir():
            raise FileNotFoundError(f"input folder not found: {in_path}")

        dbg_path = resolve_path(debug_dir) if debug_dir else None
        files = list(self.iter_inputs(in_path))
        req_workers = self.config.default_workers if workers is None else workers
        num_workers = max(1, min(int(req_workers), len(files) or 1))
        print(f"{len(files)} PDF file(s) in {in_path} (workers={num_workers})", file=log)

        bar = tqdm(total=len(files), desc="documents", unit="file", file=log, dynamic_ncols=True)
        if num_workers == 1:
            records: list[ProcessingRecord] = []
            for pdf_path in files:
                bar.set_postfix_str(pdf_path.name[:40])
                rec = self.process_file(
                    pdf_path,
                    in_path,
                    out_path,
                    classify_only=classify_only,
                    debug_dir=dbg_path,
                )
                records.append(rec)
                self._log_record(bar, rec)
                bar.update(1)
        else:
            ordered: list[ProcessingRecord | None] = [None] * len(files)
            resolved_types = str(resolve_path(types_dir)) if types_dir else None
            tasks = [
                (i, p, in_path, out_path, self.config, resolved_types, self._llm_config, classify_only, dbg_path)
                for i, p in enumerate(files)
            ]
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = [pool.submit(_process_file_worker, t) for t in tasks]
                for future in as_completed(futures):
                    idx, rec = future.result()
                    ordered[idx] = rec
                    self._log_record(bar, rec)
                    bar.update(1)
            records = [r for r in ordered if r is not None]
        bar.close()

        errors = sum(1 for r in records if r.error)
        skipped = sum(1 for r in records if not r.error and r.doc_type == OTHER)
        if classify_only:
            counts = {
                t: sum(1 for r in records if r.doc_type == t)
                for t in sorted({r.doc_type or "unknown" for r in records})
            }
            print(f"classified (no extraction): {counts}, {errors} failed", file=log)
        else:
            extracted = sum(1 for r in records if r.output and r.doc_type != OTHER)
            print(f"done: {extracted} extracted, {skipped} empty ({{}}), {errors} failed", file=log)
        if dbg_path:
            print(f"debug files: {dbg_path}", file=log)
        return records

    @staticmethod
    def _log_record(bar: tqdm, rec: ProcessingRecord) -> None:
        status = rec.error or (f"-> {rec.output}" if rec.output else "no JSON")
        bar.write(f"{rec.file}  type={rec.doc_type}  {rec.seconds}s  {status}")
