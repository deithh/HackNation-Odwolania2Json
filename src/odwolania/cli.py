"""Command-line interface for Odwołania2JSON (all defaults loaded from `config/pipeline.json`)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DocumentTypeRegistry, PipelineConfig
from .evaluation import F1Evaluator
from .optimization import GepaTypeOptimizer
from .pipeline import DocumentPipeline
from .type_tools import TypeConfigScaffolder, TypeConfigValidator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="odwolania",
        description="Odwołania2JSON — config-driven PDF documents -> structured JSON.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="process every PDF in the input folder and write JSON to the output folder")
    run_cmd.add_argument("--input", default=None, help="input folder (default: runtime.input_dir in pipeline.json)")
    run_cmd.add_argument("--output", default=None, help="output folder (default: runtime.output_dir in pipeline.json)")
    run_cmd.add_argument("--config", help="pipeline.json path (default: config/pipeline.json)")
    run_cmd.add_argument("--types", help="folder with doc type JSON configs (default: runtime.types_dir)")
    run_cmd.add_argument("--force-ocr", action="store_true", help="OCR pages even if they have a PDF text layer")
    run_cmd.add_argument("--classify-only", action="store_true", help="only classify documents, skip extraction")
    run_cmd.add_argument(
        "--debug",
        nargs="?",
        const="debug",
        metavar="DIR",
        help="write step-by-step transparency artifacts to DIR (default: debug)",
    )
    run_cmd.add_argument("--report", help="write per-file run records to this JSON file")
    run_cmd.add_argument("--workers", type=int, default=None, help="parallel workers (default: runtime.default_workers)")

    types_cmd = sub.add_parser("types", help="list all configured document types from config/types/*.json")
    types_cmd.add_argument("--types", help="folder with doc type JSON configs (default: config/types)")

    val_cmd = sub.add_parser("validate-types", help="validate all document type configs in config/types/*.json")
    val_cmd.add_argument("--types", help="folder with doc type JSON configs (default: config/types)")

    init_cmd = sub.add_parser("init-type", help="scaffold a new document type JSON config (zero code changes)")
    init_cmd.add_argument("type_id", help="lowercase snake_case identifier, e.g. wyrok_kio")
    init_cmd.add_argument("--name", help="display name of the document type")
    init_cmd.add_argument("--description", default="", help="short semantic description")
    init_cmd.add_argument("--priority", type=int, default=10, help="classification priority (default: 10)")
    init_cmd.add_argument("--pages", default="1-3", help="extraction page range spec (default: 1-3)")
    init_cmd.add_argument("--types", help="target folder (default: config/types)")
    init_cmd.add_argument("--overwrite", action="store_true", help="overwrite if file already exists")

    infer_cmd = sub.add_parser(
        "infer-type",
        help="auto-generate a new document type config from a sample .json (offline) or sample .pdf (via LLM)",
    )
    infer_cmd.add_argument("sample", help="path to a sample .json output or sample .pdf document")
    infer_cmd.add_argument("--type-id", required=True, help="lowercase snake_case identifier, e.g. wyrok_kio")
    infer_cmd.add_argument("--name", help="display name of the document type")
    infer_cmd.add_argument("--description", default="", help="short semantic description (used for .json sample)")
    infer_cmd.add_argument("--priority", type=int, default=10, help="classification priority (default: 10)")
    infer_cmd.add_argument("--config", help="pipeline.json path (for PDF sample OCR)")
    infer_cmd.add_argument("--types", help="target folder (default: config/types)")
    infer_cmd.add_argument("--overwrite", action="store_true", help="overwrite if file already exists")

    eval_cmd = sub.add_parser("evaluate", help="evaluate predicted JSONs against ground truth (official UZP F1)")
    eval_cmd.add_argument("--pred", required=True, help="folder with predicted JSON files")
    eval_cmd.add_argument("--gold", required=True, help="folder with ground truth PDF/JSON files")
    eval_cmd.add_argument("--expect-empty", action="store_true", help="expect {} for every PDF (negatives set)")
    eval_cmd.add_argument("--config", help="pipeline.json path (default: config/pipeline.json)")
    eval_cmd.add_argument("--out", help="optional path to write JSON evaluation report")

    opt_cmd = sub.add_parser(
        "optimize-type",
        help="automatically optimize extraction.uwagi for any document type using GEPA reflective evolution",
    )
    opt_cmd.add_argument("type_id", help="registered type_id to optimize (existing or newly added)")
    opt_cmd.add_argument("--data", required=True, help="folder containing labeled <doc>.pdf + <doc>.json pairs")
    opt_cmd.add_argument("--config", help="pipeline.json path (default: config/pipeline.json)")
    opt_cmd.add_argument("--types", help="folder with doc type JSON configs (default: runtime.types_dir)")
    opt_cmd.add_argument("--max-metric-calls", type=int, default=None, help="GEPA evaluation budget")
    opt_cmd.add_argument("--reflection-model", default=None, help="override reflection LLM model")
    opt_cmd.add_argument("--train-ratio", type=float, default=None, help="train/val split ratio (default from JSON)")
    opt_cmd.add_argument("--seed", type=int, default=None, help="random split seed (default from JSON)")
    opt_cmd.add_argument("--optimize-system", action="store_true", help="also evolve the global extraction prompt")
    opt_cmd.add_argument("--apply", action="store_true", help="write improved extraction.uwagi back into type JSON")
    opt_cmd.add_argument("--run-dir", help="optional folder for GEPA checkpoints")
    opt_cmd.add_argument("--out", help="optional path to write JSON optimization summary")

    web_cmd = sub.add_parser("web", help="launch the local interactive web GUI (PDF highlighting + JSON + Zero-Code Types)")
    web_cmd.add_argument("--host", default=None, help="bind host (default: web.host in pipeline.json)")
    web_cmd.add_argument("--port", type=int, default=None, help="bind port (default: web.port in pipeline.json)")
    web_cmd.add_argument("--config", help="pipeline.json path (default: config/pipeline.json)")
    web_cmd.add_argument("--types", help="folder with doc type JSON configs (default: runtime.types_dir)")
    web_cmd.add_argument("--no-browser", action="store_true", help="do not auto-open the web browser")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "types":
        registry = DocumentTypeRegistry.from_dir(args.types)
        for tid, dt in registry.items():
            print(f"[{dt.priority}] {tid:24s}  {dt.name} — {dt.description}")
        return 0

    if args.command == "validate-types":
        report = TypeConfigValidator().validate_dir(args.types)
        for issue in report.issues:
            print(f"[{issue.severity.upper()}] {issue.file}: {issue.message}", file=sys.stderr)
        if report.ok:
            print(f"OK: validated {report.checked_files} document type config(s), 0 errors.")
            return 0
        print(f"FAILED: {len(report.errors)} error(s) in {report.checked_files} file(s).", file=sys.stderr)
        return 1

    if args.command == "init-type":
        scaffolder = TypeConfigScaffolder()
        draft = scaffolder.build_template(
            args.type_id,
            name=args.name,
            description=args.description,
            priority=args.priority,
            pages=args.pages,
        )
        try:
            target, report = scaffolder.save(draft, types_dir=args.types, overwrite=args.overwrite)
        except FileExistsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"Created {target} (valid={report.ok})")
        return 0 if report.ok else 1

    if args.command == "infer-type":
        sample_path = Path(args.sample)
        if not sample_path.is_file():
            print(f"error: sample file not found: {sample_path}", file=sys.stderr)
            return 2
        scaffolder = TypeConfigScaffolder()
        suffix = sample_path.suffix.lower()
        if suffix == ".json":
            sample_obj = json.loads(sample_path.read_text(encoding="utf-8-sig"))
            draft = scaffolder.infer_from_json_sample(
                sample_obj,
                args.type_id,
                name=args.name,
                description=args.description,
                priority=args.priority,
            )
        elif suffix == ".pdf":
            cfg = PipelineConfig.load(args.config)
            draft = scaffolder.infer_from_pdf_sample(
                sample_path,
                args.type_id,
                name=args.name,
                priority=args.priority,
                pipeline_config=cfg,
            )
        else:
            print(f"error: unsupported sample extension {suffix!r} (expected .json or .pdf)", file=sys.stderr)
            return 2

        try:
            target, report = scaffolder.save(draft, types_dir=args.types, overwrite=args.overwrite)
        except FileExistsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for issue in report.issues:
            print(f"[{issue.severity.upper()}] {issue.file}: {issue.message}", file=sys.stderr)
        print(f"Inferred and saved {target} (valid={report.ok})")
        return 0 if report.ok else 1

    if args.command == "evaluate":
        cfg = PipelineConfig.load(args.config)
        evaluator = F1Evaluator.from_pipeline_config(cfg)
        report = evaluator.evaluate_dir(args.pred, args.gold, expect_empty=args.expect_empty)
        ov = report["overall"]
        print(
            f"OVERALL F1={ov['f1']:.4f}  P={ov['precision']:.4f}  R={ov['recall']:.4f}  "
            f"(TP={ov['tp']} FP={ov['fp']} FN={ov['fn']} TN={ov['tn']}, files={report['files']}, "
            f"fuzzy>={report['fuzzy_threshold']:.1f}% via {report['fuzzy_backend']})"
        )
        if args.out:
            Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Report saved to: {args.out}")
        return 0

    if args.command == "optimize-type":
        cfg = PipelineConfig.load(args.config)
        registry = DocumentTypeRegistry.from_dir(args.types or cfg.types_dir)
        optimizer = GepaTypeOptimizer(cfg, registry, types_dir=args.types)
        try:
            result = optimizer.optimize(
                args.type_id,
                args.data,
                optimize_system=args.optimize_system,
                max_metric_calls=args.max_metric_calls,
                reflection_model=args.reflection_model,
                train_ratio=args.train_ratio,
                seed=args.seed,
                apply=args.apply,
                run_dir=args.run_dir,
            )
        except (ImportError, KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        print(
            f"GEPA [{result.type_id}] ({result.elapsed_seconds:.1f}s, train={result.train_count}, val={result.val_count}):\n"
            f"  SEED  train F1={result.seed_train['f1']:.4f} | val F1={result.seed_val['f1']:.4f}\n"
            f"  BEST  train F1={result.best_train['f1']:.4f} | val F1={result.best_val['f1']:.4f}"
        )
        if result.applied_to:
            print(f"  APPLIED improved extraction.uwagi to: {result.applied_to}")
        if args.out:
            Path(args.out).write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"  Report saved to: {args.out}")
        return 0

    if args.command == "web":
        from .web import WebServer, WebWorkspaceService

        service = WebWorkspaceService(config_path=args.config, types_dir=args.types)
        server = WebServer(service, host=args.host, port=args.port)
        server.serve_forever(open_browser=not args.no_browser)
        return 0

    if args.command == "run":
        try:
            pipeline = DocumentPipeline.from_paths(
                config_path=args.config,
                types_dir=args.types,
                force_ocr=args.force_ocr,
            )
            records = pipeline.process_directory(
                input_dir=args.input,
                output_dir=args.output,
                types_dir=args.types,
                classify_only=args.classify_only,
                debug_dir=args.debug,
                workers=args.workers,
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        if args.report:
            payload = [r.to_dict() for r in records]
            Path(args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1 if any(r.error for r in records) else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
