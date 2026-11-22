"""Automated unit and integration tests for Odwołania2JSON (stdlib unittest, pytest-compatible)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from odwolania.classification import OTHER, DocumentClassifier, RuleClassifier  # noqa: E402
from odwolania.config import DocumentTypeRegistry, PipelineConfig, PromptTemplates  # noqa: E402
from odwolania.evaluation import F1Evaluator  # noqa: E402
from odwolania.extraction.grounding import SourceTextIndex  # noqa: E402
from odwolania.extraction.postprocessor import SchemaPostprocessor  # noqa: E402
from odwolania.models import ClassificationResult, DocumentText, PageText  # noqa: E402
from odwolania.ocr.deskew import crop_content_region, deskew_image  # noqa: E402
from odwolania.pipeline import DocumentPipeline  # noqa: E402
from odwolania.type_tools import TypeConfigValidator  # noqa: E402
from odwolania.utils.pages import resolve_pages  # noqa: E402


class TestPagesAndConfig(unittest.TestCase):
    def test_resolve_pages_specs(self) -> None:
        self.assertEqual(resolve_pages("1-3,last", 10), [1, 2, 3, 10])
        self.assertEqual(resolve_pages("1-3,last", 2), [1, 2])
        self.assertEqual(resolve_pages("last-1", 5), [4])
        self.assertEqual(resolve_pages("all", 3), [1, 2, 3])
        self.assertEqual(resolve_pages(2, 5), [1, 2])

    def test_config_loads_with_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cfg_src = (ROOT / "config" / "pipeline.json").read_text(encoding="utf-8")
            bom_cfg = tmp_dir / "pipeline.json"
            bom_cfg.write_bytes(b"\xef\xbb\xbf" + cfg_src.encode("utf-8"))
            cfg = PipelineConfig.load(bom_cfg)
            self.assertEqual(cfg.ocr_engine_name, "rapid")

            prompts_src = (ROOT / "config" / "prompts.json").read_text(encoding="utf-8")
            bom_prompts = tmp_dir / "prompts.json"
            bom_prompts.write_bytes(b"\xef\xbb\xbf" + prompts_src.encode("utf-8"))
            prompts = PromptTemplates.load(bom_prompts)
            self.assertTrue(bool(prompts.extraction_system_prompt))

            types_dir = tmp_dir / "types"
            types_dir.mkdir()
            przelew_src = (ROOT / "config" / "types" / "przelew.json").read_text(encoding="utf-8")
            (types_dir / "przelew.json").write_bytes(b"\xef\xbb\xbf" + przelew_src.encode("utf-8"))
            reg = DocumentTypeRegistry.from_dir(types_dir)
            self.assertIn("przelew", reg)
            report = TypeConfigValidator().validate_dir(types_dir)
            self.assertTrue(report.ok, report.issues)

    def test_builtin_type_configs_valid(self) -> None:
        report = TypeConfigValidator().validate_dir(ROOT / "config" / "types")
        self.assertTrue(report.ok, report.issues)
        self.assertGreaterEqual(report.checked_files, 3)


class TestPostprocessingAndGrounding(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = DocumentTypeRegistry.from_dir(ROOT / "config" / "types")
        self.pp = SchemaPostprocessor(PipelineConfig.load().postprocessing)

    def test_iban_preserves_pl_and_foreign_country_codes(self) -> None:
        przelew = self.registry.get("przelew")
        assert przelew is not None
        src_text = (
            "Nadawca: DE89 3704 0044 0532 0130 00\n"
            "Odbiorca: PL60 1010 1010 0081 3622 3100 0000\n"
            "Kwota: 15000,00 PLN Data: 10.10.2025\n"
        )
        raw = {
            "przelew_bankowy": {
                "data_operacji": "10.10.2025",
                "kwota": "15 000,00",
                "na_rachunek": "PL60 1010 1010 0081 3622 3100 0000",
                "numer_transakcji": None,
                "tytul": "Wpis od odwołania",
                "z_rachunku": "DE89 3704 0044 0532 0130 00",
            }
        }
        cleaned = self.pp.process(raw, przelew, source_text=src_text)
        pb = cleaned["przelew_bankowy"]
        self.assertEqual(pb["z_rachunku"], "DE89370400440532013000")
        self.assertEqual(pb["na_rachunek"], "PL60101010100081362231000000")
        self.assertEqual(pb["data_operacji"], "2025-10-10")
        self.assertEqual(pb["kwota"], 15000.0)

    def test_nrb_without_pl_stays_26_digits(self) -> None:
        przelew = self.registry.get("przelew")
        assert przelew is not None
        src_text = "Rachunek: 60 1010 1010 0081 3622 3100 0000\n"
        raw = {
            "przelew_bankowy": {
                "data_operacji": "2025-10-10",
                "kwota": 100.0,
                "na_rachunek": "60 1010 1010 0081 3622 3100 0000",
                "numer_transakcji": None,
                "tytul": "Opłata",
                "z_rachunku": None,
            }
        }
        cleaned = self.pp.process(raw, przelew, source_text=src_text)
        self.assertEqual(cleaned["przelew_bankowy"]["na_rachunek"], "60101010100081362231000000")

    def test_lookup_binding_syncs_czynnosc_opis(self) -> None:
        odw = self.registry.get("odwolanie")
        assert odw is not None
        raw = {
            "OdwolanieKIO": {
                "data": "2025-10-13",
                "czynnosc_dzien": "2025-10-01",
                "czynnosc_powod_id": 6,
                "czynnosc_opis": "Odpowiedzi i wyjaśnienia do SWZ",
                "miejscowosc": "Warszawa",
            },
            "Odwolujacy": [],
            "Zamawiajacy": [],
            "ZamowieniePubliczne": {},
        }
        cleaned = self.pp.process(raw, odw, source_text="Warszawa, 13.10.2025")
        self.assertEqual(cleaned["OdwolanieKIO"]["czynnosc_opis"], "Odpowiedzi/wyjaśnienia do SWZ")


class TestClassificationAndEvaluation(unittest.TestCase):
    def test_rule_classifier_detects_przelew_and_other(self) -> None:
        reg = DocumentTypeRegistry.from_dir(ROOT / "config" / "types")
        clf = RuleClassifier()
        przelew_txt = (
            "Potwierdzenie przelewu\n"
            "Z rachunku: 11 1140 2004 0000 3002 0135 5387\n"
            "Na rachunek: PL60 1010 1010 0081 3622 3100 0000\n"
            "Kwota: 15000,00 PLN\n"
            "Data operacji: 12.03.2026\n"
        )
        self.assertEqual(clf.classify(przelew_txt, reg).doc_type, "przelew")
        self.assertEqual(clf.classify("Zwykłe pismo bez danych finansowych.", reg).doc_type, OTHER)

    def test_f1_evaluator_exact_and_fuzzy(self) -> None:
        ev = F1Evaluator(fuzzy_threshold=80.0, long_string_fields={"ZamowieniePubliczne.tytul"})
        gold = {"ZamowieniePubliczne": {"tytul": "Dostawa sprzętu komputerowego dla urzędu", "numer_bzp": "2025/BZP 00012345"}}
        pred = {"ZamowieniePubliczne": {"tytul": "Dostawa sprzętu komputerowego dla urzędu.", "numer_bzp": "2025/BZP 00012345"}}
        bucket = ev.compare(gold, pred)
        self.assertEqual(bucket.total.tp, 2)
        self.assertEqual(bucket.total.fp, 0)
        self.assertEqual(bucket.total.fn, 0)
        self.assertAlmostEqual(bucket.total.f1, 1.0)


class _StubReader:
    def read(self, path: Path, pages=None) -> DocumentText:
        return DocumentText(
            file=str(path),
            sha1="deadbeef",
            strategy="text",
            strategy_reason="stub",
            n_pages=1,
            pages=[PageText(index=1, source="pdf", text="Przykładowa treść")],
            requested=[1],
        )


class _FailingClassifier(DocumentClassifier):
    def classify(self, text: str, registry: DocumentTypeRegistry) -> ClassificationResult:
        raise RuntimeError("simulated API outage")


class TestPipelineResilience(unittest.TestCase):
    def test_writes_empty_json_on_processing_error(self) -> None:
        cfg = PipelineConfig.load()
        reg = DocumentTypeRegistry.from_dir(ROOT / "config" / "types")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_dir = tmp_path / "input"
            out_dir = tmp_path / "output"
            (in_dir / "case1").mkdir(parents=True)
            pdf_file = in_dir / "case1" / "doc.pdf"
            pdf_file.write_bytes(b"%PDF-1.4\n")

            pipe = DocumentPipeline(
                cfg,
                reg,
                reader=_StubReader(),  # type: ignore[arg-type]
                classifier=_FailingClassifier(),
            )
            rec = pipe.process_file(pdf_file, in_dir, out_dir)
            self.assertIsNotNone(rec.error)
            out_file = out_dir / "case1" / "doc.json"
            self.assertTrue(out_file.is_file())
            self.assertEqual(json.loads(out_file.read_text(encoding="utf-8")), {})

    def test_deskew_expand_preserves_corners(self) -> None:
        from PIL import Image

        img = Image.new("RGB", (200, 300), (255, 255, 255))
        # Dark marker near top-left corner that expand=False would clip on rotation
        for x in range(2, 10):
            for y in range(2, 10):
                img.putpixel((x, y), (0, 0, 0))
        rot, applied = deskew_image(img, angle=5.0)
        self.assertEqual(applied, 5.0)
        self.assertGreater(rot.width, img.width)
        self.assertGreater(rot.height, img.height)
        # Ensure dark marker pixels are still present in the expanded image
        has_dark = any(px[0] < 50 for px in rot.getdata())
        self.assertTrue(has_dark)

    def test_crop_content_region_isolates_a5_in_a4(self) -> None:
        from PIL import Image, ImageDraw

        # Simulate A5 block centered inside a large white A4 page (1000x1400)
        a4 = Image.new("RGB", (1000, 1400), (255, 255, 255))
        draw = ImageDraw.Draw(a4)
        for y in range(450, 950, 25):
            draw.rectangle([300, y, 700, y + 8], fill=(20, 20, 20))
        cropped, box = crop_content_region(a4)
        self.assertLess(cropped.width, a4.width * 0.65)
        self.assertLess(cropped.height, a4.height * 0.55)
        self.assertLessEqual(box[0], 300)
        self.assertGreaterEqual(box[2], 700)


if __name__ == "__main__":
    unittest.main()
