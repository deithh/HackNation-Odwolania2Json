# Odwołania2JSON

Narzędzie CLI oraz lokalny interfejs WWW do klasyfikacji dokumentów PDF (natywnych i skanów) ze spraw odwoławczych KIO oraz ekstrakcji ustrukturyzowanych danych do kanonicznych schematów JSON.

Projekt realizuje zadanie **Odwołania2JSON** (Urząd Zamówień Publicznych, HackNation 2026).

---

## Opis i architektura

Aplikacja przetwarza wielodokumentowe sprawy odwoławcze KIO (natywne PDF, skany z ukrytą warstwą tekstową oraz czyste skany) do ustrukturyzowanych plików JSON:

```text
PDF (input/<sprawa>/*.pdf)
  │
  ├─► PdfInspector (pypdfium2) ──► analiza warstwy tekstowej, pokrycia obrazem i ukrytego OCR
  │       ├── tekst natywny / warstwa OCR ──► odczyt bezpośredni
  │       └── skan ──► deskew (expand=True) + crop_content_region ──► RapidOCR (PP-OCRv5 CPU) + cache
  │
  ├─► DocumentClassifier ──► klasyfikacja po treści (decider | llm | rules fallback)
  │       └── typ nierozpoznany ("other") lub błąd przetwarzania ──► zapis "{}"
  │
  └─► DocumentExtractor (OpenAI-compatible API + JSON Schema)
          └── SchemaPostprocessor + SourceTextIndex ──► normalizacja, uziemienie ID, lookup_bindings
                  │
                  ▼
          JSON (output/<sprawa>/*.json)
```

- **Konfiguracja bez zmian kodu:** typy dokumentów (`odwolanie`, `pelnomocnictwo`, `przelew`) definiują pliki JSON w [config/types/](config/types/) walidowane względem [config/type.schema.json](config/type.schema.json).
- **Lokalny OCR na CPU:** `RapidOCR` na `onnxruntime` (`PP-OCRv5 mobile`, `latin`) z rekonstrukcją układu linii `(y, x)`, prostowaniem stron (`expand=True`, bez ucinania rogów) i wycinaniem pustych marginesów (`crop_content_region`, np. skan A5 na A4).
- **Uziemienie w źródle (`SourceTextIndex`):** weryfikacja `KRS`, `NIP`, `REGON`, `PESEL`, `BZP`, `TED` oraz `NRB`/`IBAN` (z zachowaniem prefiksów krajowych `PL`/`DE` itd.) względem tekstu dokumentu oraz synchronizacja słowników (`lookup_bindings`, np. `czynnosc_powod_id` ↔ `czynnosc_opis`).

---

## Technologie i modele

- **Język i runtime:** Python 3.13+
- **Przetwarzanie PDF i obrazu:** `pypdfium2`, `Pillow`, `numpy`
- **OCR (lokalny, CPU):** `rapidocr` (`PP-OCRv5 mobile`, `ch_PP-OCRv5_det_mobile.onnx` + `latin_PP-OCRv5_rec_mobile.onnx`), `onnxruntime`
- **Ekstrakcja LLM (API kompatybilne z OpenAI):** otwarto-źródłowy model **Qwen3.5-9B** (~9 mld parametrów, `qwen3.5-9b` w Regolo.ai, `temperature=0.0`, `response_format` JSON Schema)
- **Klasyfikacja:** konfigurowalna w [config/pipeline.json](config/pipeline.json) (`decider` / `llm` / `rules`) z automatycznym fallbackiem do deterministycznego klasyfikatora cech formatowych (`RuleClassifier`)
- **Interfejs WWW:** wbudowany serwer oparty wyłącznie na bibliotece standardowej Pythona (`http.server`)

---

## Wymagania

- Python **3.13+**
- Zależności z [requirements.txt](requirements.txt) (instalacja przez `pip` / `uv`)
- Dostęp do internetu oraz klucz API w pliku `.env` dla wywołania modelu LLM
- Brak wymagań dotyczących GPU

---

## Instalacja i konfiguracja

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

Skopiuj szablon zmiennych środowiskowych i uzupełnij klucz API:

```bash
cp .env.example .env
```

Najważniejsze zmienne w `.env`:

- `LLM_BASE_URL` — adres endpointu kompatybilnego z OpenAI (domyślnie `https://api.regolo.ai/v1`)
- `LLM_MODEL` — identyfikator modelu (domyślnie `qwen3.5-9b`)
- `LLM_API_KEY` — klucz API dla endpointu LLM
- `DECIDER_REMOTE_URL` / `DECIDER_REMOTE_MODEL` / `DECIDER_API_KEY` — endpoint, model (`clef-flash`) i klucz dla klasyfikatora `decider` (fallback do `LLM_API_KEY` / `OPENROUTER_API_KEY`)

Parametry przetwarzania (progi klasyfikacji stron, liczba workerów, wątki ONNX, reguły post-processingu, progi F1) są zdefiniowane w [config/pipeline.json](config/pipeline.json) i [config/prompts.json](config/prompts.json). Wszystkie pliki JSON są wczytywane z obsługą `utf-8-sig` (tolerancja dla UTF-8 BOM).

---

## Uruchomienie

### Tryb wsadowy (CLI: `input/` → `output/`)

Umieść pliki PDF w katalogu `input/` (dowolna struktura podkatalogów spraw) i uruchom:

```bash
# Linux / macOS:
bash start.sh --workers 2

# Windows:
start.bat

# Bezpośrednio przez Pythona:
python main.py --input input --output output --workers 4
```

Dla środowiska produkcyjnego na VM (`/opt/solution`, Ubuntu 24.04 LTS, 2 vCPU) zobacz [RUN.md](RUN.md).

### Tryb interaktywny (Web GUI)

```bash
# Linux / macOS:
bash start_gui.sh

# Windows:
start_gui.bat

# Bezpośrednio:
python main.py web
```

Interfejs startuje pod `http://127.0.0.1:8000` (konfigurowalne przez `web.host` i `web.port` w `config/pipeline.json`) i udostępnia wgrywanie PDF-ów, dwukierunkowe podświetlanie pól na stronie PDF/skanu, podgląd transparentności klasyfikacji/ekstrakcji oraz edytor typów dokumentów bez zmian w kodzie.

---

## Użytkowanie (CLI)

Dostępne podkomendy `python main.py <komenda>`:

| Komenda | Przeznaczenie |
|---|---|
| `run` (domyślna) | Przetwarza pliki PDF z `--input` do `--output` (`--workers`, `--debug`, `--report`, `--force-ocr`, `--classify-only`) |
| `web` | Uruchamia lokalny serwer Web GUI |
| `types` | Wypisuje aktywne typy dokumentów z `config/types/` |
| `validate-types` | Waliduje konfiguracje typów względem `config/type.schema.json` |
| `init-type <id>` | Tworzy szkielet nowego pliku konfiguracyjnego typu w `config/types/` |
| `infer-type <plik>` | Generuje definicję typu z przykładowego `.json` (offline) lub `.pdf` (przez LLM) |
| `evaluate` | Liczy metrykę F1 między `--pred` a `--gold` (`--expect-empty` dla zbiorów negatywnych) |
| `optimize-type <id>` | Optymalizuje wskazówki ekstrakcji (`extraction.uwagi`) metodą GEPA względem F1 |

Przykłady:

```bash
python main.py --debug --report report.json       # zapis artefaktów pośrednich (01_text..05_output) do ./debug
python main.py validate-types                      # walidacja schematów typów
python main.py evaluate --pred output --gold dane  # pomiar F1
```

Dodanie 4. typu dokumentu bez edycji kodu (przykład z [config/types_examples/](config/types_examples/)):

```bash
cp config/types_examples/odpowiedz_na_odwolanie.json config/types/
python main.py validate-types
```

---

## Struktura projektu

```text
odwolania2json/
├── config/
│   ├── pipeline.json          # Parametry OCR, LLM, klasyfikacji, post-processingu, F1 i GUI
│   ├── prompts.json           # Szablony promptów (ekstrakcja, klasyfikacja, infer-type, GEPA)
│   ├── type.schema.json       # Meta-schemat walidacji konfiguracji typów
│   ├── types/                 # Aktywne definicje klas dokumentów (*.json)
│   └── types_examples/        # Przykładowe dodatkowe typy do aktywacji przez skopiowanie
├── src/odwolania/
│   ├── cli.py                 # Parser CLI i obsługa podkomend
│   ├── config.py              # Ładowanie konfiguracji i rejestru typów (utf-8-sig)
│   ├── pipeline.py            # Orkiestracja przetwarzania (sekwencyjna i ProcessPoolExecutor)
│   ├── evaluation.py          # Kalkulator metryki F1 (dokładne dopasowanie + fuzzy dla długich pól)
│   ├── optimization.py        # Integracja optymalizatora GEPA (opcjonalna)
│   ├── type_tools.py          # Walidator i generator konfiguracji typów
│   ├── ocr/                   # PdfInspector, RapidOcrEngine, deskew/crop, OcrCache, DocumentReader
│   ├── classification/        # DeciderClassifier, LLMClassifier, RuleClassifier, FallbackClassifier
│   ├── extraction/            # OpenAICompatibleClient, PromptBuilder, SourceTextIndex, SchemaPostprocessor
│   └── web/                   # Lokalny serwer HTTP, evidence locator i statyczny frontend
├── tests/                     # Testy jednostkowe i integracyjne (unittest / pytest)
├── input/                     # Katalog wejściowy
├── output/                    # Katalog wyjściowy
├── main.py                    # Punkt wejścia CLI
├── RUN.md                     # Instrukcja uruchomienia automatycznego na VM (/opt/solution)
├── start.sh / start.bat       # Skrypty uruchomienia trybu wsadowego
├── start_gui.sh / start_gui.bat
├── requirements.txt           # Zależności produkcyjne (CPU-only)
└── requirements-dev.txt       # Zależności opcjonalne (gepa, rapidfuzz, pytest)
```

---

## Testowanie i zmierzone wyniki

### Testy automatyczne

Zestaw testów w [tests/test_core.py](tests/test_core.py) korzysta z biblioteki standardowej `unittest` (jest też kompatybilny z `pytest` z `requirements-dev.txt`):

```bash
python -m unittest discover -s tests -v
python main.py validate-types
```

### Wyniki na zbiorze skanów (`zadanie/skany`, 30 skanów PDF, bez cache OCR)

| Konfiguracja | Czas łączny | Śr. czas / plik | F1 | Precision | Recall |
|---|---|---|---|---|---|
| `--workers 4`, `intra_op_num_threads=4` | 134,69 s | 4,41 s | 0,9803 | 0,9645 | 0,9967 |
| `--workers 2`, `intra_op_num_threads=1` | 426,39 s | 14,14 s | 0,9755 | 0,9553 | 0,9967 |

W obu przebiegach przetworzono 30/30 plików bez błędów wykonania (`rapidfuzz`, próg `80.0%` dla pól typu długi string).
