# Automated Document PII Redaction System

> Detects and redacts personally identifiable information (PII) from document images — Aadhaar cards, PAN cards, business cards, even newspaper scans — and maps every detected entity back to its exact pixel region for **auditable** redaction.


![Python](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue)
![Platform](https://img.shields.io/badge/runs%20on-CPU%20(no%20GPU)-success)

The design goal that drives the whole system:

```
OCR text  →  entity  →  entity type  →  OCR token  →  bounding box  →  image region
```

Every redaction can be traced back through this chain — which detector flagged it, which text it matched, and which pixels were covered. Detecting PII is the easy half; getting the coordinates right and keeping the decision explainable is the hard, interesting half.

![High-level architecture](docs/hld_architecture.png)

## Project status

| Phase | Stage | Status |
|---|---|---|
| 1–2 | Preprocessing + OCR | ✅ Done |
| 3 | Text reconstruction (word tokens, char-offset ↔ token map) | ✅ Done |
| 4 | Regex detection + format validators (email, phone, URL, UPI, date, Aadhaar, PAN, IFSC) | ✅ Done |
| 5 | Name dictionary (trie, fuzzy) + context rules (Name:, Mr., S/O, DOB, A/c No., Aadhaar No.) | ✅ Done |
| 6 | NER model: GLiNER small, int8 ONNX (person, address, organization) | ✅ Done |
| 7 | Entity resolution (noisy-OR scoring, address merging, overlap handling) + spatial mapping (sub-word, one box per line) | ✅ Done |
| 8 | Redaction: opaque fill by label policy, pixel check, OCR leak check | ✅ Done |
| 9 | Streamlit UI | ✅ Upload → redacted image, entity table, safety checks, downloads |

The redacted image is produced **only after a document passes through every stage**. If any stage fails (for example, the NER model cannot load), no redacted image is produced at all.

| Original | Redacted |
|---|---|
| ![original](sample.jpg) | ![redacted](docs/sample.redacted.jpg) |

---

## Why this project

Organisations handle large volumes of scanned documents containing PII — names, phone numbers, IDs, bank details. Manual redaction doesn't scale and is error-prone, where a **single missed field is a data leak**. Many off-the-shelf tools are black boxes: they hide something, but can't explain *why*, which is a problem in any auditable or regulated setting.

This system aims to **detect PII reliably, redact it on the actual image, and make every decision explainable** — running fully offline on a CPU, so documents never leave the machine.

## Key features

- **Hybrid PII detection** — combines regex, a NER model, name dictionaries (tries), and context rules instead of relying on any single method. Structured data (emails, phones, UPI IDs) and unstructured data (names) are each handled by the technique that suits them.
- **Full audit trail** — `result.json` records, per entity: the text, label, confidence, bounding box(es), source tokens, and which detectors fired. Nothing is a black box.
- **Precision-first** — the system must handle a newspaper as gracefully as an ID card, so name detection requires agreement of multiple signals rather than a single dictionary hit, avoiding a page lit up with false positives.
- **Exact coordinate mapping** — preprocessing (upscale / deskew) is tracked in a single affine matrix and inverted, so every box maps back to the *original* image and redactions land precisely.
- **Configurable by design** — the entity registry, regex patterns, resolver weights, and colours all live in one `config.yaml`. Nothing important is hard-coded.
- **Swappable models** — OCR and NER sit behind interfaces, so backends can be replaced without touching the pipeline.
- **CPU-friendly** — runs a full document in a few seconds on a laptop with no GPU.



## Tech stack

| Component | Choice | Why |
|---|---|---|
| OCR | **RapidOCR + ONNX Runtime** (PP-OCR models) | Lightweight, CPU-fast, painless on Windows (sidesteps heavy PaddlePaddle setup) |
| NER | **GLiNER small v2.1**, int8 ONNX (183 MB) | Fast on CPU (~0.1 s per document), and labels are plain-English prompts chosen at inference time, which fits the configurable entity registry |
| Structured detection | **Regex** + format validators | Aadhaar (Verhoeff checksum), PAN holder type, Indian mobile format, real calendar dates, known UPI handles: a pass boosts confidence, a fail lowers it, and nothing is dropped |
| Name detection | **Custom Trie** | O(length) exact lookups over first / middle / last-name lists, plus edit-distance-1 fuzzy search with branch pruning for OCR slips |
| Image ops | **OpenCV** | Preprocessing, annotation, redaction |
| Config | **YAML** | Single source of truth for all tunables |
| UI | **Streamlit** (FastAPI optional later) | Fast demo UI, backend kept separable |
| Language | **Python 3.9–3.12** | RapidOCR does not support 3.13 yet |

## How it works

1. **Preprocessing (OpenCV)** — upscale small scans, denoise (edge-preserving), and deskew, tracking every geometric change in one affine matrix.
2. **OCR (RapidOCR / PP-OCR)** — produces text lines with confidence and bounding boxes, mapped back to original image coordinates.
3. **Text reconstruction** — splits OCR lines into word tokens with interpolated boxes and builds a character-offset ↔ token map.
4. **Hybrid detection** — regex, GLiNER, name tries, and context rules each emit entity candidates.
5. **Entity resolution** — aggregates overlapping candidates, combines confidence using configurable weights, and de-conflicts.
6. **Spatial mapping** — maps each entity span back through tokens to per-line bounding boxes (multi-line safe).
7. **Output** — annotated image, redacted image, and an auditable `result.json`.




## Getting started

### Prerequisites
- Python 3.9–3.12
- No GPU required

### Installation

```bash
git clone https://github.com/Krishna-Kumar-Rathore/Automated-Document-PII-Redaction-System.git
cd Automated-Document-PII-Redaction-System
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

> **Models:** the OCR models ship inside the `rapidocr-onnxruntime` wheel. The NER model (GLiNER small, int8 ONNX, **186 MB**) downloads once on first use into `models/`, which git ignores. After that everything runs fully offline. `requirements.txt` installs the **CPU-only** PyTorch build, which GLiNER needs.

### Web UI

```bash
streamlit run app/ui.py
```

Upload a JPG/PNG (or click **try the synthetic sample**). The page shows the **original and redacted images side by side** with a download button, the results of the pixel check and OCR leak check, a table of final entities (what was redacted, what was left visible and why), each stage's status and timing, and the intermediate images (line boxes, word boxes, annotated entities). The raw detector evidence and a downloadable `result.json` are there too. Uploads are processed in memory and never written to disk.

### Command line

```bash
python -m app.main --image sample.jpg            # --outdir defaults to paths.outputs_dir
```

This produces:
- `outputs/sample.result.json`: the full audit trail (stage statuses, OCR lines, words, every detector candidate, final entities with evidence and boxes, redaction summary, leak check)
- `outputs/sample.ocr_lines.jpg`: OCR line boxes
- `outputs/sample.ocr_words.jpg`: word boxes from text reconstruction
- `outputs/sample.entities.jpg`: final entities annotated with label and confidence
- `outputs/sample.redacted.jpg`: **the redacted document**

### Running tests

```bash
pytest -q
```

## Deploying the UI (Streamlit Community Cloud)

1. Push the repo to GitHub.
2. At [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and choose **Create app** → **Deploy a public app from GitHub**.
3. Set **Repository** to this repo, **Branch** to `main`, and **Main file path** to `app/ui.py`.
4. Under **Advanced settings**, choose **Python 3.12** (the platform default) or 3.11. Both were verified to resolve `requirements.txt` on Linux. Don't pick 3.13, because RapidOCR doesn't support it.
5. Click **Deploy**. The first build installs dependencies, which takes several minutes. The first page load then downloads the NER model (186 MB) and shows "Loading models…".

The repo already contains what the platform needs:
- `requirements.txt`: dependencies pinned to the versions the pipeline was verified with, and CPU-only PyTorch (verified to resolve with no CUDA packages for Linux on Python 3.11 and 3.12)
- `packages.txt`: `libgl1` and `libglib2.0-0` for OpenCV on Linux
- `.streamlit/config.toml`: 10 MB upload limit, usage telemetry off

The same files work on any Docker or VM host: `streamlit run app/ui.py --server.port 8501`.

> **Memory:** measured on the sample document, the pipeline uses about **630 MB** steady and **810 MB at peak** with all models loaded. Most of that is PyTorch, which GLiNER needs, rather than the model itself. Streamlit Community Cloud's documented limit is up to about **2.7 GB** (figures as of February 2024; they say limits may change), so this fits with NER on. On a smaller host, set `ner.enabled: false` (the NER stage is then reported as *skipped*).

> **Sleep:** Community Cloud puts apps with no traffic for 12 hours to sleep. The next visitor wakes it, and the models download and load again on that first visit.

> **Privacy note for public deployments:** the app never stores uploads, but the document is still sent to the server that hosts it. For real ID documents, self-host it or keep it private.

## Example output

Line tokens from OCR, then word tokens from text reconstruction. Boxes are in original-image coordinates, and `start`/`end` are offsets into the reconstructed `full_text`:

```json
{
  "ocr":   [{ "id": 2, "text": "Name: Amit Kumar Sharma", "confidence": 0.9974,
              "bbox": [63, 118, 377, 144], "granularity": "line", "start": 18, "end": 41 }],
  "text": {
    "full_text": "...\nName: Amit Kumar Sharma\n...",
    "words": [
      { "id": 3, "text": "Amit",   "bbox": [145, 119, 198, 142], "parent_id": 2, "start": 24, "end": 28 },
      { "id": 4, "text": "Kumar",  "bbox": [206, 119, 280, 143], "parent_id": 2, "start": 29, "end": 34 },
      { "id": 5, "text": "Sharma", "bbox": [288, 120, 377, 144], "parent_id": 2, "start": 35, "end": 41 }
    ]
  }
}
```

A detector that finds `"Amit Kumar Sharma"` at chars 24–41 maps back to word tokens 3–5, and from there to their pixel boxes. The `700×400` sample is upscaled ×1.43 for OCR, yet every box comes back in the original `700×400` space. The affine round-trip is unit-tested, including deskew with canvas expansion.

## Project structure

```
app/
  config/         config.yaml + loader   (single source of truth)
  schemas/        typed data contracts   (OCRToken, ReconstructedText, DocumentResult, EntityCandidate, ResolvedEntity)
  utils/          logging, geometry (affine tracking)
  preprocessing/  deskew / denoise / scale with coordinate tracking
  ocr/            OCREngine interface + RapidOCR implementation
  text/           Phase 3 text reconstruction (reading order, word boxes, offset map)
  detection/      Detector interface; regex + validators (Phase 4);
                  name trie, name detector, context rules (Phase 5); GLiNER (Phase 6)
  resolution/     EntityResolver: candidates -> final entities (Phase 7)
  mapping/        SpatialMapper: entity spans -> per-line pixel boxes (Phase 7)
  redaction/      Redactor: policy, opaque fill, pixel check, leak check (Phase 8)
  visualization/  output image rendering (line/word boxes, entities, redaction)
  pipeline/       DocumentPipeline: runs every stage, reports pending ones
  main.py         CLI entrypoint
  ui.py           Streamlit web UI
data/names/       firstName / middleName / lastName lists
docs/             architecture diagrams (HLD / LLD)
eval/             (planned) labelled test set + precision/recall harness
tests/            unit tests
```

## Design decisions

- **Why hybrid, not just a model?** No single method wins. Regex nails structured entities; a NER model handles unstructured names; dictionaries help but over-fire alone. Combining signals and resolving them is more robust than any one detector.
- **Line-level vs word-level OCR.** PP-OCR detects whole lines, not words. The reconstruction stage splits lines into word tokens with interpolated boxes so a detected span maps to a tight region.
- **Multi-line entities.** Merging all boxes of a multi-line address into one rectangle would over-cover the page; instead each line gets its own box.
- **Agreement is evidence (noisy-OR).** Final confidence is `1 − Π(1 − weight × confidence)` over detectors. Independent detectors that agree raise the score, and repeats from one detector count once. A bare date (0.71) stays below the DATE threshold (0.80), while a `DOB:` date found by regex **and** a context rule reaches 0.87. A dictionary-only name (0.41) needs GLiNER to agree.
- **Partial overlaps keep both entities.** When two labels overlap, the loser is dropped only if it is mostly covered by the winner, and it is recorded on the winner. Dropping a partly overlapping entity would leave its uncovered part unredacted.
- **Addresses: a model *and* a rule.** In testing, GLiNER tagged only `House No 45`, `Sector 12` from a two-line address and missed `Dwarka, New Delhi 110075`. The resolver joins adjacent address pieces (absorbing building names and a trailing PIN), and an `Address:` context rule claims continuation rows up to the next `Label:` row. Together they covered the whole address, at 0.77 with one box per line.
- **Sub-word boxes.** OCR often merges a label and value (`DOB:01/01/1990` is one token). The mapper slices partly covered words with the same glyph-width model used for word boxes, so only the date is boxed. Boxes are padded 2 px, because over-covering by a few pixels is safe and under-covering leaks.
- **Redaction is verified, not assumed.** Boxes are filled with opaque pixels (never blurred, which can be reversed). Every box is then re-read to confirm it is 100% fill colour, and the redacted image is **OCR'd again** to confirm none of the redacted text, or any meaningful part of it, can still be read. A misplaced box shows up as a reported leak instead of a silent one.
- **Policy is separate from detection.** Organizations are detected and shown in the annotated image but left visible by default (`GOVERNMENT OF INDIA` is not personal data). Which labels get redacted is a config list, and every entity records whether it was redacted and why.
- **Coordinate correctness.** All preprocessing transforms are composed into one affine matrix and inverted, so boxes always map to the original image.
- **Validators boost, never gate.** OCR can misread one digit of a genuine Aadhaar number. Dropping it on a failed checksum would leak it, so a failed check only lowers confidence and the resolver decides.
- **OCR-tolerant numbers.** Numeric patterns run on a same-length "digit view" of the text that fixes common confusions (O→0, l→1, S→5, B→8), but only inside digit-heavy chunks. `9OOO0 l2345` is read as a phone number, while words like `BOSS` or `Oct` are left alone, and every match offset still points at the original text.
- **Weighted name signals, not a raw count.** Explicit evidence (a `Name:` label, `Mr.`, `S/O`) scores 2 and a dictionary hit scores 1, with acceptance at 2 or more. `Name: Fooby Barbaz` is caught even though it isn't in any list, `Amit Kumar Sharma` is caught in free text from two dictionary hits, and `Rose Garden` or `Kiran Industries` (one hit) is not. `Company Name: Acme` is rejected by a qualifier list.
- **Labels make generic numbers detectable.** A bare 12-digit number is too ambiguous to flag, but `A/c No. 1234 5678 9012` is clearly an account number. Context rules emit their own candidates, so a `DOB` date found by both regex and a context rule counts as agreement in the resolver.
- **The NER model is a signal, not an oracle.** A small model over-fires: on a newspaper sentence it tagged `The Prime Minister` as a person and `farmers` as an organization. Two cheap filters (person or organization spans need a capital letter; a "person" made only of titles is dropped) removed both, while `Priya Nair` and `Tata Consultancy Services` were kept. Prompt wording was compared too: `address` beat `street address` and `postal address` on recall.
- **Fail loudly, never silently.** If the NER model can't load (offline first run, out of memory), the stage is reported as *failed* with the reason, and no redacted image is produced. A redaction that quietly skipped a detector would look complete while leaking names.
- **Formatting is evidence.** `080-23456789` and `08023456789` contain the same digits, but the grouping shows the first is a landline (STD code), so the phone validator checks grouping as well as digits.

## Evaluation

A small hand-labelled test set is being built to report **precision / recall / F1 per entity type**, and to tune the resolver's confidence weights against ground truth rather than fixing them arbitrarily. Microsoft Presidio will be used as a benchmark baseline. *(Metrics will be added here as the evaluation phase completes.)*

## Future work

- Handwriting support via a vision-language OCR model
- Layout analysis for structured forms (associate labels with values spatially)
- Confidence calibration on a larger labelled set
- FastAPI service + containerised deployment
- Multilingual support (Hindi / Devanagari)

---

