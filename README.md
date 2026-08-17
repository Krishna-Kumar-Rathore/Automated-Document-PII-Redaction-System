# Automated Document PII Redaction System

> Detects and redacts personally identifiable information (PII) from document images — Aadhaar cards, PAN cards, business cards, even newspaper scans — and maps every detected entity back to its exact pixel region for **auditable** redaction.


![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/runs%20on-CPU%20(no%20GPU)-success)

The design goal that drives the whole system:

```
OCR text  →  entity  →  entity type  →  OCR token  →  bounding box  →  image region
```

Every redaction can be traced back through this chain — which detector flagged it, which text it matched, and which pixels were covered. Detecting PII is the easy half; getting the coordinates right and keeping the decision explainable is the hard, interesting half.

---



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
| NER | **GLiNER** (ONNX) | Fast on CPU, and supports configurable labels at inference — fits the configurable entity registry |
| Structured detection | **Regex** + format validators | Aadhaar (Verhoeff checksum), PAN, IFSC boost confidence when patterns appear |
| Name detection | **Custom Trie** | O(length) lookups over first / middle / last-name lists |
| Image ops | **OpenCV** | Preprocessing, annotation, redaction |
| Config | **YAML** | Single source of truth for all tunables |
| UI | **Streamlit** (FastAPI optional later) | Fast demo UI, backend kept separable |
| Language | **Python 3.10+** | — |

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
- Python 3.10–3.12
- No GPU required

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/automated-document-pii-redaction.git
cd automated-document-pii-redaction
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

### Usage (current: Phase 2 — OCR)

```bash
python -m app.main --image sample.jpg --outdir outputs
```

This produces:
- `outputs/sample.ocr.json` — structured OCR result (tokens, confidence, boxes)
- `outputs/sample.ocr_viz.jpg` — the image with detected boxes drawn

> The first run downloads the OCR ONNX models once (~8 MB); after that it runs fully offline.

### Running tests

```bash
pytest tests/ -q
```

## Example output

OCR result (excerpt), with boxes reported in original-image coordinates:

```json
{
  "document_id": "sample",
  "image": { "filename": "sample.jpg", "width": 700, "height": 400 },
  "ocr": [
    { "id": 2, "text": "Name: Krishna Kumar Rathore", "confidence": 0.99,
      "bbox": [61, 119, 447, 145], "granularity": "line" }
  ]
}
```

A `700×400` image is upscaled ×1.43 for OCR, yet every box returns in the original `700×400` space — the affine round-trip is unit-tested so coordinates survive scaling and rotation exactly.

## Project structure

```
app/
  config/         config.yaml + loader   (single source of truth)
  schemas/        typed data contracts   (OCRToken, EntityCandidate, ResolvedEntity)
  utils/          logging, geometry (affine tracking)
  preprocessing/  deskew / denoise / scale with coordinate tracking
  ocr/            OCREngine interface + RapidOCR implementation
  visualization/  OCR + entity box drawing
  pipeline/       stage orchestration
  main.py         CLI entrypoint
data/names/       firstName / middleName / lastName lists
docs/             architecture diagrams
eval/             test set + precision/recall harness
tests/            unit tests
```

## Design decisions

- **Why hybrid, not just a model?** No single method wins. Regex nails structured entities; a NER model handles unstructured names; dictionaries help but over-fire alone. Combining signals and resolving them is more robust than any one detector.
- **Line-level vs word-level OCR.** PP-OCR detects whole lines, not words. The reconstruction stage splits lines into word tokens with interpolated boxes so a detected span maps to a tight region.
- **Multi-line entities.** Merging all boxes of a multi-line address into one rectangle would over-cover the page; instead each line gets its own box.
- **Coordinate correctness.** All preprocessing transforms are composed into one affine matrix and inverted, so boxes always map to the original image.

## Evaluation

A small hand-labelled test set is being built to report **precision / recall / F1 per entity type**, and to tune the resolver's confidence weights against ground truth rather than fixing them arbitrarily. Microsoft Presidio will be used as a benchmark baseline. *(Metrics will be added here as the evaluation phase completes.)*

## Future work

- Handwriting support via a vision-language OCR model
- Layout analysis for structured forms (associate labels with values spatially)
- Confidence calibration on a larger labelled set
- FastAPI service + containerised deployment
- Multilingual support (Hindi / Devanagari)

---

