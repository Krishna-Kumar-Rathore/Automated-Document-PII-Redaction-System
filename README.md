# Intelligent Document Scan Auto-Redaction System

Detects PII in document images (Aadhaar/PAN/business cards/newspapers/etc.),
maps every detected entity back to its pixel region, and produces an
annotated image, a redacted image, and an auditable `result.json`.
     
The design goal that drives everything: preserve the chain

```
OCR text -> entity -> entity type -> OCR token -> bounding box -> image region
```

so every redaction can be explained.

## Architecture

```
Image
  -> Preprocessing (OpenCV)        affine-tracked: boxes always map to original coords
  -> OCR (RapidOCR / PP-OCR, ONNX) line-level tokens
  -> Text reconstruction           full text + char-offset <-> word-token map   [Phase 3]
  -> Hybrid PII detection          regex + GLiNER + name-trie + context rules    [Phase 4-6]
  -> Entity resolution             combine, score, de-conflict                   [Phase 7]
  -> Entity -> token -> bbox map   per-line boxes (multi-line safe)              [Phase 8]
  -> Annotated / redacted image    configurable colors                          [Phase 7-8]
  -> result.json                   auditable                                     [Phase 8]
  -> Streamlit UI                                                                [Phase 9]
```

## Status

- [x] **Phase 1** — foundations: config, schemas, logging, geometry
- [x] **Phase 2** — preprocessing + OCR + OCR JSON + visualization
- [ ] Phase 3 — text reconstruction (line -> word tokens, char offsets)
- [ ] Phase 4 — regex detectors (email/phone/url/upi + validators)
- [ ] Phase 5 — name detection (tries + context rules)
- [ ] Phase 6 — GLiNER NER
- [ ] Phase 7 — entity resolution
- [ ] Phase 8 — spatial mapping + result.json
- [ ] Phase 9 — Streamlit UI
- [ ] Phase 10-11 — eval harness, tests, deployment

## Setup

```bash
pip install -r requirements.txt
```

## Usage (Phase 2)

```bash
python -m app.main --image sample.jpg --outdir outputs
```

Produces `outputs/sample.ocr.json` and `outputs/sample.ocr_viz.jpg`.

## Project layout

```
app/
  config/         config.yaml + loader   (single source of truth)
  schemas/        typed data contracts   (OCRToken, EntityCandidate, ResolvedEntity)
  utils/          logging, geometry (affine tracking)
  preprocessing/  deskew / denoise / scale with coordinate tracking
  ocr/            OCREngine interface + RapidOCR implementation
  visualization/  OCR + entity box drawing
  pipeline/       stage orchestration
  main.py         CLI
data/names/       firstName / middleName / lastName lists
eval/             test set + precision/recall harness
```
