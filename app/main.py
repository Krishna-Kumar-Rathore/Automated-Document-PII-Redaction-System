"""CLI entrypoint.

Usage:
    python -m app.main --image path/to/doc.jpg [--outdir outputs] [--config cfg.yaml]

Writes (into --outdir, default: config ``paths.outputs_dir``):
    <doc>.result.json        full pipeline result (stages, OCR, text, entities)
    <doc>.ocr_lines.jpg      OCR line boxes
    <doc>.ocr_words.jpg      reconstructed word boxes
    <doc>.redacted.jpg       only once every pipeline stage is implemented
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from app.config.loader import load_config
from app.pipeline.document_pipeline import DocumentPipeline
from app.utils.logging import configure_logging
from app.visualization.render import render_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document PII redaction pipeline")
    parser.add_argument("--image", required=True, help="Path to a JPG/PNG document")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: paths.outputs_dir in config)")
    parser.add_argument("--config", default=None, help="Optional config.yaml path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.logging.get("level", "INFO"))
    outdir = Path(args.outdir or cfg.paths.get("outputs_dir", "outputs"))
    outdir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(args.image)          # read once; pipeline works in memory
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    pipeline = DocumentPipeline(cfg=cfg)
    # Load lazy models first so "Processing time" measures the document, not the
    # one-off model load. A load error is reported by the NER stage itself.
    warmup_error = pipeline.warm_up()
    if warmup_error:
        print(f"Warning: NER model failed to load ({warmup_error})")
    result = pipeline.run_image(image, filename=Path(args.image).name)

    stem = result.document_id
    json_path = outdir / f"{stem}.result.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                         encoding="utf-8")
    written = [json_path]
    for name, img in render_outputs(image, result, cfg).items():
        path = outdir / f"{stem}.{name}.jpg"
        cv2.imwrite(str(path), img)
        written.append(path)

    print(f"Lines / words   : {result.token_count} / {result.word_count}")
    print(f"Processing time : {result.processing_ms:.0f} ms")
    marks = {"done": "x", "failed": "!", "skipped": "-"}
    for stage in result.stages:
        note = f"  ({stage.note})" if stage.note else ""
        print(f"  [{marks.get(stage.status, ' ')}] Phase {stage.phase:<4} {stage.name}{note}")
    print(f"PII candidates  : {len(result.candidates)}")
    for c in result.candidates:
        check = {True: "valid", False: "INVALID"}.get(c.attributes.get("validated"), "")
        print(f"  {c.label:<14} {c.confidence:.2f}  {c.detector:<12} {c.text!r}  {check}")
    print(f"Entities        : {len(result.entities)}")
    for e in result.entities:
        print(f"  #{e.entity_id} {e.label:<14} {e.confidence:.2f}  {e.text!r}  "
              f"boxes={e.bboxes}  by={'+'.join(e.detectors)}")
    if not result.is_complete:
        print("Redacted image  : not produced - pipeline incomplete")
    else:
        r = result.redaction
        lc = r.get("leak_check", {})
        print(f"Redaction       : {r['redacted']} entities ({r['style']}); "
              f"pixel check {'passed' if r['pixel_check_passed'] else 'FAILED'}; "
              f"leak check {'skipped' if not lc.get('ran') else 'passed' if lc['passed'] else 'FAILED'}")
        for k in r["kept_visible"]:
            print(f"  kept visible  : #{k['entity_id']} {k['label']} ({k['reason']})")
        for leak in lc.get("leaks", []):
            print(f"  ! LEAK ({leak['match']}): #{leak['entity_id']} {leak['label']} "
                  f"readable: {leak['readable']}")
    for path in written:
        print(f"Wrote           : {path}")


if __name__ == "__main__":
    main()
