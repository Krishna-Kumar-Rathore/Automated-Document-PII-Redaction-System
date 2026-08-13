"""CLI entrypoint.

Phase 2 usage:
    python -m app.main --image path/to/doc.jpg --outdir outputs

Writes:
    <outdir>/<doc>.ocr.json     structured OCR result
    <outdir>/<doc>.ocr_viz.jpg  original image with detected boxes drawn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from app.config.loader import load_config
from app.pipeline.ocr_pipeline import OCRPipeline
from app.visualization.ocr_viz import draw_ocr_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document OCR (Phase 2)")
    parser.add_argument("--image", required=True, help="Path to a JPG/PNG document")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--config", default=None, help="Optional config.yaml path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pipeline = OCRPipeline(cfg=cfg)
    result = pipeline.run(args.image)

    stem = Path(args.image).stem
    json_path = outdir / f"{stem}.ocr.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                         encoding="utf-8")

    image = cv2.imread(args.image)
    viz = draw_ocr_boxes(image, result.ocr)
    viz_path = outdir / f"{stem}.ocr_viz.jpg"
    cv2.imwrite(str(viz_path), viz)

    print(f"Tokens detected : {result.token_count}")
    print(f"Processing time : {result.processing_ms:.0f} ms")
    print(f"OCR JSON        : {json_path}")
    print(f"Visualization   : {viz_path}")


if __name__ == "__main__":
    main()
