"""Streamlit web UI: upload a document image -> run the full pipeline -> output images.

Run locally:   streamlit run app/ui.py
Deploy:        Streamlit Community Cloud (main file: app/ui.py) - see README.

Privacy: uploads are decoded in memory and never written to disk. Results are kept
only in the visitor's own session state (not a shared cache), so one user's
document is never served to another.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# `streamlit run app/ui.py` puts app/ (not the repo root) on sys.path.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from app.config.loader import Config, load_config  # noqa: E402
from app.pipeline.document_pipeline import DocumentPipeline  # noqa: E402
from app.schemas.document import DONE, FAILED, SKIPPED  # noqa: E402
from app.utils.logging import configure_logging  # noqa: E402
from app.visualization.render import OUTPUT_TITLES, render_outputs  # noqa: E402

SAMPLE_PATH = _ROOT / "sample.jpg"

st.set_page_config(page_title="Document PII Redaction", page_icon="🛡️", layout="wide")


# -- cached resources (shared across sessions: code + models only, no user data) --

@st.cache_resource(show_spinner="Loading models (first run downloads the NER model, ~183 MB)...")
def get_pipeline() -> tuple[Config, DocumentPipeline, str | None]:
    cfg = load_config()
    configure_logging(cfg.logging.get("level", "INFO"))
    pipeline = DocumentPipeline(cfg=cfg)
    return cfg, pipeline, pipeline.warm_up()


# -- helpers ----------------------------------------------------------------

def decode_image(data: bytes) -> np.ndarray | None:
    # IMREAD_COLOR also applies EXIF orientation, so phone photos come out upright.
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def evidence(c) -> str:
    """One-line, human-readable reason a detector flagged this candidate."""
    a = c.attributes
    if "signals" in a:                                   # name detector
        parts = [f"{s} ({a['keyword']})" if s in ("context_label", "relation", "honorific")
                 and a.get("keyword") else s for s in a["signals"]]
        return f"score {a.get('score', 0):g}: " + ", ".join(parts)
    if "rule" in a:                                      # context rule
        return f"label '{a.get('keyword', '')}'"
    if "prompt" in a:                                    # GLiNER
        return f"NER prompt '{a['prompt']}'"
    return f"pattern {a.get('pattern', '')}"             # regex


def process(data: bytes, filename: str) -> dict | None:
    """Run the pipeline once per distinct upload; cache in this session only."""
    key = hashlib.sha256(data).hexdigest()
    cached = st.session_state.get("last")
    if cached and cached["key"] == key:
        return cached

    image = decode_image(data)
    if image is None:
        st.error("Could not read that file as an image. Please upload a JPG or PNG.")
        return None

    cfg, pipeline, _ = get_pipeline()
    with st.spinner("Running pipeline..."):
        result = pipeline.run_image(image, filename=filename)
        outputs = render_outputs(image, result, cfg)
    entry = {"key": key, "image": image, "result": result, "outputs": outputs}
    st.session_state["last"] = entry
    return entry


# -- page -------------------------------------------------------------------

def render_sidebar(result) -> None:
    st.sidebar.header("Pipeline stages")
    for stage in result.stages:
        if stage.status == DONE:
            st.sidebar.markdown(f"✅ **{stage.name}** · Phase {stage.phase} · {stage.ms:.0f} ms")
        elif stage.status == FAILED:
            st.sidebar.markdown(f"❌ **{stage.name}** · Phase {stage.phase} · *failed*")
            st.sidebar.caption(stage.note)
        elif stage.status == SKIPPED:
            st.sidebar.markdown(f"⏭️ {stage.name} · Phase {stage.phase} · *{stage.note}*")
        else:
            st.sidebar.markdown(f"⏳ {stage.name} · Phase {stage.phase} · *not built yet*")


def render_results(entry: dict) -> None:
    image, result, outputs = entry["image"], entry["result"], entry["outputs"]
    stem = result.document_id
    render_sidebar(result)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Image", f"{result.image.width}×{result.image.height}")
    m2.metric("OCR lines", result.token_count)
    m3.metric("Words", result.word_count)
    m4.metric("Processing", f"{result.processing_ms / 1000:.2f} s")

    # Final output: only once the document has gone through EVERY stage.
    st.subheader("Redacted output")
    if result.is_complete:
        left, right = st.columns(2)
        left.image(to_rgb(image), caption="Original", width="stretch")
        right.image(to_rgb(outputs["redacted"]), caption="Redacted", width="stretch")
        st.download_button("Download redacted image", to_png(outputs["redacted"]),
                           file_name=f"{stem}.redacted.png", mime="image/png",
                           type="primary")
    else:
        failed = [s for s in result.pending_stages if s.status == FAILED]
        pending = [s for s in result.pending_stages if s.status != FAILED]
        if failed:
            st.error("Stage failed, so no redacted image was produced: " + "; ".join(
                f"**{s.name}**: {s.note}" for s in failed))
        if pending:
            names = ", ".join(f"{s.name} (Phase {s.phase})" for s in pending)
            st.info("The redacted image is generated only after the document passes "
                    f"through every pipeline stage. Not built yet: **{names}**. "
                    "Intermediate outputs from the completed stages are shown below.")

    st.subheader(f"Final entities ({len(result.entities)})")
    if result.entities:
        st.caption("After resolution: duplicates merged, agreeing detectors combined, "
                   "address pieces joined, and each entity mapped to one box per line.")
        st.dataframe([{
            "#": e.entity_id,
            "label": e.label,
            "text": e.text.replace("\n", " ⏎ "),
            "confidence": e.confidence,
            "detectors": " + ".join(e.detectors),
            "boxes": len(e.bboxes),
            "notes": "; ".join(filter(None, [
                f"merged {len(e.attributes['merged_from'])} pieces"
                if e.attributes.get("merged_from") else "",
                "suppressed " + ", ".join(s["label"] for s in e.attributes["suppressed"])
                if e.attributes.get("suppressed") else "",
            ])),
        } for e in result.entities], hide_index=True, width="stretch")
    else:
        st.caption("No entity passed the resolver's confidence thresholds.")

    with st.expander(f"Raw detector candidates ({len(result.candidates)}) - the evidence"):
        if not result.candidates:
            st.caption("No detector produced a candidate.")
        else:
            st.caption("Before resolution. Overlaps and duplicates are expected here.")
            st.dataframe([{
                "label": c.label,
                "text": c.text,
                "confidence": c.confidence,
                "detector": c.detector,
                "evidence": evidence(c),
                "validated": {True: "✅ pass", False: "❌ fail"}.get(
                    c.attributes.get("validated"), "—"),
                "OCR-corrected": c.attributes.get("ocr_corrected", ""),
            } for c in result.candidates], hide_index=True, width="stretch")

    st.subheader("Stage outputs")
    names = [n for n in OUTPUT_TITLES if n in outputs and n != "redacted"]
    tabs = st.tabs([OUTPUT_TITLES[n] for n in names] + ["Original"])
    for tab, name in zip(tabs, names):
        with tab:
            st.image(to_rgb(outputs[name]), width="stretch")
            st.download_button(f"Download {OUTPUT_TITLES[name].lower()}",
                               to_png(outputs[name]), file_name=f"{stem}.{name}.png",
                               mime="image/png", key=f"dl_{name}")
    with tabs[-1]:
        st.image(to_rgb(image), width="stretch")

    with st.expander("Reconstructed text (what the PII detectors will read)"):
        st.code(result.text.full_text if result.text else "", language=None)

    st.download_button("Download result.json",
                       json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                       file_name=f"{stem}.result.json", mime="application/json")


def main() -> None:
    st.title("🛡️ Document PII Redaction")
    st.caption("Upload a document image (ID card, form, letter, business card). "
               "It is processed in memory and never stored.")

    cfg, _, warmup_error = get_pipeline()
    if warmup_error:
        st.warning(f"The NER model could not be loaded ({warmup_error}). Documents will "
                   "still be processed, but the NER stage will fail and no redacted "
                   "image will be produced until this is fixed.")
    allowed = list(cfg.ui.get("allowed_types", ["jpg", "jpeg", "png"]))
    uploaded = st.file_uploader("Document image", type=allowed)

    if uploaded is None and SAMPLE_PATH.exists() and \
            st.button("…or try the synthetic sample document"):
        st.session_state["use_sample"] = True
    use_sample = uploaded is None and st.session_state.get("use_sample", False)

    if uploaded is not None:
        st.session_state["use_sample"] = False
        entry = process(uploaded.getvalue(), uploaded.name)
    elif use_sample:
        entry = process(SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name)
    else:
        return

    if entry:
        render_results(entry)


main()
