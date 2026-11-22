"""Phase 6: NER with GLiNER - the unstructured complement to regex and rules.

GLiNER takes entity labels as natural-language prompts at inference time
("person", "address"), which keeps the entity registry configurable.

* The model is loaded lazily (``load()``), so building the pipeline is cheap and
  tests can inject a fake model with the same ``predict_entities`` interface.
* GLiNER reads ~384 tokens at a time, so the document text is split into chunks
  on row boundaries and every predicted span is shifted back into
  ``full_text`` offsets.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config.loader import Config
from app.detection.base import Detector
from app.schemas.entities import EntityCandidate
from app.schemas.text import ReconstructedText
from app.utils.logging import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_WORD_RE = re.compile(r"\S+")


def chunk_rows(text: str, max_words: int) -> List[Tuple[int, int]]:
    """Split ``text`` into contiguous [start, end) chunks of whole rows with at
    most ``max_words`` words each. A single over-long row is split on spaces.

    Known limitation: that word split has no overlap, so an entity straddling the
    cut is missed. OCR rows are far shorter than the default 200 words, so this
    only matters for unusual inputs (e.g. one huge paragraph detected as a row).
    """
    chunks: List[Tuple[int, int]] = []
    start = end = words = 0
    pos = 0
    for row in text.split("\n"):
        r_start = pos
        pos += len(row) + 1
        spans = [(r_start + m.start(), r_start + m.end()) for m in _WORD_RE.finditer(row)]
        if not spans:
            continue
        if len(spans) > max_words:                       # over-long row: split on words
            if words:
                chunks.append((start, end))
                words = 0
            for i in range(0, len(spans), max_words):
                group = spans[i:i + max_words]
                chunks.append((group[0][0], group[-1][1]))
            continue
        if words and words + len(spans) > max_words:     # row would overflow: flush
            chunks.append((start, end))
            words = 0
        if words == 0:
            start = spans[0][0]
        end, words = spans[-1][1], words + len(spans)
    if words:
        chunks.append((start, end))
    return chunks


class NERDetector(Detector):
    name = "gliner"

    def __init__(self, cfg: Config, model: Optional[Any] = None):
        nc = cfg.get("ner") or {}
        self.enabled = bool(nc.get("enabled", True))
        self.model_id = nc.get("model_id", "onnx-community/gliner_small-v2.1")
        self.onnx_file = nc.get("onnx_file")
        self.threshold = float(nc.get("threshold", 0.5))
        self.max_words = int(nc.get("max_words_per_chunk", 200))
        self.label_map: Dict[str, str] = dict(nc.get("labels", {}) or {})
        self.require_capital = set(nc.get("require_capital_for", []) or [])
        stop = (cfg.get("names") or {}).get("stopwords", []) or []
        self.non_name_words = {w.lower() for w in (nc.get("non_name_words", []) or []) + stop}
        models_dir = Path((cfg.get("paths") or {}).get("models_dir", "models"))
        self.cache_dir = models_dir if models_dir.is_absolute() else _PROJECT_ROOT / models_dir
        self._model = model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Download (first run only) and load the model. Raises on failure."""
        if self._model is not None:
            return
        # Windows without Developer Mode can't symlink the HF cache; it falls back
        # to copies, which is fine - silence the long warning about it.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from gliner import GLiNER                      # heavy import: torch + transformers
        from huggingface_hub import snapshot_download

        logger.info("Loading GLiNER %s (%s)", self.model_id, self.onnx_file or "pytorch")
        if not self.onnx_file:
            self._model = GLiNER.from_pretrained(self.model_id, cache_dir=str(self.cache_dir))
            return
        # The ONNX repo ships 8 variants (~2.9 GB in total). Fetch ONLY the chosen
        # file plus config / tokenizer, then load from that local snapshot.
        local_dir = snapshot_download(
            self.model_id, cache_dir=str(self.cache_dir),
            allow_patterns=["*.json", "spm.model", self.onnx_file],
        )
        self._model = GLiNER.from_pretrained(
            local_dir, load_onnx_model=True, load_tokenizer=True,
            onnx_model_file=self.onnx_file,
        )

    def detect(self, text: ReconstructedText) -> List[EntityCandidate]:
        self.load()
        full = text.full_text
        prompts = list(self.label_map)
        candidates: List[EntityCandidate] = []
        for c_start, c_end in chunk_rows(full, self.max_words):
            chunk = full[c_start:c_end]
            for ent in self._model.predict_entities(chunk, prompts, threshold=self.threshold):
                label = self.label_map.get(ent["label"])
                if label is None:
                    continue
                start, end = c_start + int(ent["start"]), c_start + int(ent["end"])
                reason = self._reject_reason(label, full[start:end])
                if reason:
                    logger.debug("GLiNER %s %r dropped: %s", label, full[start:end], reason)
                    continue
                candidates.append(EntityCandidate(
                    text=full[start:end], label=label, start=start, end=end,
                    confidence=round(float(ent["score"]), 3), detector=self.name,
                    attributes={"prompt": ent["label"], "model": self.model_id},
                ))
        logger.info("GLiNER: %d candidates", len(candidates))
        return candidates

    def _reject_reason(self, label: str, span: str) -> Optional[str]:
        """Precision filters for known small-model failure modes."""
        if label in self.require_capital and not any(c.isupper() for c in span):
            return "no capital letter (common noun)"
        if label == "PERSON_NAME":
            words = re.findall(r"[A-Za-z]+", span.lower())
            if words and all(w in self.non_name_words for w in words):
                return "only title / stop words"
        return None
