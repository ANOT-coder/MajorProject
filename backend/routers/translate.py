"""
/api/translate  — full text-to-animation pipeline

POST /api/translate
  Body:  { "text": "Hello, how are you?" }
  Response: TranslationResponse (see schema below)
"""
import time
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.preprocessor import text_to_gloss
from services.vocab        import glosses_to_word_id_pairs, get_vocab_size  
from services.inference    import run_inference_per_word, model_is_loaded    

logger = logging.getLogger(__name__)
router = APIRouter(tags=["translate"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500,
                      example="Hello, how are you?")


class JointPosition(BaseModel):
    x: float
    y: float
    z: float   # always present for ASL 3-D output


class KeypointFrame(BaseModel):
    joints: dict[str, JointPosition]

class WordSegment(BaseModel):
    word:        str
    start_frame: int
    end_frame:   int

class TranslationResponse(BaseModel):
    original_text:  str
    glosses:        list[str]
    gloss_ids:      list[int]
    oov_glosses:    list[str]
    frames:         list[KeypointFrame]
    word_segments:  list[WordSegment]    # ← new
    frame_count:    int
    fps:            int
    duration_ms:    float
    processing_ms:  float
    model_loaded:   bool


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslateRequest):
    """
    Full pipeline:
      raw text → preprocessing → gloss sequence
               → vocab mapping  → gloss ID sequence
               → model inference → keypoint frames
               → JSON response   → frontend animation
    """
    t0 = time.perf_counter()
    text = req.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Input text is empty.")

    # ── Step 1: Text → Gloss ────────────────────────────────────────────────
    logger.info(f"[translate] input: {text!r}")
    glosses = text_to_gloss(text)
    logger.info(f"[translate] glosses: {glosses}")

    if not glosses:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any gloss words from input. "
                   "Try a longer or more specific sentence."
        )

    # ── Step 2: Gloss → (word, id) pairs ────────────────────────────────────
    pairs, oov = glosses_to_word_id_pairs(glosses)
    gloss_ids  = [gid for _, gid in pairs]
    logger.info(f"[translate] pairs: {pairs}, oov: {oov}")

    if not pairs:
        raise HTTPException(
            status_code=422,
            detail=f"None of the gloss words were found in the vocabulary. "
                   f"OOV words: {oov}"
        )

    # ── Step 3: Per-word inference → concatenated frames ────────────────────
    raw_frames, word_segments = run_inference_per_word(pairs)
    logger.info(f"[translate] generated {len(raw_frames)} frames across {len(pairs)} words")

    # ── Step 4: Assemble response ────────────────────────────────────────────
    fps = 25
    duration_ms = (len(raw_frames) / fps) * 1000
    processing_ms = (time.perf_counter() - t0) * 1000

    return TranslationResponse(
        original_text=text,
        glosses=glosses,
        gloss_ids=gloss_ids,
        oov_glosses=oov,
        frames=raw_frames,
        word_segments=word_segments,    # ← new
        frame_count=len(raw_frames),
        fps=fps,
        duration_ms=round(duration_ms, 1),
        processing_ms=round(processing_ms, 1),
        model_loaded=model_is_loaded(),
    )



# ---------------------------------------------------------------------------
# Helper endpoint: preview preprocessing only (no model call)
# ---------------------------------------------------------------------------

class GlossPreviewResponse(BaseModel):
    original_text: str
    glosses:       list[str]
    gloss_ids:     list[int]
    oov_glosses:   list[str]
    vocab_size:    int


@router.post("/translate/preview-gloss", response_model=GlossPreviewResponse)
async def preview_gloss(req: TranslateRequest):
    """
    Returns gloss + IDs without running the model.
    Useful for debugging the preprocessing pipeline.
    """
    glosses = text_to_gloss(req.text.strip())
    gloss_ids, oov = glosses_to_ids(glosses, skip_unknown=False)
    return GlossPreviewResponse(
        original_text=req.text,
        glosses=glosses,
        gloss_ids=gloss_ids,
        oov_glosses=oov,
        vocab_size=get_vocab_size(),
    )