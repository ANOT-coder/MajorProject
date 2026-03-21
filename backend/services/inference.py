"""
ASL Gloss → Pose Inference Service
====================================
Model: gloss2pose_lstm.pt  (Gloss2PoseSeq2Seq trained in Gloss_To_Pose_LSTM_Model.ipynb)

Architecture (from training notebook):
  Embedding       vocab_size × 128
  Encoder         BiLSTM  input=128, hidden=256, bidirectional → enc_dim=512
  init_h/init_c   Linear(512 → 512)  initialise decoder from masked-mean of encoder
  Attention       BahdanauAttention(enc_dim=512, dec_dim=512, attn_dim=256)
                  uses enc_mask to ignore padding tokens
  Decoder         2-layer LSTM  input = d_out(225) + enc_dim(512) = 737, hidden=512
  out_proj        Linear(512 → 225)   raw normalised coords  (NO sigmoid)
  stop_proj       Linear(512 → 1)     EOS logit  (sigmoid applied at generate time)

Stop logic (from generate()):
  sigmoid(stop_proj) > stop_thresh  AND  t >= 5   →  break
  i.e. minimum 5 frames always generated; model decides the rest.

Coordinate space (from preprocess_to_npz_v3.py):
  anchor = mid-hip  (fallback: mid-shoulder)
  scale  = shoulder-width
  X_norm = (xyz - anchor) / scale
  Values centred at 0, range roughly [-3, 3].  NOT [0,1].

Joint layout (225 = 75 joints × 3):
  pose[0..32]  MediaPipe Pose  (33 joints, official landmark order)
  lh[0..20]    MediaPipe Left Hand  (21 joints)
  rh[0..20]    MediaPipe Right Hand (21 joints)
"""

import logging
from pathlib import Path
from typing import Any
import math

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "gloss2pose_lstm.pt"

# ── Architecture constants (must match training config) ───────────────────────
D_OUT       = 225
EMB_DIM     = 128
ENC_HIDDEN  = 256          # per direction; enc_dim = 512
DEC_HIDDEN  = 512
ENC_LAYERS  = 1
DEC_LAYERS  = 2
ATTN_DIM    = 256
PAD_GLOSS   = 0

# ── Inference constants (from training notebook) ──────────────────────────────
MAX_T_OUT          = 300    
STOP_THRESH        = 0.55   
STOP_MIN_STEPS     = 12    
STOP_CONSEC_FRAMES = 5      # NEW — was missing entirely
STATIC_TAIL_EPS    = 0.0025 # NEW — was missing entirely
STATIC_TAIL_PATIENCE = 8    # same default as training's generate(stop_thresh=0.5)


# ── Joint names ───────────────────────────────────────────────────────────────
# Order matches preprocess_to_npz_v3.py:
#   pose (33) → left_hand (21) → right_hand (21)
NUM_JOINTS = 75
OUTPUT_DIM = 3

JOINT_NAMES: list[str] = [
    # Pose landmarks 0–32  (MediaPipe Pose official order)
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky1", "right_pinky1",
    "left_index1", "right_index1",
    "left_thumb1", "right_thumb1",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
    # Left hand landmarks 0–20
    *[f"left_hand_{i}" for i in range(21)],
    # Right hand landmarks 0–20
    *[f"right_hand_{i}" for i in range(21)],
]
assert len(JOINT_NAMES) == NUM_JOINTS


# ─────────────────────────────────────────────────────────────────────────────
# Model definition  —  copied verbatim from training notebook
# ─────────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn

    def lengths_to_mask(lengths, max_len=None):
        B = lengths.size(0)
        max_len = int(max_len or lengths.max().item())
        ar = torch.arange(max_len, device=lengths.device)[None, :].expand(B, max_len)
        return (ar < lengths[:, None]).float()   # (B, max_len)

    class BahdanauAttention(nn.Module):
        def __init__(self, enc_dim, dec_dim, attn_dim=256):
            super().__init__()
            self.W = nn.Linear(enc_dim, attn_dim, bias=False)
            self.U = nn.Linear(dec_dim, attn_dim, bias=False)
            self.v = nn.Linear(attn_dim, 1,        bias=False)

        def forward(self, enc_out, enc_mask, dec_h):
            score = self.v(
                torch.tanh(self.W(enc_out) + self.U(dec_h)[:, None, :])
            ).squeeze(-1)                                    # (B, S)
            score = score.masked_fill(enc_mask <= 0.0, -1e9)
            attn_w = torch.softmax(score, dim=1)             # (B, S)
            ctx = torch.bmm(attn_w[:, None, :], enc_out).squeeze(1)  # (B, enc_dim)
            return ctx, attn_w

    class Gloss2PoseSeq2Seq(nn.Module):
        def __init__(
            self,
            vocab_size,
            d_out=D_OUT,
            emb_dim=EMB_DIM,
            enc_hidden=ENC_HIDDEN,
            dec_hidden=DEC_HIDDEN,
            enc_layers=ENC_LAYERS,
            dec_layers=DEC_LAYERS,
            dropout=0.2,
            use_attention=True,
            use_stop_head=True,
        ):
            super().__init__()
            self.use_attention = use_attention
            self.use_stop_head = use_stop_head

            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_GLOSS)

            self.encoder = nn.LSTM(
                input_size=emb_dim,
                hidden_size=enc_hidden,
                num_layers=enc_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.0 if enc_layers == 1 else dropout,
            )
            enc_dim = enc_hidden * 2   # 512

            self.init_h = nn.Linear(enc_dim, dec_hidden)
            self.init_c = nn.Linear(enc_dim, dec_hidden)

            dec_in_dim = d_out
            if use_attention:
                self.attn = BahdanauAttention(enc_dim=enc_dim, dec_dim=dec_hidden, attn_dim=ATTN_DIM)
                dec_in_dim = d_out + enc_dim   # 737

            self.decoder = nn.LSTM(
                input_size=dec_in_dim,
                hidden_size=dec_hidden,
                num_layers=dec_layers,
                batch_first=True,
                dropout=0.0 if dec_layers == 1 else dropout,
            )

            self.out_proj = nn.Linear(dec_hidden, d_out)
            if use_stop_head:
                self.stop_proj = nn.Linear(dec_hidden, 1)

        def encode(self, gloss, gloss_lens):
            emb = self.emb(gloss)
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, gloss_lens.cpu(), batch_first=True, enforce_sorted=False
            )
            enc_packed, _ = self.encoder(packed)
            enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_packed, batch_first=True)

            enc_mask = lengths_to_mask(gloss_lens.to(enc_out.device), max_len=enc_out.size(1))

            denom   = enc_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            enc_sum = (enc_out * enc_mask[:, :, None]).sum(dim=1) / denom

            h0 = torch.tanh(self.init_h(enc_sum))
            c0 = torch.tanh(self.init_c(enc_sum))
            return enc_out, enc_mask, h0, c0

        @torch.no_grad()
        def generate(self, gloss, gloss_lens, max_T=MAX_T_OUT, stop_thresh=STOP_THRESH,
                    stop_min_steps=STOP_MIN_STEPS, stop_consec_frames=STOP_CONSEC_FRAMES):
            self.eval()
            B = gloss.size(0)
            enc_out, enc_mask, h0, c0 = self.encode(gloss, gloss_lens)
            dec_layers = self.decoder.num_layers
            h = h0.unsqueeze(0).repeat(dec_layers, 1, 1).contiguous()
            c = c0.unsqueeze(0).repeat(dec_layers, 1, 1).contiguous()
            prev = enc_out.new_zeros((B, self.out_proj.out_features))

            preds, stop_probs = [], []
            stop_counter  = torch.zeros(B, dtype=torch.long, device=enc_out.device)
            static_counter= torch.zeros(B, dtype=torch.long, device=enc_out.device)

            for t in range(max_T):
                if self.use_attention:
                    ctx, _ = self.attn(enc_out, enc_mask, h[-1])
                    dec_in = torch.cat([prev, ctx], dim=-1)[:, None, :]
                else:
                    dec_in = prev[:, None, :]

                dec_out, (h, c) = self.decoder(dec_in, (h, c))
                step_h = dec_out[:, 0, :]
                x_pred = self.out_proj(step_h)
                preds.append(x_pred)

                # static-tail detector
                if t > 0:
                    frame_motion = torch.mean(torch.abs(x_pred - prev), dim=1)
                    static_counter = torch.where(
                        frame_motion < STATIC_TAIL_EPS,
                        static_counter + 1,
                        torch.zeros_like(static_counter)
                    )

                if self.use_stop_head:
                    sp = torch.sigmoid(self.stop_proj(step_h)).squeeze(-1)
                    stop_probs.append(sp)
                    stop_counter = torch.where(
                        sp > stop_thresh,
                        stop_counter + 1,
                        torch.zeros_like(stop_counter)
                    )
                    if t >= stop_min_steps:
                        if (stop_counter >= stop_consec_frames).all():
                            break
                        if (static_counter >= STATIC_TAIL_PATIENCE).all():
                            break

                prev = x_pred

            X = torch.stack(preds, dim=1)
            S = torch.stack(stop_probs, dim=1) if self.use_stop_head else None
            return X, S

    TORCH_AVAILABLE = True
    logger.info("PyTorch available.")

except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — MOCK inference mode.")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
_model_cache: Any = None

def _load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not TORCH_AVAILABLE:
        return None
    if not MODEL_PATH.exists():
        logger.warning(f"Model not found at {MODEL_PATH} — running in MOCK mode.")
        return None

    import torch
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    # Checkpoint format from training: {"model": state_dict, "vocab_size": N, ...}
    state_dict   = ckpt["model"]
    vocab_size   = ckpt.get("vocab_size",   72)
    use_attention = ckpt.get("use_attention", True)
    use_stop_head = ckpt.get("use_stop_head", True)

    model = Gloss2PoseSeq2Seq(
        vocab_size=vocab_size,
        use_attention=use_attention,
        use_stop_head=use_stop_head,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        logger.error(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    model.eval()
    _model_cache = model
    logger.info(
        f"gloss2pose_lstm.pt loaded — vocab={vocab_size}, "
        f"attention={use_attention}, stop_head={use_stop_head}, "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Output conversion
# ─────────────────────────────────────────────────────────────────────────────
def _tensor_to_frames(X) -> list[dict]:
    """
    X: (1, T, 225) tensor  →  list of T frame dicts
    Each frame: { joints: { name: {x, y, z} } }
    Values are in anchor/scale normalised space (centred ~0, range ~[-3,3]).
    """
    frames = []
    for flat in X[0].detach().cpu().tolist():   # flat: list of 225 floats
        joints = {}
        for j, name in enumerate(JOINT_NAMES):
            b = j * OUTPUT_DIM
            joints[name] = {
                "x": round(flat[b],     4),
                "y": round(flat[b + 1], 4),
                "z": round(flat[b + 2], 4),
            }
        frames.append({"joints": joints})
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Mock frames (used when model file is absent)
# ─────────────────────────────────────────────────────────────────────────────
# Mock uses anchor/scale space too: centred at 0, shoulder-width ≈ 1 unit
_MOCK_BODY = {
    "nose":           ( 0.00, -1.80, 0.00),
    "left_shoulder":  (-0.50, -1.20, 0.00),
    "right_shoulder": ( 0.50, -1.20, 0.00),
    "left_elbow":     (-0.80, -0.50, 0.05),
    "right_elbow":    ( 0.80, -0.50, 0.05),
    "left_wrist":     (-0.90,  0.20, 0.08),
    "right_wrist":    ( 0.90,  0.20, 0.08),
    "left_hip":       (-0.25,  0.00, 0.00),
    "right_hip":      ( 0.25,  0.00, 0.00),
    "left_knee":      (-0.28,  0.90, 0.00),
    "right_knee":     ( 0.28,  0.90, 0.00),
}

def _mock_frames(num_glosses: int) -> list[dict]:
    n = max(STOP_MIN_STEPS + 1, min(MAX_T_OUT, num_glosses * 20))
    frames = []
    for fi in range(n):
        t = fi / n
        joints = {}
        for j, name in enumerate(JOINT_NAMES):
            ph = j * 0.22
            if name in _MOCK_BODY:
                bx, by, bz = _MOCK_BODY[name]
                x = bx + 0.05 * math.sin(2 * math.pi * t + ph)
                y = by + 0.03 * math.cos(2 * math.pi * t * 0.7)
                z = bz
            elif name.startswith("right_hand"):
                idx = int(name.split("_")[-1]); fp = idx // 4
                x = 0.90 + 0.20 * math.sin(2 * math.pi * t * 2 + fp * 0.4)
                y = 0.20 - 0.40 * math.sin(math.pi * t)
                z = 0.08 + 0.05 * math.cos(2 * math.pi * t)
            else:
                idx = int(name.split("_")[-1]) if name.startswith("left_hand") else j
                fp  = idx // 4 if name.startswith("left_hand") else 0
                x = -0.90 - 0.15 * math.sin(2 * math.pi * t * 2 + fp * 0.4)
                y =  0.20 - 0.25 * math.sin(math.pi * t * 1.3)
                z =  0.08 + 0.03 * math.cos(2 * math.pi * t)
            joints[name] = {"x": round(x, 4), "y": round(y, 4), "z": round(z, 4)}
        frames.append({"joints": joints})
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(gloss_ids: list[int]) -> list[dict]:
    """
    gloss_ids: list of integer gloss token IDs from vocab.py
    Returns:   list of frame dicts ready for JSON serialisation
    """
    if not gloss_ids:
        return []

    model = _load_model()
    if model is None:
        return _mock_frames(len(gloss_ids))

    import torch

    # Build (1, S) gloss tensor + (1,) lengths tensor — both required by encode()
    src      = torch.tensor([gloss_ids], dtype=torch.long)          # (1, S)
    src_lens = torch.tensor([len(gloss_ids)], dtype=torch.long)     # (1,)

    # Use the model's own generate() — identical to training notebook inference
    X, stop_probs = model.generate(src, src_lens, max_T=MAX_T_OUT, stop_thresh=STOP_THRESH)

    frames = _tensor_to_frames(X)
    logger.info(
        f"Inference: {len(gloss_ids)} glosses → {len(frames)} frames "
        f"(max_T={MAX_T_OUT}, stopped={'early' if len(frames) < MAX_T_OUT else 'at cap'})"
    )
    return frames

def _interpolate_joints(joints_a: dict, joints_b: dict, t: float) -> dict:
    """
    Pure linear interpolation between two joint dicts.
    t=0.0 → joints_a,  t=1.0 → joints_b
    """
    result = {}
    for name in joints_a:
        if name in joints_b:
            a, b = joints_a[name], joints_b[name]
            result[name] = {
                "x": round(a["x"] + (b["x"] - a["x"]) * t, 4),
                "y": round(a["y"] + (b["y"] - a["y"]) * t, 4),
                "z": round(a["z"] + (b["z"] - a["z"]) * t, 4),
            }
        else:
            result[name] = joints_a[name]
    return result


def _ease_in_out(t: float) -> float:
    """
    Smoothstep — accelerates out of the first pose, decelerates into the next.
    Feels much more natural than linear for body motion.
    """
    return t * t * (3.0 - 2.0 * t)


# ── Replace _smooth_concatenate entirely ─────────────────────────────────────

# ── Replace _trim_static_tail with this ──────────────────────────────────────

def _trim_tail(frames: list[dict], trim: int = 10, min_keep: int = 10) -> list[dict]:
    """
    Always remove the last `trim` frames from a word's frame list.
    Never trims below `min_keep` frames regardless.
    """
    keep = max(min_keep, len(frames) - trim)
    return frames[:keep]


def _smooth_concatenate(
    segments: list[list[dict]],
    transition_frames: int = 8,
) -> tuple[list[dict], list[dict]]:
    """
    Concatenates per-word frame lists with smooth interpolated transitions
    inserted between words.

    Each word's static tail is trimmed first so signs don't appear
    cut-off — the trim removes only the frozen hold frames the model
    appends after the sign is complete, not the sign itself.
    """
    if not segments:
        return [], []

    all_frames:    list[dict] = []
    word_segments: list[dict] = []

    for i, seg_frames in enumerate(segments):
        if not seg_frames:
            continue

        # Trim static tail from every word except the very last
        # (keep the last word's tail so the animation ends cleanly)
        if i < len(segments) - 1:
            seg_frames = _trim_tail(seg_frames, trim=10, min_keep=10)

        start = len(all_frames)

        if i == 0:
            all_frames.extend(seg_frames)
        else:
            # Insert eased transition frames between the two words
            last_frame  = all_frames[-1]["joints"]
            first_frame = seg_frames[0]["joints"]

            for step in range(transition_frames):
                t_raw = (step + 1) / (transition_frames + 1)   # never 0 or 1
                t     = _ease_in_out(t_raw)
                all_frames.append({
                    "joints": _interpolate_joints(last_frame, first_frame, t)
                })

            start = len(all_frames)   # word starts AFTER transition
            all_frames.extend(seg_frames)

        end = len(all_frames) - 1
        word_segments.append({"start_frame": start, "end_frame": end})

    return all_frames, word_segments
# Add this at the bottom of inference.py, after run_inference()

def run_inference_per_word(
    words_and_ids: list[tuple[str, int]],
    transition_frames: int = 8,
) -> tuple[list[dict], list[dict]]:
    """
    Word-level inference: runs the model once per word, then smoothly
    concatenates with interpolated transition frames between words.

    Args:
        words_and_ids:     [(gloss_word, gloss_id), ...]
        transition_frames: frames to insert between words (default 8 = 320ms at 25fps)

    Returns:
        (frames, word_segments)
    """
    if not words_and_ids:
        return [], []

    model = _load_model()

    # ── Step 1: run inference for every word independently ───────────────────
    per_word_frames: list[list[dict]] = []
    gloss_words: list[str] = []

    for gloss, gloss_id in words_and_ids:
        if model is None:
            word_frames = _mock_frames(1)
        else:
            import torch
            src      = torch.tensor([[gloss_id]], dtype=torch.long)
            src_lens = torch.tensor([1],          dtype=torch.long)
            X, _ = model.generate(src, src_lens, max_T=MAX_T_OUT, stop_thresh=STOP_THRESH)
            word_frames = _tensor_to_frames(X)

        per_word_frames.append(word_frames)
        gloss_words.append(gloss)
        logger.info(f"  [{gloss}] id={gloss_id} → {len(word_frames)} frames")

    # ── Step 2: smooth concatenation with inserted transition frames ─────────
    all_frames, raw_segments = _smooth_concatenate(per_word_frames, transition_frames)

    # ── Step 3: attach word names to segments ────────────────────────────────
    word_segments = [
        {
            "word":        gloss_words[i],
            "start_frame": raw_segments[i]["start_frame"],
            "end_frame":   raw_segments[i]["end_frame"],
        }
        for i in range(len(gloss_words))
    ]

    logger.info(
        f"run_inference_per_word: {len(words_and_ids)} words → "
        f"{len(all_frames)} frames "
        f"({len(words_and_ids)-1} transitions × {transition_frames} frames each)"
    )
    return all_frames, word_segments

def model_is_loaded() -> bool:
    return _load_model() is not None