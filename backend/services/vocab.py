"""
Gloss → ID Mapping Service

Loads word2id.pkl once at startup and maps gloss tokens to integer IDs.
Unknown tokens are handled via a configurable strategy (skip / UNK token).
"""

import pickle
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — update path to match where you store model artefacts
# ---------------------------------------------------------------------------
WORD2ID_PATH = Path(__file__).parent.parent / "models" / "word2id.pkl"

# Training used PAD_GLOSS=0 with no special UNK token.
# word2id assigns IDs from 1 upward (alphabetically sorted).
# Unknown glosses are skipped — never mapped to 0 (that's the padding token).
UNK_TOKEN = None   # no UNK in this vocab
PAD_TOKEN = None   # padding is handled by the model internally (PAD_GLOSS=0)


@lru_cache(maxsize=1)
def _load_vocab() -> dict[str, int]:
    """Load and cache the word2id mapping from disk."""
    if not WORD2ID_PATH.exists():
        logger.warning(
            f"word2id.pkl not found at {WORD2ID_PATH}. "
            "Using empty vocab (mock mode)."
        )
        return _mock_vocab()

    with open(WORD2ID_PATH, "rb") as f:
        vocab: dict = pickle.load(f)

    logger.info(f"Loaded vocab with {len(vocab)} entries from {WORD2ID_PATH}")
    return vocab


def _mock_vocab() -> dict[str, int]:
    """
    Fallback vocab used when word2id.pkl is absent.
    Replace with your real file in production.
    """
    words = [
        PAD_TOKEN, UNK_TOKEN,
        "HELLO", "WORLD", "I", "YOU", "WE", "THEY", "HE", "SHE",
        "WHAT", "WHERE", "WHEN", "WHY", "HOW",
        "GOOD", "BAD", "HAPPY", "SAD", "THANK", "SORRY",
        "HELP", "WANT", "NEED", "HAVE", "LIKE", "LOVE",
        "GO", "COME", "SEE", "KNOW", "THINK", "UNDERSTAND",
        "YES", "NO", "PLEASE", "MORE", "STOP", "WAIT",
        "NAME", "AGE", "SCHOOL", "WORK", "FOOD", "WATER", "HOME",
        "AM", "GOING", "STORE", "TOMORROW", "TODAY", "NOW",
    ]
    return {w: idx for idx, w in enumerate(words)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_vocab_size() -> int:
    return len(_load_vocab())


def glosses_to_ids(
    glosses: list[str],
    skip_unknown: bool = False,
) -> tuple[list[int], list[str]]:
    """
    Map a list of gloss strings to their integer IDs.

    Args:
        glosses:       List of gloss words (uppercase).
        skip_unknown:  If True, drop tokens not in vocab.
                       If False (default), map them to UNK id.

    Returns:
        (id_sequence, oov_list)
        id_sequence — integer IDs ready for model input
        oov_list    — gloss words that were out-of-vocabulary
    """
    vocab = _load_vocab()
    ids: list[int] = []
    oov: list[str] = []

    for gloss in glosses:
        if gloss in vocab:
            ids.append(vocab[gloss])
        else:
            oov.append(gloss)
            # Always skip unknown glosses — vocab has no UNK token,
            # and 0 is the padding ID which must not appear in input.

    if oov:
        logger.debug(f"OOV glosses: {oov}")

    return ids, oov

# Add after glosses_to_ids()

def glosses_to_word_id_pairs(
    glosses: list[str],
) -> tuple[list[tuple[str, int]], list[str]]:
    """
    Like glosses_to_ids() but returns (word, id) pairs instead of just ids.
    OOV words are skipped (never passed to the model).

    Returns:
        (pairs, oov_list)
        pairs    — [(gloss_word, gloss_id), ...] for known words only
        oov_list — words that were not found in vocab
    """
    vocab = _load_vocab()
    pairs: list[tuple[str, int]] = []
    oov:   list[str]             = []

    for gloss in glosses:
        if gloss in vocab:
            pairs.append((gloss, vocab[gloss]))
        else:
            oov.append(gloss)

    if oov:
        logger.debug(f"OOV glosses (skipped): {oov}")

    return pairs, oov