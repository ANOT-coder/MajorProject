"""
Text → ASL Gloss Preprocessing
================================
Uses the project's own english_to_asl_gloss() logic (from data_preprocessing.py).

Requires:
  pip install spacy
  python -m spacy download en_core_web_sm
"""

import re
import logging
import spacy

logger = logging.getLogger(__name__)

# Load once at import time
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy en_core_web_sm loaded.")
except OSError:
    raise RuntimeError(
        "spaCy model not found. Run:\n"
        "  python -m spacy download en_core_web_sm"
    )

# ─── Constants (copied exactly from data_preprocessing.py) ───────────────────

STOP_WORDS = {
    "a", "an", "the",
    "am", "is", "are", "was", "were",
    "be", "been", "being",
    "do", "does", "did",
    "to",
    "this", "that", "these", "those",
    "please"
}

TIME_WORDS = {
    "today", "tomorrow", "yesterday",
    "now", "later", "tonight",
    "morning", "evening", "night",
    "week", "month", "year"
}

ORDINAL_WORDS = {
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth"
}

NUMBER_MAP = {
    "0": "ZERO",  "1": "ONE",   "2": "TWO",    "3": "THREE",
    "4": "FOUR",  "5": "FIVE",  "6": "SIX",    "7": "SEVEN",
    "8": "EIGHT", "9": "NINE",  "10": "TEN",   "11": "ELEVEN",
    "12": "TWELVE", "13": "THIRTEEN", "14": "FOURTEEN",
    "15": "FIFTEEN", "16": "SIXTEEN", "17": "SEVENTEEN",
    "18": "EIGHTEEN", "19": "NINETEEN", "20": "TWENTY"
}

# ─── Helpers (copied exactly from data_preprocessing.py) ─────────────────────

def remove_special_characters(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_contractions(text: str) -> str:
    contractions = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "can't": "cannot", "won't": "will not",
        "shouldn't": "should not", "wouldn't": "would not",
        "couldn't": "could not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "hasn't": "has not",
        "haven't": "have not", "hadn't": "had not"
    }
    for c, e in contractions.items():
        text = re.sub(rf"\b{c}\b", e, text, flags=re.IGNORECASE)
    # 's → is
    text = re.sub(r"(\w+)'s\b", r"\1 is", text)
    return text

def number_to_gloss(num: str) -> str:
    if num in NUMBER_MAP:
        return NUMBER_MAP[num]
    if len(num) == 2:
        return " ".join(NUMBER_MAP.get(d, d) for d in num)
    return num

def english_to_asl_gloss(sentence: str) -> str:
    """Copied verbatim from data_preprocessing.py."""
    sentence = remove_special_characters(sentence)
    sentence = preprocess_contractions(sentence)
    doc = nlp(sentence)

    time_tokens = []
    subjects    = []
    noun_phrases = []
    verbs       = []
    others      = []
    negation    = []
    current_np  = []

    for token in doc:
        word = token.text.lower()

        if token.is_punct:
            continue
        if word in TIME_WORDS:
            time_tokens.append(word.upper())
            continue
        if word in STOP_WORDS and token.pos_ in ["DET", "AUX", "PART"]:
            continue
        if token.dep_ == "poss":
            current_np.append(token.text.upper())
            continue
        if word in ORDINAL_WORDS:
            current_np.append(token.text.upper())
            continue
        if token.like_num:
            others.append(number_to_gloss(token.text))
            continue
        if token.pos_ == "NOUN":
            current_np.append(token.text.upper())
            noun_phrases.append(" ".join(current_np))
            current_np = []
            continue
        if token.dep_ in ["nsubj", "nsubjpass"]:
            subjects.append(token.text.upper())
            continue
        if token.pos_ in ["VERB", "AUX"]:
            verbs.append(token.lemma_.upper())
            continue
        if token.lemma_.upper() == "NOT":
            negation.append("NOT")
            continue

        others.append(token.text.upper())

    gloss = time_tokens + subjects + noun_phrases + verbs + others + negation
    return " ".join(gloss)

# ─── Public API (called by routers/translate.py) ──────────────────────────────

def text_to_gloss(text: str) -> list[str]:
    """
    Entry point for the backend pipeline.
    Returns a list of uppercase gloss tokens.

    Example:
        text_to_gloss("She's going to school tomorrow.")
        → ["TOMORROW", "SHE", "SCHOOL", "GO"]
    """
    if not text or not text.strip():
        return []

    gloss_str = english_to_asl_gloss(text.strip())
    logger.debug(f"[preprocessor] '{text}' → '{gloss_str}'")

    # Split into list, drop any empty strings
    return [g for g in gloss_str.split() if g]