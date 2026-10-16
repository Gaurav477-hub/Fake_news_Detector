"""Shared text-preprocessing utilities for the fake-news project."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


PROJECT_ROOT = Path(__file__).resolve().parent
NLTK_DATA_DIRECTORY = PROJECT_ROOT / "nltk_data"

if NLTK_DATA_DIRECTORY.exists():
    nltk.data.path.insert(0, str(NLTK_DATA_DIRECTORY))


@lru_cache(maxsize=1)
def _get_preprocessing_components() -> tuple[set[str], WordNetLemmatizer]:
    """Load the NLTK resources bundled during the deployment build."""
    for resource_path, package_name in (
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ):
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.data.find(f"{resource_path}.zip")
            except LookupError:
                raise RuntimeError(
                    f"Missing NLTK resource '{package_name}'. Run `python build.py` before starting the app."
                ) from None

    stop_words = set(stopwords.words("english"))
    stop_words.difference_update({"no", "nor", "not"})
    return stop_words, WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """Clean, tokenize, remove stopwords, and lemmatize article text."""
    stop_words, lemmatizer = _get_preprocessing_components()

    normalized_text = str(text or "").lower()
    normalized_text = re.sub(r"https?://\S+|www\.\S+", " ", normalized_text)
    tokens = re.findall(r"[a-z]+", normalized_text)

    clean_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    return " ".join(clean_tokens)
