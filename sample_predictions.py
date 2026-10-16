"""Run sample fake-news predictions using the saved model artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from text_processing import preprocess_text


MODEL_DIRECTORY = Path("models")
LABEL_NAMES = {0: "Fake", 1: "Real"}

SAMPLE_ARTICLES = [
    "The local government released its annual public budget report today.",
    "A viral post claims that a household item can cure every disease overnight.",
]


def predict_news(text: str) -> str:
    """Return the model label for one new article or headline."""
    model = joblib.load(MODEL_DIRECTORY / "fake_news_model.joblib")
    vectorizer = joblib.load(MODEL_DIRECTORY / "tfidf_vectorizer.joblib")
    cleaned_text = preprocess_text(text)

    if not cleaned_text:
        raise ValueError("Provide meaningful English text for prediction.")

    features = vectorizer.transform([cleaned_text])
    prediction = int(model.predict(features)[0])
    return LABEL_NAMES[prediction]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fake or real news text.")
    parser.add_argument("--text", help="One article or headline to classify.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    texts_to_test = [arguments.text] if arguments.text else SAMPLE_ARTICLES

    for index, article in enumerate(texts_to_test, start=1):
        print(f"Sample {index}: {predict_news(article)}")
        print(f"Text: {article}\n")
