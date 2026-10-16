"""Flask application for Vercel deployment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
from flask import Flask, jsonify, request, send_from_directory

from text_processing import preprocess_text


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "fake_news_model.joblib"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"
LABEL_NAMES = {0: "Fake", 1: "Real"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024


@lru_cache(maxsize=1)
def load_artifacts():
    """Load prediction artifacts once for each function instance."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Trained model artifacts are missing from the models directory.")

    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def get_probability(model, features, prediction: int) -> float | None:
    """Return a probability only when the selected model supports it."""
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(features)[0]
    class_index = list(model.classes_).index(prediction)
    return float(probabilities[class_index])


@app.get("/")
def index():
    """Serve the browser interface."""
    return send_from_directory(PROJECT_ROOT / "public", "index.html")


@app.get("/health")
def health_check():
    """Report whether the application can load its model artifacts."""
    try:
        load_artifacts()
        preprocess_text("health check")
        return jsonify({"status": "ok"})
    except (FileNotFoundError, RuntimeError) as error:
        return jsonify({"status": "error", "message": str(error)}), 503


@app.post("/api/predict")
def predict():
    """Classify submitted news title and text as fake or real."""
    payload = request.get_json(silent=True) or {}
    title = payload.get("title", "")
    article_text = payload.get("text", "")

    if not isinstance(title, str) or not isinstance(article_text, str):
        return jsonify({"error": "The title and article text must be strings."}), 400

    input_text = f"{title} {article_text}".strip()
    if not input_text:
        return jsonify({"error": "Enter a news headline or article text."}), 400

    cleaned_text = preprocess_text(input_text)
    if not cleaned_text:
        return jsonify({"error": "Enter meaningful English words for a prediction."}), 400

    try:
        model, vectorizer = load_artifacts()
        features = vectorizer.transform([cleaned_text])
        prediction_label = int(model.predict(features)[0])
        result = LABEL_NAMES[prediction_label]
        probability = get_probability(model, features, prediction_label)
    except (FileNotFoundError, RuntimeError) as error:
        return jsonify({"error": str(error)}), 503

    response = {
        "prediction": result,
        "message": f"Possibly {result} News",
        "disclaimer": (
            "This result is based on learned text patterns and is not independent factual verification."
        ),
    }
    if probability is not None:
        response["probability"] = round(probability, 4)

    return jsonify(response)


@app.errorhandler(413)
def request_too_large(_error):
    """Return JSON when a submitted article exceeds the request limit."""
    return jsonify({"error": "The submitted article is too long."}), 413
