"""Train and save fake-news classification models using the ISOT dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from text_processing import preprocess_text


LABEL_NAMES = {0: "Fake", 1: "Real"}


def load_isot_dataset(fake_path: Path, true_path: Path) -> pd.DataFrame:
    """Load ISOT CSV files, add labels, and return clean training records."""
    fake_news = pd.read_csv(fake_path)
    true_news = pd.read_csv(true_path)

    required_columns = {"title", "text"}
    for dataset_name, dataset in (("Fake.csv", fake_news), ("True.csv", true_news)):
        missing_columns = required_columns.difference(dataset.columns)
        if missing_columns:
            raise ValueError(
                f"{dataset_name} is missing required columns: {sorted(missing_columns)}"
            )

    fake_news["label"] = 0
    true_news["label"] = 1
    dataset = pd.concat([fake_news, true_news], ignore_index=True)

    dataset["title"] = dataset["title"].fillna("").astype(str)
    dataset["text"] = dataset["text"].fillna("").astype(str)
    dataset = dataset[dataset["text"].str.strip().ne("")].copy()
    dataset = dataset.drop_duplicates(subset=["title", "text"])

    dataset["content"] = (dataset["title"] + " " + dataset["text"]).str.strip()
    dataset["clean_text"] = dataset["content"].map(preprocess_text)
    dataset = dataset[dataset["clean_text"].str.strip().ne("")].reset_index(drop=True)
    return dataset


def train_and_save(fake_path: Path, true_path: Path, model_directory: Path) -> None:
    """Train candidate classifiers, select the best one, and save artifacts."""
    dataset = load_isot_dataset(fake_path, true_path)
    print(f"Training records after cleaning: {len(dataset):,}")
    print(dataset["label"].map(LABEL_NAMES).value_counts().to_string())

    train_text, test_text, train_labels, test_labels = train_test_split(
        dataset["clean_text"],
        dataset["label"],
        test_size=0.20,
        random_state=42,
        stratify=dataset["label"],
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=30000,
    )
    train_features = vectorizer.fit_transform(train_text)
    test_features = vectorizer.transform(test_text)

    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=40,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = []
    trained_models = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)
        trained_models[model_name] = model
        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(test_labels, predictions),
                "Precision": precision_score(
                    test_labels, predictions, average="macro", zero_division=0
                ),
                "Recall": recall_score(
                    test_labels, predictions, average="macro", zero_division=0
                ),
                "F1-score": f1_score(
                    test_labels, predictions, average="macro", zero_division=0
                ),
            }
        )

    results_frame = pd.DataFrame(results).sort_values("F1-score", ascending=False)
    best_model_name = results_frame.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    model_directory.mkdir(parents=True, exist_ok=True)
    results_frame.to_csv(model_directory / "model_results.csv", index=False)
    joblib.dump(best_model, model_directory / "fake_news_model.joblib")
    joblib.dump(vectorizer, model_directory / "tfidf_vectorizer.joblib")

    metadata = {
        "best_model": best_model_name,
        "label_mapping": {str(label): name for label, name in LABEL_NAMES.items()},
        "test_metrics": json.loads(results_frame.to_json(orient="records")),
    }
    (model_directory / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nModel comparison:")
    print(results_frame.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nSaved best model: {best_model_name}")
    print(f"Artifacts saved to: {model_directory.resolve()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fake-news models from the ISOT dataset."
    )
    parser.add_argument("--fake-path", default="data/Fake.csv")
    parser.add_argument("--true-path", default="data/True.csv")
    parser.add_argument("--model-directory", default="models")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    train_and_save(
        Path(arguments.fake_path),
        Path(arguments.true_path),
        Path(arguments.model_directory),
    )
