# Fake News Detection

A mini-project that classifies English news text as **Fake** or **Real** using the ISOT Fake News Dataset, TF-IDF features, and machine-learning classifiers.

## Project workflow

1. Load `Fake.csv` and `True.csv` from the ISOT dataset.
2. Combine the headline and article text.
3. Tokenize, remove stopwords, and lemmatize the text.
4. Convert processed text into TF-IDF features.
5. Train Naive Bayes, SVM, and Random Forest models.
6. Select the model with the best macro F1-score.
7. Use the saved model in a Streamlit prediction app.

## Setup

Create this folder structure and place the ISOT CSV files inside `data`:

```text
Fake_News_Detector/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── app.py
├── sample_predictions.py
├── text_processing.py
├── train_model.py
└── requirements.txt
```

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train and save the model

```powershell
python train_model.py
```

This creates these files in `models`:

```text
fake_news_model.joblib
tfidf_vectorizer.joblib
model_metadata.json
model_results.csv
```

## Run the application

```powershell
streamlit run app.py
```

Enter a headline and/or article body, then select **Check News**.

## Test sample predictions

After training, run:

```powershell
python sample_predictions.py
```

To test your own text:

```powershell
python sample_predictions.py --text "Paste a news headline or article here"
```

The prediction indicates patterns learned from the dataset; it does not independently verify whether a claim is factually true.
