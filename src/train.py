# src/train.py

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

from src.preprocess import preprocess_spacy  # if this ever breaks, use: from .preprocess import preprocess_spacy


# 1) Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tweets_labeled.csv")

VEC_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "logreg_model.joblib")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.joblib")


def main():
    # 2) Load data
    print(f"Loading data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV not found at {DATA_PATH}. Make sure 'tweets_labeled.csv' is in the 'data' folder.")

    df = pd.read_csv(DATA_PATH)

    # Expecting columns: 'content' (text) and 'label' (sentiment)
    required_cols = {"content", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}. Found: {df.columns.tolist()}")

    print("Data shape:", df.shape)
    print("Label distribution:")
    print(df["label"].value_counts())

    # 3) Preprocess text
    print("Preprocessing text with spaCy... (this may take a bit)")
    df["clean"] = df["content"].astype(str).apply(preprocess_spacy)

    # 4) TF-IDF vectorization
    print("Vectorizing text with TF-IDF...")
    vec = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)  # unigrams + bigrams
    )
    X = vec.fit_transform(df["clean"])

    # 5) Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    # 6) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=None
    )

    # 7) Train model
    print("Training Logistic Regression model...")
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # 8) Evaluate
    print("\nClassification report:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix (will pop up a window)
    disp = ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y_test, display_labels=le.classes_
    )
    plt.title("Confusion Matrix - Twitter Sentiment")
    plt.tight_layout()
    plt.show()

    # 9) Save vectorizer, model, and label encoder
    print(f"Saving vectorizer to: {VEC_PATH}")
    joblib.dump(vec, VEC_PATH)

    print(f"Saving model to: {MODEL_PATH}")
    joblib.dump(clf, MODEL_PATH)

    print(f"Saving label encoder to: {LABEL_ENCODER_PATH}")
    joblib.dump(le, LABEL_ENCODER_PATH)

    print("\nAll done! Model and vectorizer saved.")


if __name__ == "__main__":
    main()
