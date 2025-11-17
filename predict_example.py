# predict_example.py

import joblib
from src.preprocess import preprocess_spacy

VEC_PATH = "tfidf_vectorizer.joblib"
MODEL_PATH = "logreg_model.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"

vec = joblib.load(VEC_PATH)
clf = joblib.load(MODEL_PATH)
le = joblib.load(LABEL_ENCODER_PATH)


def predict_text(text: str) -> str:
    clean = preprocess_spacy(text)
    X = vec.transform([clean])
    y_pred = clf.predict(X)
    label = le.inverse_transform(y_pred)[0]
    return label


if __name__ == "__main__":
    print("Twitter Sentiment Predictor (type 'q' to quit)")
    while True:
        txt = input("Enter a tweet: ")
        if txt.lower().strip() == "q":
            break
        print("Predicted sentiment:", predict_text(txt))
        print("-" * 40)

