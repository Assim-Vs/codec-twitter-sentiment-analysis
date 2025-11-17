

# src/preprocess.py

import re
import spacy

# Load spaCy English model once
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def clean_text(text: str) -> str:
    """
    Basic cleaning:
    - lowercase
    - remove URLs, mentions, hashtag symbol, punctuation
    - normalize whitespace
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)      # remove urls
    text = re.sub(r"@\w+", "", text)         # remove mentions
    text = re.sub(r"#", "", text)            # remove hashtag symbol (#love -> love)
    text = re.sub(r"[^\w\s]", " ", text)     # remove punctuation
    text = re.sub(r"\s+", " ", text).strip() # normalize spaces
    return text


def preprocess_spacy(text: str) -> str:
    """
    Clean + lemmatize using spaCy, remove stopwords and non-alphabetic tokens.
    Returns a cleaned string ready for TF-IDF.
    """
    text = clean_text(text)
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return " ".join(tokens)
