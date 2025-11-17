# Twitter Sentiment Analysis 🐦

A simple machine learning web app built with **Python**, **spaCy**, **scikit-learn**, and **Streamlit** to classify tweet sentiment as **positive**, **negative**, or **neutral**.

## Features

- Clean Twitter-style UI with animated header
- Text preprocessing using spaCy (tokenization, lemmatization, stopword removal)
- TF-IDF vectorization
- Logistic Regression sentiment classifier
- Streamlit web interface for easy interaction

## Tech Stack

- Python 3.11
- Streamlit
- spaCy (`en_core_web_sm`)
- scikit-learn
- pandas, numpy
- joblib

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis

python -m venv venv
venv\Scripts\activate   # on Windows

pip install -r requirements.txt
python -m spacy download en_core_web_sm
Run the app
streamlit run app.py
