import streamlit as st
import joblib
from src.preprocess import preprocess_spacy

# --------- Paths to saved artifacts ----------
VEC_PATH = "tfidf_vectorizer.joblib"
MODEL_PATH = "logreg_model.joblib"
LABEL_ENCODER_PATH = "label_encoder.joblib"


@st.cache_resource
def load_artifacts():
    vec = joblib.load(VEC_PATH)
    clf = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    return vec, clf, le


vec, clf, le = load_artifacts()

# --------- GLOBAL CSS: background + animation ---------
st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 0;
        padding-right: 0;
    }

    body, .stApp {
        background: radial-gradient(circle at 0% 0%, #1DA1F2 0, #15202B 55%, #020617 100%);
        color: #ffffff;
        font-family: "Segoe UI", sans-serif;
    }

    /* ================= HEADER STRIP (animated “black box”) ================= */
    .twitter-header {
        max-width: 900px;
        margin: 1.8rem auto 0.5rem auto;
        padding: 1.1rem 1.6rem;
        border-radius: 24px;
        position: relative;
        overflow: hidden;
        background: #020617;
        border: 1px solid rgba(148, 163, 184, 0.4);
    }

    .twitter-header::before {
        content: "";
        position: absolute;
        inset: -40%;
        background: conic-gradient(
            from 0deg,
            #1DA1F2,
            #22c55e,
            #eab308,
            #f97373,
            #1DA1F2
        );
        opacity: 0.25;
        animation: spinGradient 10s linear infinite;
    }

    .twitter-header-inner {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .twitter-header-icon {
        font-size: 1.8rem;
        filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.85));
    }

    .twitter-header-title {
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #e5e7eb;
    }

    .twitter-header-sub {
        font-size: 0.8rem;
        color: #9ca3af;
    }

    @keyframes spinGradient {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* ================= MAIN CARD ================= */
    .twitter-card {
        max-width: 720px;
        margin: 1.5rem auto 3rem auto;
        padding: 2.5rem 2.2rem;
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.90);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.45);
        backdrop-filter: blur(18px);
    }

    .twitter-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .twitter-subtitle {
        font-size: 0.95rem;
        color: #9ca3af;
        margin-bottom: 1.8rem;
    }

    .twitter-icon {
        font-size: 2rem;
    }

    textarea {
        border-radius: 18px !important;
        border: 1px solid #1DA1F2 !important;
        box-shadow: 0 0 0 1px rgba(29, 161, 242, 0.25);
    }

    .stButton>button {
        background: #1DA1F2;
        color: #ffffff;
        border-radius: 999px;
        padding: 0.4rem 1.8rem;
        border: none;
        font-weight: 600;
        transition: transform 0.12s ease-out, box-shadow 0.12s ease-out,
                    background 0.12s ease-out;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 22px rgba(29, 161, 242, 0.45);
        background: #1a91da;
    }

    @keyframes fadeSlideUp {
        0% { opacity: 0; transform: translateY(8px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .prediction-box {
        margin-top: 1.4rem;
        padding: 0.9rem 1.1rem;
        border-radius: 16px;
        background: rgba(15, 118, 110, 0.12);
        border: 1px solid rgba(45, 212, 191, 0.5);
        animation: fadeSlideUp 0.28s ease-out;
    }

    .prediction-label {
        font-weight: 700;
        color: #e5e7eb;
    }

    .prediction-value {
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .sentiment-positive { color: #22c55e; }
    .sentiment-negative { color: #f97373; }
    .sentiment-neutral  { color: #eab308; }

    </style>
    """,
    unsafe_allow_html=True,
)

# --------- Animated header (uses the “black box” area) ---------
st.markdown(
    """
    <div class="twitter-header">
        <div class="twitter-header-inner">
            <span class="twitter-header-icon">🐦</span>
            <div>
                <div class="twitter-header-title">Live Tweet Sentiment</div>
                <div class="twitter-header-sub">
                    Powered by Python · spaCy · Logistic Regression
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)



st.markdown(
    """
    <div class="twitter-title">
        <span class="twitter-icon">🐦</span>
        <span>Twitter Sentiment Analysis</span>
    </div>
    <div class="twitter-subtitle">
        Type a tweet below and see if it sounds positive, negative, or neutral.
    </div>
    """,
    unsafe_allow_html=True,
)

tweet = st.text_area("Enter a tweet:", "", height=150)

pred_label = None
if st.button("Predict sentiment"):
    if tweet.strip():
        clean = preprocess_spacy(tweet)
        X = vec.transform([clean])
        y_pred = clf.predict(X)
        pred_label = le.inverse_transform(y_pred)[0]
    else:
        st.warning("Please type something first 😊")

if pred_label is not None:
    sentiment_class = {
        "positive": "sentiment-positive",
        "negative": "sentiment-negative",
        "neutral": "sentiment-neutral",
    }.get(pred_label, "sentiment-neutral")

    st.markdown(
        f"""
        <div class="prediction-box">
            <span class="prediction-label">Predicted sentiment:&nbsp;</span>
            <span class="prediction-value {sentiment_class}">{pred_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)
