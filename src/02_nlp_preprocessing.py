import re
import spacy
import pandas as pd

# Load SpaCy Spanish model
nlp = spacy.load("es_core_news_sm")

# Custom stopwords for audio/call-center context
CALLCENTER_STOPWORDS = {
    "eh", "ah", "este", "pues", "bueno", "ok", "okay", "mmm",
    "aja", "ajá", "vale", "sí", "no", "ya"
}

SPANISH_STOPS = nlp.Defaults.stop_words.union(CALLCENTER_STOPWORDS)

# Date patterns to remove noise
DATE_PATTERNS = [
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
    r"\b\d{1,2}\s+[a-zA-Z]+\s+\d{4}\b",
    r"\b[a-zA-Z]+\s+\d{1,2},\s*\d{4}\b"
]


def clean_basic(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # remove dates
    for pattern in DATE_PATTERNS:
        text = re.sub(pattern, " ", text)

    # remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # remove numbers
    text = re.sub(r"\d+", " ", text)

    # remove special characters
    text = re.sub(r"[^\w\sáéíóúüñ]", " ", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text):
    text = clean_basic(text)
    doc = nlp(text)

    tokens = []

    for token in doc:
        if token.is_stop:
            continue
        lemma = token.lemma_.strip()
        if len(lemma) < 3:
            continue
        tokens.append(lemma)

    return " ".join(tokens)


def nlp_preprocess(df, column="transcript"):
    df["processed"] = df[column].apply(preprocess_text)
    df["token_count"] = df["processed"].apply(lambda x: len(x.split()))
    return df
