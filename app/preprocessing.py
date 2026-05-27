
import re
import nltk
import spacy
from nltk.corpus import stopwords

nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)

    doc = nlp(text)

    tokens = []

    for token in doc:
        if token.text not in stop_words:
            tokens.append(token.lemma_)

    return " ".join(tokens)
