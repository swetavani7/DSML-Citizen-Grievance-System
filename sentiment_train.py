
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib

from app.preprocessing import clean_text

df = pd.read_csv("data/raw/grievances.csv")

df["cleaned"] = df["complaint"].apply(clean_text)

X = df["cleaned"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LinearSVC())
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "models/sentiment_model.pkl")

print("Sentiment model saved successfully.")
