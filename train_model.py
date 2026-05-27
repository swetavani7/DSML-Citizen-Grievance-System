
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

from app.preprocessing import clean_text

df = pd.read_csv("data/raw/grievances.csv")

df["cleaned"] = df["complaint"].apply(clean_text)

X = df["cleaned"]
y = df["department"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

print(classification_report(y_test, predictions))

joblib.dump(pipeline, "models/department_model.pkl")

print("Department model saved successfully.")
