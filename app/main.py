
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from app.preprocessing import clean_text

app = FastAPI()

department_model = joblib.load("models/department_model.pkl")
sentiment_model = joblib.load("models/sentiment_model.pkl")


class ComplaintRequest(BaseModel):
    complaint: str


@app.get("/")
def home():
    return {"message": "Citizen Grievance NLP API Running"}


@app.post("/predict")
def predict(request: ComplaintRequest):

    cleaned = clean_text(request.complaint)

    department = department_model.predict([cleaned])[0]
    sentiment = sentiment_model.predict([cleaned])[0]

    priority_map = {
        "Positive": 1,
        "Neutral": 3,
        "Negative": 7,
        "Critical": 10
    }

    priority_score = priority_map.get(sentiment, 0)

    return {
        "department": department,
        "sentiment": sentiment,
        "priority_score": priority_score
    }
