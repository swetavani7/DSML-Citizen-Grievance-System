from pathlib import Path
import joblib
from fastapi import FastAPI

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

department_model = joblib.load(
    BASE_DIR / "models" / "department_classifier.pkl"
)

vectorizer = joblib.load(
    BASE_DIR / "models" / "tfidf_vectorizer.pkl"
)

@app.get("/")
def home():
    return {
        "message": "Citizen Grievance API"
    }

@app.post("/predict")
def predict(data: dict):

    try:
        complaint = data["complaint"]

        print("Complaint:", complaint)

        vector = vectorizer.transform([complaint])

        print("Vector created")

        department = department_model.predict(vector)[0]

        print("Prediction:", department)

        return {
            "department": str(department)
        }

    except Exception as e:
        print("ERROR:", str(e))

        return {
            "error": str(e)
        }