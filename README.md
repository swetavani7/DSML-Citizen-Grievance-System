
# AI-Driven Citizen Grievance & Sentiment Analysis System

## Run Project

### Create Virtual Environment
python -m venv venv

### Activate Environment

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

### Install Requirements
pip install -r requirements.txt

### Train Models
python train_model.py
python sentiment_train.py

### Run FastAPI
uvicorn app.main:app --reload

### Swagger Docs
http://127.0.0.1:8000/docs
