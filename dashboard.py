import streamlit as st
import pandas as pd
import joblib

department_model = joblib.load(
    "models/department_classifier.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

st.title("AI Citizen Grievance System")

df = pd.read_csv(
    "data/processed/cleaned_grievances.csv"
)

st.subheader("Department Analytics")

department_counts = (
    df["department"]
    .value_counts()
)

st.bar_chart(department_counts)

st.subheader("Sentiment Analytics")

sentiment_counts = (
    df["sentiment"]
    .value_counts()
)

st.bar_chart(sentiment_counts)

st.subheader("Complaint Records")

st.dataframe(df)

st.subheader("Project Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Complaints",
        len(df)
    )

with col2:
    st.metric(
        "Departments",
        df["department"].nunique()
    )

with col3:
    st.metric(
        "Sentiment Classes",
        df["sentiment"].nunique()
    )

st.subheader("AI Complaint Prediction")

complaint = st.text_area(
    "Enter Citizen Complaint"
)

if st.button("Predict Department"):

    vector = vectorizer.transform(
        [complaint]
    )

    prediction = department_model.predict(
        vector
    )[0]

    st.success(
        f"Predicted Department: {prediction}"
    )