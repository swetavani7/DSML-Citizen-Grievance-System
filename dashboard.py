import streamlit as st
import joblib
import pandas as pd

st.title("AI Citizen Grievance System")

st.write("Government Complaint Classification & Sentiment Analysis")

# Load models safely
try:
    department_model = joblib.load("models/department_model.pkl")
    sentiment_model = joblib.load("models/sentiment_model.pkl")

    st.success("Models loaded successfully")

except Exception as e:
    st.error(f"Model Loading Error: {e}")

# Input box
complaint = st.text_area("Enter Citizen Complaint")

# Predict
if st.button("Analyze Complaint"):

    if complaint.strip() == "":
        st.warning("Please enter complaint")

    else:

        try:
            print("\n========== NEW COMPLAINT ==========")
            print(f"Complaint: {complaint}")
            with open("complaint_logs.txt", "a") as file:
                file.write(f"Complaint: {complaint}\n")
                file.write(f"Department: {department}\n")
                file.write(f"Sentiment: {sentiment}\n")
                file.write(f"Priority: {priority_score}\n")
                file.write("---------------------------------\n")
            # Prediction
            department = department_model.predict([complaint])[0]
            sentiment = sentiment_model.predict([complaint])[0]

            priority_map = {
                "Positive": 1,
                "Neutral": 3,
                "Negative": 7,
                "Critical": 10
            }

            priority_score = priority_map.get(sentiment, 0)
            print(f"Department: {department}")
            print(f"Sentiment: {sentiment}")
            print(f"Priority Score: {priority_score}")
            print("===================================\n")

            st.success("Analysis Completed")

            st.write(f"Department: {department}")
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Priority Score: {priority_score}")

        except Exception as e:
            print(f"Error: {e}")
            st.error(f"Prediction Error: {e}")

st.subheader("Complaint Analytics Dashboard")

# Sample department data
department_data = pd.DataFrame({
    "Department": ["Water", "Roads", "Electricity", "Sanitation"],
    "Complaints": [45, 30, 25, 20]
})

# Bar Chart
st.write("Department-wise Complaints")

st.bar_chart(
    department_data.set_index("Department")
)

# Sample sentiment data
sentiment_data = pd.DataFrame({
    "Sentiment": ["Positive", "Neutral", "Negative", "Critical"],
    "Count": [10, 20, 40, 15]
})

# Line Chart
st.write("Sentiment Trends")

st.line_chart(
    sentiment_data.set_index("Sentiment")
)           