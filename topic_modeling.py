import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load dataset
df = pd.read_csv("data/raw/grievances.csv")

print("Dataset Loaded")
print(df.head())

# Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df["cleaned_text"] = df["complaint"].apply(clean_text)

# Convert text to vectors
vectorizer = CountVectorizer(
    stop_words="english"
)

X = vectorizer.fit_transform(
    df["cleaned_text"]
)

# Topic Modeling
lda = LatentDirichletAllocation(
    n_components=4,
    random_state=42
)

lda.fit(X)

# Display Topics
feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):

    print(f"\nTopic {topic_idx + 1}")

    top_words = [
        feature_names[i]
        for i in topic.argsort()[-10:]
    ]

    print(top_words)

# Assign Topic
topic_results = lda.transform(X)

df["predicted_topic"] = topic_results.argmax(axis=1)

print("\nTopic Assignment")

print(
    df[
        ["complaint", "predicted_topic"]
    ].head()
)

# Save Results
df.to_csv(
    "data/processed/topic_results.csv",
    index=False
)

print("\nResults Saved")

topic_counts = df["predicted_topic"].value_counts()

topic_counts.plot(
    kind="bar",
    title="Topic Distribution"
)

plt.show()