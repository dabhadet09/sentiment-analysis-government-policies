import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt
from youtube_fetcher import fetch_youtube_comments

# Load model and vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z_\s]", "", text)
    return text

st.set_page_config(page_title="Live Public Opinion Analysis", layout="centered")

st.title("🗳️ Live Public Opinion Analysis on Government Policies")
st.write("YouTube Comment Sentiment Analysis using Machine Learning")

st.markdown("---")

topic = st.text_input("Enter Government Policy / Topic:")

if st.button("Fetch YouTube Comments & Analyze"):
    if topic.strip() == "":
        st.warning("⚠️ Please enter a topic")
    else:
        with st.spinner("Fetching YouTube comments..."):
            comments = fetch_youtube_comments(topic)

        if len(comments) == 0:
            st.error("No comments found.")
        else:
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            results = []

            for comment in comments:
                clean = preprocess_text(comment)
                vector = vectorizer.transform([clean])
                prediction = model.predict(vector)[0]
                sentiment_counts[prediction] += 1
                results.append((comment, prediction))

            df = pd.DataFrame(results, columns=["YouTube Comment", "Predicted Sentiment"])
            st.dataframe(df, use_container_width=True)

            st.markdown("### 📊 Sentiment Summary")
            st.write(sentiment_counts)

            # Pie chart
            fig1 = plt.figure()
            plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(),
                    autopct="%1.1f%%", startangle=90)
            plt.title("Sentiment Distribution")
            st.pyplot(fig1)

            # Bar chart
            fig2 = plt.figure()
            plt.bar(sentiment_counts.keys(), sentiment_counts.values())
            plt.title("Sentiment Count")
            st.pyplot(fig2)

st.markdown("---")
st.caption("BE Final Year Project | YouTube-based Public Opinion Analysis")
