import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# Load cleaned dataset
df = pd.read_csv("cleaned_reddit_sentiment.csv")

X = df["clean_comment"]
y = df["sentiment"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = vectorizer.fit_transform(X)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ TF-IDF Feature Extraction Completed")
print("📌 Total Features:", X_tfidf.shape[1])
print("📌 Training Samples:", X_train.shape[0])
print("📌 Testing Samples:", X_test.shape[0])
