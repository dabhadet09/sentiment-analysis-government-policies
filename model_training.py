import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned dataset
df = pd.read_csv("cleaned_reddit_sentiment.csv")

X = df["clean_comment"]
y = df["sentiment"]

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

X_tfidf = vectorizer.transform(X)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------
# Naive Bayes Model
# ------------------
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_pred)

print("🔹 Naive Bayes Accuracy:", nb_accuracy)
print(classification_report(y_test, nb_pred))

# --------------------------
# Logistic Regression Model
# --------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\n🔹 Logistic Regression Accuracy:", lr_accuracy)
print(classification_report(y_test, lr_pred))

# --------------------------
# Save the better model
# --------------------------
best_model = lr_model if lr_accuracy >= nb_accuracy else nb_model

with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n✅ Best model saved as sentiment_model.pkl")
