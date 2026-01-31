import pickle
import re

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load trained model
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Simple preprocessing (same logic as training)
def preprocess_input(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z_\s]", "", text)
    return text

# Take user input
while True:
    user_input = input("\nEnter Hinglish + Emoji text (or type 'exit'): ")
    if user_input.lower() == "exit":
        break

    clean_text = preprocess_input(user_input)
    vector = vectorizer.transform([clean_text])
    prediction = model.predict(vector)[0]

    print("👉 Predicted Sentiment:", prediction.upper())
