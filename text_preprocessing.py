


import pandas as pd
import re
import emoji
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("reddit_hinglish_sentiment.csv")

# English stopwords
english_stopwords = set(stopwords.words('english'))

# Hinglish stopwords (custom)
hinglish_stopwords = {
    "hai", "ka", "ki", "ke", "ko", "se", "par", "aur",
    "ye", "wo", "is", "us", "mein", "abhi", "bahut"
}

all_stopwords = english_stopwords.union(hinglish_stopwords)

# Emoji sentiment mapping
positive_emojis = {"😊", "😄", "👍", "🎉", "✅"}
negative_emojis = {"😡", "😠", "👎", "❌", "💔"}
neutral_emojis  = {"🤔", "😐", "🤷‍♂️", "🧐", "📊"}

def convert_emojis(text):
    for e in positive_emojis:
        text = text.replace(e, " positive_emoji ")
    for e in negative_emojis:
        text = text.replace(e, " negative_emoji ")
    for e in neutral_emojis:
        text = text.replace(e, " neutral_emoji ")
    return text

def clean_text(text):
    text = text.lower()
    text = convert_emojis(text)
    text = re.sub(r"http\S+", "", text)   # remove URLs
    text = re.sub(r"[^a-z_\s]", "", text) # keep letters & emoji tokens
    tokens = text.split()
    tokens = [word for word in tokens if word not in all_stopwords]
    return " ".join(tokens)

# Apply preprocessing
df["clean_comment"] = df["comment"].apply(clean_text)

# Save cleaned dataset
df.to_csv("cleaned_reddit_sentiment.csv", index=False, encoding="utf-8")

# Show sample
print(df[["comment", "clean_comment"]].head(10))
print("\n✅ Preprocessing completed. Cleaned file saved.")
