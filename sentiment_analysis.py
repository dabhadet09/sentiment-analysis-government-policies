import pandas as pd
import random

positive_templates = [
    "Government ki {} policy kaafi achhi hai {}",
    "Public ke liye {} scheme bahut helpful hai {}",
    "{} policy se log kaafi khush hain {}",
    "Ye {} reform future ke liye beneficial hoga {}",
    "Government ka {} decision sahi direction mein hai {}"
]

negative_templates = [
    "Government ki {} policy bilkul bekar hai {}",
    "{} decision se public kaafi naraaz hai {}",
    "Is {} policy ne logon ko disappoint kiya {}",
    "Government ka {} decision galat lag raha hai {}",
    "{} reform se middle class ko nuksaan hua {}"
]

neutral_templates = [
    "{} policy par log mixed reactions de rahe hain {}",
    "Is {} decision ko samajhne mein time lagega {}",
    "{} reform ka impact abhi clear nahi hai {}",
    "Government ki {} policy par discussion chal rahi hai {}",
    "Log {} policy ko lekar confused hain {}"
]

policies = [
    "education", "budget", "tax", "healthcare", "employment",
    "digital india", "agriculture", "startup", "infrastructure", "energy"
]

positive_emojis = ["😊", "👍", "🎉", "😄", "✅"]
negative_emojis = ["😡", "👎", "😠", "❌", "💔"]
neutral_emojis  = ["🤔", "😐", "🤷‍♂️", "🧐", "📊"]

data = []

def generate_rows(templates, emojis, sentiment, count):
    for _ in range(count):
        template = random.choice(templates)
        policy = random.choice(policies)
        emoji = random.choice(emojis)
        comment = template.format(policy, emoji)
        data.append([comment, sentiment])

generate_rows(positive_templates, positive_emojis, "positive", 650)
generate_rows(negative_templates, negative_emojis, "negative", 650)
generate_rows(neutral_templates, neutral_emojis, "neutral", 650)

df = pd.DataFrame(data, columns=["comment", "sentiment"])
df.to_csv("reddit_hinglish_sentiment.csv", index=False, encoding="utf-8")

print("✅ Dataset generated: reddit_hinglish_sentiment.csv")
print(df.sample(5))
