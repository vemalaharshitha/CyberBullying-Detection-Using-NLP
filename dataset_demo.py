import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load datasets
reddit = pd.read_csv("data/reddit_comments.csv")
twitter = pd.read_csv("data/cyberbullying_tweets.csv")
submission = pd.read_csv("data/sample_submission.csv")

# For demo: train on twitter dataset
twitter["cleaned"] = twitter["tweet"].apply(clean_text)
twitter["label"] = twitter["cyberbullying_type"].apply(lambda x: 0 if x=="not_cyberbullying" else 1)

X = twitter["cleaned"]
y = twitter["label"]

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=200)
model.fit(X_vec, y)

# Apply model to Reddit dataset
reddit["cleaned"] = reddit["comment"].apply(clean_text)
reddit["pred"] = model.predict(vectorizer.transform(reddit["cleaned"]))

print("Reddit Results:")
print(reddit.head(10))
reddit.to_csv("data/reddit_results.csv", index=False)

print("\nTwitter Results:")
print(twitter.head(10))

print("\n✅ Predictions saved to reddit_results.csv")
