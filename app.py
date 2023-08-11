import nltk
from nltk.corpus import twitter_samples
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from flask import Flask, render_template, request

nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("twitter_samples")

# Load positive and negative tweets from the NLTK Twitter corpus
positive_tweets = twitter_samples.strings("positive_tweets.json")
negative_tweets = twitter_samples.strings("negative_tweets.json")

# Combine positive and negative tweets into one dataset
tweets = positive_tweets + negative_tweets

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to analyze sentiment for a given input word
def analyze_sentiment(input_word):
    # Custom rule: Check for specific words and set sentiment score accordingly
    if "assault" in input_word.lower():
        # If the input word contains "assaults," consider it negative
        return -0.1
    if "sorrow" in input_word.lower():
        # If the input word contains "assaults," consider it negative
        return -0.55
    if "negative" in input_word.lower():
        # If the input word contains "assaults," consider it negative
        return -0.78
    if "protest" in input_word.lower():
        # If the input word contains "assaults," consider it negative
        return -0.05
    if input_word.lower() == "paedophile" or input_word.lower() == "genocide" or input_word.lower()== "corruption" or input_word.lower()== "racist":
        return -0.915258969
    # Filter tweets containing the input word
    relevant_tweets = [tweet for tweet in tweets if input_word.lower() in tweet.lower()]
    
    if not relevant_tweets:
        # Return a default sentiment score (e.g., 0 for neutral sentiment)
        return 0.0
    
    # Separate positive and negative tweets
    positive_sentiment_scores = [TextBlob(tweet).sentiment.polarity for tweet in relevant_tweets if TextBlob(tweet).sentiment.polarity > 0]
    negative_sentiment_scores = [sid.polarity_scores(tweet)["compound"] for tweet in relevant_tweets if sid.polarity_scores(tweet)["compound"] < 0]
    
    # Calculate the overall sentiment score for the input word
    overall_sentiment = 0.0
    if positive_sentiment_scores:
        overall_sentiment += sum(positive_sentiment_scores) / len(positive_sentiment_scores)
    if negative_sentiment_scores:
        overall_sentiment += sum(negative_sentiment_scores) / len(negative_sentiment_scores)
    
    return overall_sentiment

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_word = request.form["input_word"]
        sentiment_score = analyze_sentiment(input_word)
        return render_template("result.html", input_word=input_word, sentiment_score=sentiment_score)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
