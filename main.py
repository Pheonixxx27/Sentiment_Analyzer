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

# Custom function to mark negations in the text
def mark_negation(text):
    negation_words = set(["not", "no", "never", "n't"])
    negation = False
    result = []
    for word in text:
        if word in negation_words:
            negation = not negation
        if negation:
            word = f"not_{word}"
        result.append(word)
    return result

# Function to analyze sentiment for a given input word
def analyze_sentiment(input_word):
    # Filter tweets containing the input word
    relevant_tweets = [tweet for tweet in tweets if input_word.lower() in tweet.lower()]
    
    if not relevant_tweets:
        # Return a default sentiment score (e.g., 0 for neutral sentiment)
        return 0.0
    
    # Prepare the tweets for sentiment analysis using VADER sentiment analyzer
    relevant_tweets_with_negation = [mark_negation(tweet.split()) for tweet in relevant_tweets]
    
    # Calculate the overall sentiment score for the input word using VADER
    sentiment_scores = [sid.polarity_scores(" ".join(tweet))["compound"] for tweet in relevant_tweets_with_negation]
    overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    
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
