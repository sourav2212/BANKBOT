from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
ESCALATION_THRESHOLD = -0.6
def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    
    if compound <= ESCALATION_THRESHOLD:
        label = "FRUSTRATED"
    elif compound < 0.05:
        label = "NEUTRAL"
    else:
        label = "POSITIVE"
    
    return {
        "compound": round(compound, 3),
        "label": label,
        "should_escalate": compound <= ESCALATION_THRESHOLD
    }

if __name__ == "__main__":
    test_messages = [
        "I am so angry, I've been waiting for 3 days and nobody helps me!",
        "What is the interest rate for a home loan?",
        "Thank you, the service was excellent!"
    ]
    for msg in test_messages:
        result = analyze_sentiment(msg)
        print("\nMessage: " + msg)
        print("Result: " + str(result))