from sentiment_data import read_sentiment_examples, SentimentExample

text = read_sentiment_examples('data/train.txt')

print(text[0].words)