from sentiment_data import read_sentiment_examples, SentimentExample
from models import UnigramFeatureExtractor
from utils import Indexer
from collections import Counter
import numpy as np

texts = read_sentiment_examples('data/train.txt')
feature_extractor = UnigramFeatureExtractor(indexer=Indexer())

frequency_dicts = [feature_extractor.extract_features(sentence.words, add_to_indexer=True) for sentence in texts]

combined_counter = Counter()

for x in frequency_dicts:
    combined_counter.update(x)

combined_dict = dict(combined_counter)

# Try scoring on example sentence

ex = "Chateauneuf du Pape is better than Napa Cabs"
ex1 = dict(Counter(ex.split()))


def get_feature_vector(sentence, dictionary):
    vector = []
    word_dict = dict(Counter(sentence.split()))
    for i, (word, count) in enumerate(dictionary.items()):
        if word in word_dict.keys():
            vector.append(word_dict[word])
        else:
            vector.append(0)

    return vector


feature_vector = get_feature_vector(ex, combined_dict)
print(len(feature_vector))
print(len(combined_dict))
print(sum(feature_vector))
# print(feature_vector)
