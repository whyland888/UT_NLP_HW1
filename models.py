# models.py

from sentiment_data import *
from utils import *
import numpy as np
from collections import Counter


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")

    def remove_punctuation(self, word):
        punctuation =  '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        return word.translate(str.maketrans("", "", punctuation))


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        sentence = [self.remove_punctuation(word) for word in sentence]
        for word in sentence:
            self.indexer.add_and_get_index(word, add=add_to_indexer)
        return Counter(sentence)

    def convert_to_vector(self, sentence, indexer):
        feat_vector = np.zeros(len(indexer))
        sent_counter = Counter(sentence)
        words = list(sent_counter.keys())
        counts = list(sent_counter.values())
        indices = [indexer.index_of(word) for word in words]
        for i in range(len(indices)):
            index = indices[i]
            feat_vector[index] = counts[i]
        return feat_vector

    def get_indexer(self):
        return self.indexer


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        sentence = [self.remove_punctuation(word) for word in sentence]
        bigrams = [f"{w1} {w2}" for w1, w2 in zip(sentence, sentence[1:])]
        for bigram in bigrams:
            self.indexer.add_and_get_index(bigram, add=add_to_indexer)
        return Counter(bigrams)

    def convert_to_vector(self, sentence, indexer):
        feat_vector = np.zeros(len(indexer))
        bigrams = [f"{w1} {w2}" for w1, w2 in zip(sentence, sentence[1:])]
        bigram_counter = Counter(bigrams)
        words = list(bigram_counter.keys())
        counts = list(bigram_counter.values())
        indices = [indexer.index_of(word) for word in words]
        for i in range(len(indices)):
            index = indices[i]
            feat_vector[index] = counts[i]
        return feat_vector

    def get_indexer(self):
        return self.indexer


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        sentence = [self.remove_punctuation(word) for word in sentence]



class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def __init__(self, feature_extractor, indexer, data):
        self.feature_extractor = feature_extractor
        self.indexer = indexer
        self.frequency_dicts = [feature_extractor.extract_features(sentence.words, add_to_indexer=True) for sentence
                                in data]

        # Add up individual dictionaries to get vocab
        combined_counter = Counter()
        for frequency_dict in self.frequency_dicts:
            combined_counter.update(frequency_dict)

        self.vocab = dict(combined_counter)

        # Initialize weights and bias
        self.weights = np.zeros(len(self.vocab))
        self.bias = 0

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        x = self.feature_extractor.convert_to_vector(sentence, self.indexer)
        z = np.dot(self.weights, x) + self.bias
        prediction = 1 if z > 0 else 0
        return prediction

    def train(self, data, num_epochs=20, learning_rate=1):
        for epoch in range(num_epochs):
            for sentence in data:
                x = self.feature_extractor.convert_to_vector(sentence.words, self.indexer)
                prediction = self.predict(sentence.words)
                update = learning_rate * (sentence.label - prediction)
                self.weights += (update * x)
                self.bias += update
            np.random.shuffle(data)


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feature_extractor, indexer, data):
        super().__init__(feature_extractor, indexer, data)
        X = np.array([self.feature_extractor.convert_to_vector(sentence.words, self.indexer) for sentence in data])
        self.l1 = np.linalg.norm(X)
        X = X/self.l1
        self.mean = np.mean(X)
        self.std_dev = np.std(X)
        self.X = (X - self.mean) / self.std_dev
        self.labels = np.array([sentence.label for sentence in data])

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))

    def binary_cross_entropy(self, y, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -1/len(y) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def predict(self, sentence: List[str]) -> int:
        # Convert to array, L1 penalty, normalize
        x = np.array(self.feature_extractor.convert_to_vector(sentence, self.indexer))
        x = x /self.l1
        x = (x - self.mean) / self.std_dev

        # Predict
        linear_preds = np.dot(x, self.weights) + self.bias
        z = self.sigmoid(linear_preds)
        prediction = 1 if z > .5 else 0
        return prediction

    def train(self, iterations=1000, learning_rate=.006):
        loss_history = []
        n_samples = len(self.labels)

        for i in range(iterations):
            linear_preds = np.dot(self.X, self.weights) + self.bias
            predictions = self.sigmoid(linear_preds)

            # Get gradients
            dw = (1/n_samples) * np.dot(self.X.T, predictions - self.labels)
            db = (1/n_samples) * np.sum(predictions - self.labels)

            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            # Calculate loss
            loss = self.binary_cross_entropy(self.labels, predictions)
            loss_history.append(loss)

        print(loss_history)


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    indexer = feat_extractor.get_indexer()
    perceptron = PerceptronClassifier(feature_extractor=feat_extractor, indexer=indexer, data=train_exs)
    perceptron.train(data=train_exs)
    return perceptron


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    indexer = feat_extractor.get_indexer()
    logreg = LogisticRegressionClassifier(feature_extractor=feat_extractor, indexer=indexer, data=train_exs)
    logreg.train()
    return logreg


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model