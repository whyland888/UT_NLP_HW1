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
        for element in word:
            if element in punctuation:
                word = word.replace(element, "")
        return word


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        sentence = [self.remove_punctuation(word) for word in sentence]
        indexer = self.indexer
        for word in sentence:
            indexer.add_and_get_index(word, add=add_to_indexer)
        return Counter(sentence)

    def get_indexer(self):
        return self.indexer


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def __init__(self, feature_dict, indexer):
        self.feature_dict = feature_dict
        self.indexer = indexer

        # Add up individual dictionaries to get vocab
        combined_counter = Counter()
        for x in feature_dict:
            combined_counter.update(x)

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

    def convert_to_vector(self, sentence, indexer):
        feat_vector = np.zeros(len(self.vocab))
        sent_counter = Counter(sentence)
        words = list(sent_counter.keys())
        counts = list(sent_counter.values())
        indices = [indexer.index_of(word) for word in words]
        for i in range(len(indices)):
            index = indices[i]
            feat_vector[index] = counts[i]
        return feat_vector


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
        x = self.convert_to_vector(sentence, self.indexer)
        z = np.dot(self.weights, x) + self.bias
        prediction = 1 if z > 0 else 0
        return prediction

    def train(self, data, num_epochs=20, learning_rate=1):
        for epoch in range(num_epochs):
            for sentence in data:
                x = self.convert_to_vector(sentence.words, self.indexer)
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
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))

    def binary_cross_entropy(self, y, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -1/len(y) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def predict(self, sentence: List[str]) -> int:
        X = np.array(self.convert_to_vector(sentence, self.indexer))
        X = X/np.linalg.norm(X)

        # Normalize
        mean = np.mean(X)
        std_dev = np.std(X)
        X = (X - mean) / std_dev

        linear_preds = np.dot(X, self.weights) + self.bias
        z = self.sigmoid(linear_preds)
        prediction = 1 if z > .5 else 0
        return prediction

    def train(self, data, iterations=4000, learning_rate=.002):
        loss_history = []
        n_samples = len(data)
        X = np.array([self.convert_to_vector(sentence.words, self.indexer) for sentence in data])
        X = X/np.linalg.norm(X)
        labels = np.array([sentence.label for sentence in data])

        # Normalize
        mean = np.mean(X)
        std_dev = np.std(X)
        X = (X - mean) / std_dev

        for i in range(iterations):
            linear_preds = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_preds)

            # Get gradients
            dw = (1/n_samples) * np.dot(X.T, predictions - labels)
            db = (1/n_samples) * np.sum(predictions - labels)

            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            # Increase learning rate
            # if i//10 == 0:
                # learning_rate -= .2
                # print(learning_rate)

            # Calculate loss
            loss = self.binary_cross_entropy(labels, predictions)
            loss_history.append(loss)

        print(loss_history)


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    frequency_dicts = [feat_extractor.extract_features(sentence.words, add_to_indexer=True) for sentence in train_exs]
    indexer = feat_extractor.get_indexer()
    perceptron = PerceptronClassifier(feature_dict=frequency_dicts, indexer=indexer)
    perceptron.train(data=train_exs)
    return perceptron


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    frequency_dicts = [feat_extractor.extract_features(sentence.words, add_to_indexer=True) for sentence in train_exs]
    indexer = feat_extractor.get_indexer()
    logreg = LogisticRegressionClassifier(feature_dict=frequency_dicts, indexer=indexer)
    logreg.train(data=train_exs)
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