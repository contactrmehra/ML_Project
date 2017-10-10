import os
import json
from csv import DictReader, DictWriter
import nltk as nl
import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


SEED = 5


'''
The ItemSelector class was created by Matt Terry to help with using
Feature Unions on Heterogeneous Data Sources

All credit goes to Matt Terry for the ItemSelector class below

For more information:
http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
'''
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


"""
This is an example of a custom feature transformer. The constructor is used
to store the state (e.g like if you need to store certain words/vocab), the
fit method is used to update the state based on the training data, and the
transform method is used to transform the data into the new feature(s). In
this example, we simply use the length of the movie review as a feature. This
requires no state, so the constructor and fit method do nothing.
"""
class TextLengthTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = len(ex)
            i += 1

        return features


# Adjective Feature
class AdjectivesNumber(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.countVectorizer = CountVectorizer(min_df=.15,max_df=.80)

    def fit(self, examples):
        adjective_list=self.get_adjectives(examples)
        self.countVectorizer.fit(adjective_list)
        return self

    def get_adjectives(self, examples):
        adjectives = []
        for ex in examples:
            tokens = nl.word_tokenize(ex)
            pos = nl.pos_tag(tokens)
            adj = ''
            for word in pos:
                if word[1] in ["JJ", "JJR","JJS"]:
                    adj += word[0] + ' '
            adjectives.append(adj)
        return adjectives

    def transform(self, examples):
        adjective_list = self.get_adjectives(examples)

        return self.countVectorizer.transform(adjective_list)


# TODO: Add custom feature transformers for the movie review data


class Featurizer:
    def __init__(self):
        # To add new features, just add a new pipeline to the feature union
        # The ItemSelector is used to select certain pieces of the input data
        # In this case, we are selecting the plaintext of the input data

        # TODO: Add any new feature transformers or other features to the FeatureUnion
        self.all_features = FeatureUnion([
            ('text_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('text_length', TextLengthTransformer())
            ])),
            ('text_stats1', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('adj_length', AdjectivesNumber())
            ])),
            ('text_stats2', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('adj_ngrams', CountVectorizer(ngram_range=(1,3)))
            ])),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # ('ngrams', Pipeline([
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    #     ('selector', ItemSelector(key='text')),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    #     ('ngramsVectorize', CountVectorizer(ngram_range=(1,3)))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    # ]))
        ])

    def train_feature(self, examples):
        return self.all_features.fit_transform(examples)

    def test_feature(self, examples):
        return self.all_features.transform(examples)

if __name__ == "__main__":

    # Read in data

    dataset_x = []
    dataset_y = []

    with open('../data/movie_review_data.json') as f:
        data = json.load(f)
        for d in data['data']:
            dataset_x.append(d['text'])
            dataset_y.append(d['label'])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=SEED)

    feat = Featurizer()

    labels = []
    for l in y_train:
        if not l in labels:
            labels.append(l)

    print("Label set: %s\n" % str(labels))

    # Here we collect the train features
    # The inner dictionary contains certain pieces of the input data that we
    # would like to be able to select with the ItemSelector
    # The text key refers to the plaintext
    feat_train = feat.train_feature({
        'text': [t for t in X_train]
    })
    # Here we collect the test features
    feat_test = feat.test_feature({
        'text': [t for t in X_test]
    })

    #print(feat_train)
    #print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', alpha=0.5, max_iter=15000, shuffle=True, verbose=2)


    lr.fit(feat_train, y_train)
    y_pred = lr.predict(feat_train)
    accuracy = accuracy_score(y_pred, y_train)
    print("Accuracy on training set =", accuracy)
    y_pred = lr.predict(feat_test)
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy on test set =", accuracy)

    # Extra credit solution start______________________________________________________________________________

    # gridSearch = GridSearchCV(lr,param_grid={'alpha':[0.001,0.01,0.1,1,10]},cv=3)
    # gridSearch.fit(feat_train, y_train)
    # predictedTrain = gridSearch.predict(feat_train)
    # print("Best Parameter:"+gridSearch.best_params_)
    # print("Best Score:"+gridSearch.best_score_)
    # accuracy=accuracy_score(predictedTrain,y_train)
    # print("Accuracy on training set =", accuracy)
    # predictedTest = gridSearch.predict(feat_test)
    # accuracy = accuracy_score(predictedTest, y_test)
    # print("Accuracy on test set =", accuracy)

    # Extra credit solution ends__________________________________________________________________________________

