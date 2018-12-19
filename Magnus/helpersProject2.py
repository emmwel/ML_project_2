'''
Comments about the code:

the reason for having two functions for processing is that for word2vec we want each tweet to be a list, 
while for CNN we want each tweet to be an element in a list

We load both the small and the large training set
We want to use the full set of tweets for accuracy, 
but when creating the feature-matrix for LR this will use too much memory
'''

from __future__ import print_function
import numpy as np
import gensim
from gensim.models import Word2Vec, Doc2Vec, FastText, KeyedVectors
import gensim.parsing.preprocessing as prep
import csv
import random
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import linear_model

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_validate
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import datasets, svm
from scipy.stats import randint as sp_randint
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, concatenate, Activation, Dropout, Flatten
from keras.models import Model, load_model, Sequential
from keras.layers.embeddings import Embedding


def open_file(fileName):
    with open(str(fileName), "r", encoding="utf8") as sample:
        s = sample.readlines()
    return s


def create_X(list_of_tweets, w2v, features):

    # this function needs some love!

    X = np.zeros((len(list_of_tweets), features))

    for indeks, tweet in enumerate(list_of_tweets):
        for word in tweet:
            try:
                X[indeks, :] = X[indeks, :] + model_tot.wv[str(word)]
            except:
                pass
        N = len(tweet)
        if N > 0:
            X[indeks] = X[indeks] / N
    return X


def createWordEmbedding(list_of_tweets, features, epoc, sg=0):
    model = Word2Vec(
        list_of_tweets, size=features, window=5, min_count=1, workers=4, sg=sg)
    print('word embedding created, start training:')
    model.train(
        list_of_tweets, total_examples=len(list_of_tweets), epochs=epoc)
    return model


def createSentEmbedding(list_of_tweets, features, epoc):
    model = Doc2Vec(list_of_tweets, size=features, min_count=1)
    model.train(
        list_of_tweets, total_examples=len(list_of_tweets), epochs=epoc)
    return model


def save_csv(fileName, test_y):
    ids = np.arange(len(test_y))
    with open(fileName, 'w') as csvfile:
        tempwriter = csv.writer(csvfile)
        tempwriter.writerow(["Id", "Prediction"])
        count = 0
        for row in test_y:
            if row == 0:
                row = -1
            tempwriter.writerow([(ids[count]) + 1, str(row)])
            count = count + 1


def train(method, x, y, x_test):
    met = method.fit(x, y)
    test_y = met.predict(x_test)
    return test_y


def shuffle_tweets(listTweets, y):
    SEED = 42
    c = list(zip(listTweets, y))
    random.seed(SEED)
    random.shuffle(c)
    listTweets, y = zip(*c)
    return listTweets, y


def processTrainingData(list_of_tweets):
    list_of_tweets = [prep.strip_short(line) for line in list_of_tweets]
    list_of_tweets = prep.preprocess_documents(list_of_tweets)
    return list_of_tweets


def easyProcess(list_of_tweets):
    list_of_tweets = [prep.strip_short(line) for line in list_of_tweets]
    return list_of_tweets


def process_set(negFileName, posFileName):
    positive_tweets_full = open_file(str(posFileName))
    negative_tweets_full = open_file(str(negFileName))

    positive_tweets_full = processTrainingData(positive_tweets_full)
    negative_tweets_full = processTrainingData(negative_tweets_full)
    y_full = [1] * len(positive_tweets_full) + [0] * len(negative_tweets_full)

    all_tweets_full = positive_tweets_full + negative_tweets_full  #list of tweets
    all_tweets_full, y_full = shuffle_tweets(all_tweets_full, y_full)
    return all_tweets_full, y_full


def easyProcess_set(negFileName, posFileName):
    positive = open_file(str(posFileName))
    negative = open_file(str(negFileName))

    list_all_tweets = positive + negative
    y_full = [1] * len(positive) + [0] * len(negative)
    list_all_tweets, y_full = shuffle_tweets(list_all_tweets, y_full)
    list_all_tweets = easyProcess(list_all_tweets)
    return list_all_tweets, y_full