from __future__ import print_function
import numpy as np

# Gensim will be used to create the word embedding
import gensim
from gensim.models import Word2Vec, Doc2Vec, FastText, KeyedVectors
import gensim.parsing.preprocessing as prep

import csv
import random

# Simple models, such as logist regression is imported
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
    # Use with to ensure the file closes after
    with open(str(fileName), "r", encoding="utf8") as sample:
        s = sample.readlines()
    return s



def create_X(list_of_tweets, model_tot, features):
    
    # This function creates a design matrix from the average of each word-vector 
    # in a tweet created in word2vec.
    
    X = np.zeros((len(list_of_tweets), features))
    for indeks, tweet in enumerate(list_of_tweets):
        for word in tweet:
            try:
                X[indeks, :] = X[indeks, :] + model_tot.wv[str(word)] #adds word vector from tweet
            except:
                pass # if the word is not present in the vocabulary, pass.
            
        N = len(tweet)
        if N > 0:
            X[indeks] = X[indeks] / N # take the average for each tweet.
    return X


def createWordEmbedding(list_of_tweets, features, epoc, sg=0):
    # creates a word embedding using word2vec.
    model = Word2Vec(
        list_of_tweets, size=features, window=5, min_count=1, workers=4, sg=sg)
    
    #trains the word embedding:
    model.train(
        list_of_tweets, total_examples=len(list_of_tweets), epochs=epoc)
    
    return model


def save_csv(fileName, test_y):
    # saves the result for submission
    
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
    # trains model and returns the prediction for test file.
    
    met = method.fit(x, y)
    test_y = met.predict(x_test)
    return test_y


def shuffle_tweets(listTweets, y):
    
    # shuffles both the list of tweets and the sentiment
    SEED = 42
    c = list(zip(listTweets, y)) #zip is used to ensure they are shuffled in unison
    random.seed(SEED)
    random.shuffle(c)
    listTweets, y = zip(*c) # unzip
    return listTweets, y


def processTrainingData(list_of_tweets):
    list_of_tweets = [prep.strip_short(line) for line in list_of_tweets] # removes short words with less than 3 characters
    list_of_tweets = prep.preprocess_documents(list_of_tweets) # removes punctuation, numbers, whitespace, 
    
    return list_of_tweets # each tweet is returned as a list


def easyProcess(list_of_tweets):
    list_of_tweets = [prep.strip_short(line) for line in list_of_tweets]
    return list_of_tweets # each tweet is an element


def process_set(negFileName, posFileName):
    
    # Opens two files, then processes each file, shuffles them and return a list of tweets.
    # Each tweet is a list (list of lists)
    positive_tweets_full = open_file(str(posFileName))
    negative_tweets_full = open_file(str(negFileName))

    positive_tweets_full = processTrainingData(positive_tweets_full)
    negative_tweets_full = processTrainingData(negative_tweets_full)
    y_full = [1] * len(positive_tweets_full) + [0] * len(negative_tweets_full)

    all_tweets_full = positive_tweets_full + negative_tweets_full  #list of tweets
    all_tweets_full, y_full = shuffle_tweets(all_tweets_full, y_full)
    return all_tweets_full, y_full


def easyProcess_set(negFileName, posFileName):
    
    # Opens two files, then processes each file, shuffles them and return a list of tweets.
    # Each tweet is an element
    
    positive = open_file(str(posFileName))
    negative = open_file(str(negFileName))

    list_all_tweets = positive + negative
    y_full = [1] * len(positive) + [0] * len(negative)
    list_all_tweets, y_full = shuffle_tweets(list_all_tweets, y_full)
    list_all_tweets = easyProcess(list_all_tweets)
    return list_all_tweets, y_full


def LR_with_w2v(model_tot, testData, negFile, posFile):
    # prediction using word embedding from word2vec and LR as classifier.
    
    testd_tweets = open_file(str(testData))
    all_tweets, y = process_set(str(posFile), str(negFile))
    
    features = len(model_tot.wv[str(all_tweets[0][0])]) #number of features in word embedding
    
    X_old = create_X(all_tweets, model_tot, features)
    X_old = preprocessing.scale(X_old) # stanrdazies each column
    X = np.ones((X_old.shape[0], X_old.shape[1] + 1)) # Creates matrix of ones
    X[:, 1:] = X_old #adds the design matrix to a column of ones.

    cv_results_lr = cross_validate(
        LogisticRegression(solver='lbfgs'),
        X,
        y,
        return_train_score=False,
        cv=5) # test using cross validation
    print('Score using logistic regression and word2vec', np.mean(np.asarray(cv_results_lr['test_score']))) #print average result from cross validation


def tfidf(testData, negFile, posFile):
    
    testd_tweets = open_file(str(testData))
    list_all_tweets, y_full = easyProcess_set(str(posFile),
                                              str(negFile))
    
    tfidfMatrix = TfidfVectorizer(max_features=100000, ngram_range=(1, 3))
    tfidfMatrix.fit(list_all_tweets) # turn tweets into a matrix of TF-IDF features
    
    x_train_tfidf = tfidfMatrix.transform(list_all_tweets) #Transform training tweets to document-term matrix.
    
    cv_results_lr = cross_validate(
        LogisticRegression(solver='lbfgs', max_iter=1000),
        x_train_tfidf,
        y_full,
        return_train_score=False,
        cv=5) # test using cross validation
 
    print('Score using TF-IDF:', np.mean(np.asarray(cv_results_lr['test_score']))) #print average result from cross validation
    
    '''
    # To produce a test file for submission run the following code:
    x_test_tfidf = tfidfMatrix.transform(testd_tweets) #Transform test tweets to document-term matrix.
    testy = train(LogisticRegression(solver='lbfgs', max_iter=1000) , x_train_tfidf, y_full, x_test_tfidf)
    save_csv('test_tfid.csv', testy)
    '''