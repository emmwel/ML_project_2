#!/usr/bin/env python
# coding: utf-8

# # Text classification of tweets:

# In[1]:


import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
import csv
#import pandas as pd
#import spacy
#import nltk
from sklearn import naive_bayes as nb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier

from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import datasets, svm
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing 


# In[2]:


def open_file(fileName):
    with open(str(fileName), "r", encoding="utf8") as sample:
        s = sample.readlines() 
    return s

def create_X(list_of_tweets, w2v, features):
    
    # this function needs some love!
    
    X = np.zeros((len(list_of_tweets),features)) 
    
    for indeks, tweet in enumerate(list_of_tweets):
        for word in tweet:
            try:
                X[indeks,:] = X[indeks,:] + model_tot.wv[str(word)]
            except:
                pass
        N = len(tweet)
        if N>0:
            X[indeks] = X[indeks]/N
    return X

def processTrainingData(list_of_tweets):
    list_of_tweets = list(set(list_of_tweets)) # remove duplicate lines, should not be done for test-data
    list_of_tweets = [gensim.utils.simple_preprocess(line) for line in list_of_tweets] # simple preprocessing
    return list_of_tweets

def createWordEmbedding(list_of_tweets, features, epoc):
    model = Word2Vec(list_of_tweets, size=features, window=5, min_count=1, workers=4)
    model.train(list_of_tweets, total_examples=len(list_of_tweets), epochs=epoc)
    return model

def createSentEmbedding(list_of_tweets, features, epoc):
    model = Doc2Vec(list_of_tweets, size=features, min_count=1)
    model.train(list_of_tweets, total_examples=len(list_of_tweets), epochs=epoc)
    return model

def save_csv(fileName, test_y):
    ids = np.arange(len(test_y))  
    with open(fileName, 'w') as csvfile:
        tempwriter = csv.writer(csvfile)
        tempwriter.writerow(["Id","Prediction"])
        count = 0
        for row in test_y:
            if row == 0:
                row = -1
            tempwriter.writerow([(ids[count])+1,str(row)])
            count = count + 1
            
def train(method, x, y, x_test):
    met = method.fit(x,y)
    test_y = met.predict(x_test)
    return test_y


# In[3]:


features = 60
epoch = 20
positive_tweets = open_file("train_pos.txt")
negative_tweets = open_file("train_neg.txt")

positive_tweets = processTrainingData(positive_tweets)
negative_tweets = processTrainingData(negative_tweets)

y = [1]*len(positive_tweets)+[0]*len(negative_tweets)

all_tweets = positive_tweets+negative_tweets #list of tweets

model_tot = createWordEmbedding(all_tweets, features, epoch) #word embedding

X = create_X(all_tweets,model_tot, features) 
print('ferdig med trening-ish')


# In[4]:


testd = open_file("test_data.txt")
testd = [gensim.utils.simple_preprocess(line) for line in testd]

model_test = createWordEmbedding(testd, features, epoch)


# In[5]:


X = preprocessing.scale(X)
X_test = create_X(testd,model_test, features)
X_test = preprocessing.scale(X_test)
# Build logistic regression classifiers to identify the polarity of words
test_y = train(LogisticRegression(), X, y, X_test)

# Build naive bayes classifiers to identify the polarity of words
#test_y_nb = train(nb.GaussianNB(), X, y, X_test) # this one isn't working


# In[6]:


save_csv('test_resultLR.csv', test_y)
#save_csv('test_resultNB.csv', test_y_nb)


# ## Doing some cross validation:

# In[9]:


clf = linear_model.SGDClassifier(max_iter=500, tol=1e-3)

cv_results_clf = cross_validate(clf, X, y, return_train_score=False)
print(cv_results_clf['test_score'])

cv_results_lr = cross_validate(LogisticRegression(), X, y, return_train_score=False)
print(cv_results_lr['test_score'])

#cv_results_nb = cross_validate(nb.GaussianNB(), X, y, return_train_score=False)
#print(cv_results_nb['test_score'])


# In[8]:


print('Done')


# In[ ]:




