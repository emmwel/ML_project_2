#!/usr/bin/env python
# coding: utf-8

# # Text classification of tweets:

# In[1]:


import numpy as np
import gensim
from gensim.models import Word2Vec
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


features = 350
epoc = 50
spos = open_file("train_pos.txt")
sneg = open_file("train_neg.txt")

spos = processTrainingData(spos)
sneg = processTrainingData(sneg)

y = [1]*len(spos)+[0]*len(sneg)

stotal = spos+sneg

model_tot = createWordEmbedding(stotal, features, epoc)

X = create_X(stotal,model_tot, features) 
X = preprocessing.scale(X)


# In[4]:


testd = open_file("test_data.txt")
testd = [gensim.utils.simple_preprocess(line) for line in testd]

model_test = createWordEmbedding(testd, features, epoc)

X_test = create_X(testd,model_test, features)


# In[5]:


# Build logistic regression classifiers to identify the polarity of words
test_y = train(LogisticRegression(), X, y, X_test)

# Build naive bayes classifiers to identify the polarity of words
test_y_nb = train(nb.GaussianNB(), X, y, X_test) # this one isn't working


# In[6]:


cv_results_lr = cross_validate(LogisticRegression(), X, y, return_train_score=False)
print(cv_results_lr['test_score'])


# In[7]:


clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)

cv_results_lr = cross_validate(clf, X, y, return_train_score=False)
print(cv_results_lr['test_score'])


# In[8]:


save_csv('test_resultLR.csv', test_y)
save_csv('test_resultNB.csv', test_y_nb)


# In[9]:


print('Done')


# In[ ]:



'''
#function from scikit:
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()
'''


# In[ ]:




