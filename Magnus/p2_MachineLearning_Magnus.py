#!/usr/bin/env python
# coding: utf-8

# # Text classification of tweets:

# In[1]:


import numpy as np
import gensim
from gensim.models import Word2Vec
import csv
import pandas as pd
#import spacy
import nltk
from sklearn import naive_bayes as nb
from sklearn.linear_model import LogisticRegression


# In[2]:


def open_file(fileName):
    with open(str(fileName), "r", encoding="utf8") as sample:
        s = sample.readlines() 
    return s

def create_X(list_of_tweets, w2v, features):
    
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
    list_of_tweets = list(set(list_of_tweets)) # remove duplicates
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


save_csv('test_resultLR.csv', test_y)
save_csv('test_resultNB.csv', test_y_nb)


# In[7]:


print('Done')

