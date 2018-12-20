
# coding: utf-8

# # Text classification of tweets:

# In[1]:


from __future__ import print_function

from implementations import *
from cnn import *
import numpy as np
import csv
import random

import gensim
from gensim.models import Word2Vec, Doc2Vec, FastText, KeyedVectors
import gensim.parsing.preprocessing as prep

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_validate
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import datasets, svm
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


# ## Create word embedding based on full set:

# In[2]:


features = 200 # number of features in word embedding
epoch = 10 # number of epochs for word embedding

#all_tweets_full, y_full = process_set('train_neg_full.txt', 'train_pos_full.txt') # corpus for word embedding
#model_tot_200 = createWordEmbedding(all_tweets_full, features, epoch) # creates word embedding 
#model_tot_200.save('model_tot_200.word2vec') # saves word embedding
#model_tot = Word2Vec.load('model_tot_200.word2vec') # loads word embedding


# # Cross validation for word2vec + LR and tfidf + LR:

# In[5]:


# LR_with_w2v(model_tot, 'test_data.txt', 'train_neg.txt', 'train_pos.txt') # LR and word2vec
# tfidf('test_data.txt', 'train_neg_full.txt', 'train_pos_full.txt') # TF-IDF with full set

CNN('model_tot_200.word2vec', 'train_neg.txt', 'train_pos.txt', 512, 'test_data.txt', 1, 20000, features)
