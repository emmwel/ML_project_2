from __future__ import print_function
import numpy as np
from implementations import *
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

def CNN(w2vName, negFile, posFile, batch_size, testData, epochs, word_count, features):

    list_all_tweets, y_full = easyProcess_set(str(negFile), str(posFile))
    SEED = 42
    x_train, x_validation, y_train, y_validation = train_test_split(
        list_all_tweets, y_full, test_size=.1, random_state=SEED)
    model_tot_k = KeyedVectors.load(str(w2vName))

    embed_index = {}
    for w in model_tot_k.wv.vocab.keys():
        embed_index[w] = model_tot_k.wv[w] # get index of words

    word_count = word_count
    tokenizer = Tokenizer(word_count)
    tokenizer.fit_on_texts(x_train) # represent each tweet as sequence
    sequences = tokenizer.texts_to_sequences(x_train) #each tweet represented with indexes from word embedding

    #Finds longest tweet:
    length = []
    for x in x_train:
        length.append(len(x.split()))
    length = max(length)+2

    x_train_seq = pad_sequences(sequences, maxlen=length) # creates matrix of dimensions (word_count x length)

    sequences_val = tokenizer.texts_to_sequences(x_validation)
    x_val_seq = pad_sequences(sequences_val, maxlen=length)

    # use the most common words, number of words = word_count
    embed_matrix = np.zeros((word_count, features))
    for word, i in tokenizer.word_index.items():
        if i >= word_count:
            continue
        embed_vector = embed_index.get(word)
        if embed_vector is not None:
            embed_matrix[i] = embed_vector

    hFeatures = int(features * 0.5) # value used in filters in our model

    tweet_input = Input(shape=(length,), dtype='int32')
    
    # Setting up cnn, adding layers 
    embedding_cnn = Embedding(word_count, features, weights=[embed_matrix], input_length=length, trainable=True)(tweet_input)
    k2 = Conv1D(filters=hFeatures, kernel_size=2, padding='valid', activation='relu', strides=1)(embedding_cnn)
    k2 = GlobalMaxPooling1D()(k2)
    k4 = Conv1D(filters=hFeatures, kernel_size=4, padding='valid', activation='relu', strides=1)(embedding_cnn)
    k4 = GlobalMaxPooling1D()(k4)
    k6 = Conv1D(filters=hFeatures, kernel_size=6, padding='valid', activation='relu', strides=1)(embedding_cnn)
    k6 = GlobalMaxPooling1D()(k6)
    merged = concatenate([k2, k4, k6], axis=1)
    merged = Dense(256, activation='relu')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(1)(merged)
    output = Activation('sigmoid')(merged)
    model = Model(inputs=[tweet_input], outputs=[output])
    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])


    filepath = "cnn_result{epoch:02d}_{val_acc:.2f}.hdf5" # saves best  result 
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    model.fit(
        x_train_seq,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val_seq, y_validation),
        callbacks=[checkpoint])

    testd_tweets = open_file(str(testData))
    testd_tweets = easyProcess(testd_tweets)

    sequences_test = tokenizer.texts_to_sequences(testd_tweets)
    x_test_seq = pad_sequences(sequences_test, maxlen=length)
    y_cnn = model.predict(x=x_test_seq) # prediction based on test-file

    y_cnn_rounded = (np.around(y_cnn)).flatten() 
    y_cnn_rounded[y_cnn_rounded == 0] = -1 # turns all zeros into -1
    save_csv('test_resultCNN.csv', y_cnn_rounded)
    return ycnn_rounded
