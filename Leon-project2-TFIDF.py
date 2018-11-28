import numpy as np
import csv
import gensim
import re
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup


def tweet_cleaner(text):
	stripped = re.sub("<user>", ' ', text)
	stripped = re.sub("<url>", ' ', stripped)
	#letters_only = re.sub("[^a-zA-Z]", " ", stripped)
	without_tag = re.sub('@[A-Za-z0-9]+', " ", stripped)
	without_html = re.sub('https?://[A-Za-z0-9./]+', " ", without_tag)
	tok = WordPunctTokenizer()
	words = tok.tokenize(without_html)
	return (" ".join(words)).strip()
	#return without_html

def open_file(fileName):
    with open(str(fileName), "r", encoding="utf8") as sample:
        s = sample.readlines() 
    return s

def save_file(data, fileName):
    with open(str(fileName), "w", encoding="utf8") as file:
    	for item in data:
    		file.write(item)
    		file.write("\n")

def processTrainingData(list_of_tweets):
	tempList = []
	for lines in list_of_tweets:
		tempList.append(tweet_cleaner(lines))
	cleaned_data = list(set(tempList)) # remove duplicate lines, should not be done for test-data
	cleaned_data = [gensim.utils.simple_preprocess(line) for line in tempList] # simple preprocessing
	return tempList #cleaned_data

# Data cleaning 
positive_tweets = open_file("./Magnus/train_pos.txt")
negative_tweets = open_file("./Magnus/train_neg.txt")
X_test = open_file("./Magnus/test_data.txt")
positive_tweets_processed = processTrainingData(positive_tweets)
save_file(positive_tweets_processed,"./train_pos.txt")
negative_tweets_processed = processTrainingData(negative_tweets)
save_file(negative_tweets_processed,"./train_neg.txt")
X_test_processed = processTrainingData(X_test)
save_file(X_test_processed,"./test_data.txt")
positive_tweets = open_file("./train_pos.txt")
negative_tweets = open_file("./train_neg.txt")

# This is the training set's Y
y = [1]*len(positive_tweets)+[0]*len(negative_tweets)


# Seperating Id and Data frmo the testing set
field = []
count = 0
test_X = []
test_X_id = []
with open('./test_data.txt', 'r') as f:
	for line in f:
		temp = line.split(",",1)
		test_X.append(temp[1])
		test_X_id.append(temp[0])
		
#Start predicting
tvec = TfidfVectorizer(max_features=100000,ngram_range=(1, 3))
tvec.fit(positive_tweets + negative_tweets)
x_train_tfidf = tvec.transform(positive_tweets + negative_tweets)
x_test_tfidf = tvec.transform(X_test)
lr_with_tfidf = LogisticRegression()
lr_with_tfidf.fit(x_train_tfidf,y)
test_y = lr_with_tfidf.predict(x_test_tfidf)

with open('test_result_TFIDF.csv', 'w') as csvfile:
    tempwriter = csv.writer(csvfile)
    tempwriter.writerow(["Id","Prediction"])
    count = 0
    for row in test_y:
        if row == 0:
            row = -1
        tempwriter.writerow([test_X_id[count],str(row)])
        count = count + 1
