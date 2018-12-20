# Sentiment analysis of tweets.
In this project, the goal is to determine the sentiment of tweets. Specifically we will try to predict whether a tweet contained a negative or a positive emoji. The submission used on crowdai was made with a convolutional neural network.

## CrowdAI submission.
Username: anonym Grevling

Submission ID: 24231 

## Prerequisites

For this project we use:

```
	Python 3.6.5
        numpy 1.14.3
        csv 1.0
	gensim 3.4.0
	sklearn 0.20.1
	keras 2.2.4
	random  
	tensorflow 1.12.0
```

To install a package simple input
~~~~
	pip install "name of package"
~~~~

## Files included

1. readme.md
2. run.py
3. cnn.py
4. implementations.py

## Instructions to set up
All the text files should be place in the directory "data". 
The training data and test data should be the files: 

1. train_pos_full.txt
2. train_neg_full.txt
3. test_data.txt

## Instructions to replicate test score
All the parameters are set in
1. execute "python3 run.py" in a terminal
