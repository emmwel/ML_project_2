def LR_with_w2v(all_tweets, model_tot, features, testd_tweets, y):
    X_old = create_X(all_tweets, model_tot, features)
    X_old = preprocessing.scale(X_old)
    print('processed feature matrix for tweets')
    # Add column of ones
    X = np.ones((X_old.shape[0], X_old.shape[1] + 1))
    X[:, 1:] = X_old

    testd = processTrainingData(testd_tweets)

    X_test_old = create_X(testd, model_tot, features)
    X_test_old = preprocessing.scale(X_test_old)
    X_test = np.ones((X_test_old.shape[0], X_test_old.shape[1] + 1))
    X_test[:, 1:] = X_test_old
    print('start training')
    cv_results_lr = cross_validate(
        LogisticRegression(solver='lbfgs'),
        X,
        y,
        return_train_score=False,
        cv=5)
    print(np.mean(np.asarray(cv_results_lr['test_score'])))
    #Build logistic regression classifier
    #test_y_lr = train(LogisticRegression(solver='lbfgs', max_iter = 1000), X, y, X_test)
    #save_csv('test_resultLR.csv', test_y_lr)
    #print('saved')


def tfidf(testd_tweets):
    list_all_tweets, y_full = easyProcess_set('train_neg_full.txt',
                                              'train_pos_full.txt')
    tvec = TfidfVectorizer(max_features=100000, ngram_range=(1, 3))
    print('start fitting')
    tvec.fit(list_all_tweets)
    print('done fitting')
    x_train_tfidf = tvec.transform(list_all_tweets)
    x_test_tfidf = tvec.transform(testd_tweets)
    print('start LR')
    lr_with_tfidf = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr_with_tfidf.fit(x_train_tfidf, y_full)
    print('Predict LR')
    testy = lr_with_tfidf.predict(x_test_tfidf)
    save_csv('test_resulttfid.csv', testy)
    print('saved')
    cv_results_lr = cross_validate(
        LogisticRegression(solver='lbfgs', max_iter=1000),
        x_train_tfidf,
        y_full,
        return_train_score=False,
        cv=5)
    #print(cv_results_lr['test_score'])
    print(np.mean(np.asarray(cv_results_lr['test_score'])))