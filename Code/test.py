from functions import *     #imports our functions


X = get_dataone('Sleep_HR_0.csv')
y = get_dataone('Sleep_HR_0.csv')



del X['SleepLVL'] 
del y['HR'] 
display(X)
display(y)


scores_train = []
scores_test = []
best_svc = svm.SVC(kernel='poly')
#cv = KFold(n_splits=10, random_state=42, shuffle=True)
#for train_index, test_index in cv.split(X):
best_svc.fit(X, y)
#	display(train_index)
#    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y[train_index], y[test_index]#
#	X_train = np.nan_to_num(X_train)
#	X_test = np.nan_to_num(X_test)
#	best_svc.fit(X_train, y_train)

	# Now lets do some predicitions
X_test = X
y_test = y
X_train = X
y_train = y
tempTest = accuracy_score(best_svc.predict(X_test), y_test)
tempTrain = accuracy_score(best_svc.predict(X_train), y_train)
scores_test.append(tempTest)
scores_train.append(tempTrain)

print('SVM 10 fold: ')
print('Train Results: ', scores_train)
print('Test Results: ', scores_test)
print('\nMean Train: ', np.mean(scores_train), '\nMean Test: ', np.mean(scores_test))