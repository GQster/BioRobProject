from functions import *     #imports our functions

#os.chdir('/workspace/BioRobProject/Data/Merged')
X = get_dataone('Sleep_HR_0.csv')
y = get_dataone('Sleep_HR_0.csv')



del X['SleepLVL'] 
del y['HR'] 
display(X)
display(y)


scores_train = []
scores_test = []
best_svc = svm.SVC(kernel='poly')

best_svc.fit(X, y)

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