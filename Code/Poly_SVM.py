from functions import *     #imports our functions



X = get_data('Sleep_HR_0.csv')
y = get_data('Sleep_HR_0.csv')

del X['SleepLVL'] 
del y['HR'] 
#display(X)
#display(y)

print("Training SVM...(takes few mintues)")

scores_train = []
scores_test = []
best_svc = svm.SVC(kernel='poly')

best_svc.fit(X, y.values.ravel())

print("Predicting Values...")

X_train = X
y_train = y
tempTrain = accuracy_score(best_svc.predict(X_train), y_train)
scores_train.append(tempTrain)



X = get_data('Sleep_HR_10.csv')
y = get_data('Sleep_HR_10.csv')
del X['SleepLVL'] 
del y['HR'] 
X_test = X
y_test = y
pred = best_svc.predict(X_test)
tempTest = accuracy_score(pred, y_test)
scores_test.append(tempTest)

print('SVM: ')
print('Train Results: ', scores_train)
print('Test Results: ', scores_test)


y_testnp = np.asarray(y_test)                               # converts pandas to numpy array
prednp = np.asarray(pred)                                   # converts pandas to numpy array

display(Accuracy_of_actual(y_testnp, prednp))
display(Accuracy_of_awake(y_testnp, prednp))
