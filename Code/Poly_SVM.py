from functions import *     #imports our functions

# Get Data
X = get_data('combined_csv.csv')
y = get_data('combined_csv.csv')

#X = get_data('Sleep_HR_0.csv')                #if you swap lines 4 and 5 with these two itll only take about 3 minutes to run (VS 2.5 hours)
#y = get_data('Sleep_HR_0.csv')


# Format Data (X = HR data, y = SleepLVL data)
del X['SleepLVL'] 
del y['HR'] 
#display(X)
#display(y)

# Set-Up Model
print("Training SVM...(takes 2.5 hours)")

scores_train = []
scores_test = []
best_svc = svm.SVC(kernel='poly')

best_svc.fit(X, y.values.ravel())

print("Predicting Values...")

X_train = X
y_train = y
tempTrain = accuracy_score(best_svc.predict(X_train), y_train)
scores_train.append(tempTrain)


#get test data
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

# Converts Pandas df to Numpy Array
y_testnp = np.asarray(y_test)
prednp = np.asarray(pred)

# Display Accuracy
display(Accuracy_of_actual(y_testnp, prednp))
display(Accuracy_of_awake(y_testnp, prednp))

# Once we train a model, we will want to save it
best_svc.save('Saved_Models/all_Poly_SVM')

# Did not save, but does run
# Takes 2 HOURS and 25 MINUTES to run through the combined_csv data.