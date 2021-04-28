from functions import *     #imports our functions


#preprocess_data()                                                          # Fixes all our data. 
#combineAllCSVs()                                                           # Merges all CSV files into one. Columns: ['Original Index','TimeSec', 'SleepLVL', 'HR']


# X = sleep
# Y = HR
# N = number of states the hidden variable (sleep) can be in at any time t. (0-5)
#os.chdir("/workspace/BioRobProject")
X = get_data('Sleep_HR_14.csv')                                          # Returns X and Y. X holding sleep states and Y holding HR data
#X = get_dataALL('combined_csv.csv')   
#display(X, Y)


model = hmm.GaussianHMM(n_components=6, n_iter = 100, covariance_type="full")
#n_components = 2                                                           # number of states in the model(awake, asleep)
model.fit(X)
#model.startprob_ = np.array([0.8, 0.1, 0.025, 0.025, 0.025, 0.025])       # starting probabilities, most likely start awake, maybe in state N1 bc of -1 removal

#model.transmat_ = np.array([[0.8, 0.1, 0.025, 0.025, 0.025, 0.025],        # probabilities of transitioning from one state to another, or staying in the same state
#                           [0.1, 0.7, 0.1, 0.034, 0.033, 0.033],
#                            [0.033, 0.1, 0.7, 0.1, 0.034, 0.033],
#                            [0.033, 0.034, 0.1, 0.7, 0.1, 0.033],
#                            [0.033, 0.033, 0.034, 0.1, 0.7, 0.1],
#                            [0.025, 0.025, 0.025, 0.025, 0.1, 0.8]])

#model.means_ = np.array([[ 3, 59.7419355 ],                                 # the mean parameters for each state
#                        [ 1, 58.3814433 ],
#                        [0, 61.46195652],
#                        [2, 57.96015403],
#                        [5, 62.67856466],
#                        [4, 57.19449757]])

#model.covars_ ="full"

#model.score(X)
#display(model.score(X))

remodel = hmm.GaussianHMM(n_components=6, covariance_type="full",  n_iter = 100)
remodel.fit(X)
#display(X)
remodel.monitor_
predictionOutput = remodel.predict(X)
display(predictionOutput)

extra_val = predictionOutput[-1]
extra_val_index = 1 + len(predictionOutput)
#display(extra_val)
#display(len(predictionOutput))
#display(extra_val_index)
#predictionOutput[extra_val_index] = predictionOutput[extra_val]
#predictionOutput.loc[len(predictionOutput.index)] = [extra_val]
display(type(predictionOutput))
display(len(predictionOutput))
predictionOutput = np.append(predictionOutput, 0)
display(len(predictionOutput))


np.savetxt("mainPrediction.csv", predictionOutput, delimiter=",")                      # saves as a .csv 


print('Score: ', np.exp(remodel.score(X)))
print('\n')
print('Means: ','\n', remodel.means_)
print('\n')
print('transmat: ','\n', remodel.transmat_)
print('\n')
print('Covariance: ','\n', remodel.covars_)



scores_train = []
scores_test = []
#best_svc = svm.SVC(kernel='poly')
#best_svc.fit(X, y)
X = get_data('Sleep_HR_14.csv')
y = get_data('Sleep_HR_14.csv')
#X = get_dataALL('Sleep_HR_14.csv')
#y = get_dataALL('Sleep_HR_14.csv')
X_test = X
y_test = predictionOutput
X_train = X
y_train = y

display("Length X_test: " + len(X_test))



#tempTest = accuracy_score(X_test, y_test)
#tempTrain = accuracy_score(X_train, y_train)
#scores_test.append(tempTest)
#scores_train.append(tempTrain)

accuracy_score(y_train, y_test)
print(accuracy_score)
print('SVM 10 fold: ')
print('Train Results: ', scores_train)
print('Test Results: ', scores_test)
print('\nMean Train: ', np.mean(scores_train), '\nMean Test: ', np.mean(scores_test))