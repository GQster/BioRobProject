from functions import *     # imports our functions

# get data
X = get_data('combined_csv.csv')
y = get_data('combined_csv.csv')

# split into input (X) and output (y) variables
del X['SleepLVL'] 
del y['HR'] 

# initialize model with parameters
model = hmm.GaussianHMM(n_components=6, n_iter = 100, covariance_type="full")
model.fit(X)

# remodel to improve accuracy (suggested by library programmer)
remodel = hmm.GaussianHMM(n_components=6, covariance_type="full",  n_iter = 100)
remodel.fit(X)
remodel.monitor_
predictionOutput = remodel.predict(X)

# Once we train a model, we will want to save it
with open("all_hmm.pkl", "wb") as f: pickle.dump(remodel, f)

# Saves as a .csv file
np.savetxt("HMMPrediction.csv", predictionOutput, delimiter=",")

# formatting of test data
X_test = X
y_test = predictionOutput
y_test = np.asarray(y_test)

# formatting of train data
X_train = X
y_train = y.to_numpy()
y_train = np.asarray(y_train)

# display accuracy scores
display(Accuracy_of_actual(y_test, y_train))
display(Accuracy_of_awake(y_test, y_train))

# Notes: Takes roughly 55 seconds to run
# run using 'python Code/HMM.py' in command line