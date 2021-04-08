from functions import *     #imports our functions

#preprocess_data()                                                          # Fixes all our data. 
#combineAllCSVs()                                                           # Merges all CSV files into one. Columns: ['Original Index','TimeSec', 'SleepLVL', 'HR']




# df_merged = pd.read_csv('Data/Merged/{}'.format(sleep_file))
# df_merged.columns =['','TimeSec','SleepLVL','HR']

# X = sleep
# Y = HR
# N = number of states the hidden variable (sleep) can be in at any time t. (0-5)
os.chdir("/workspace/BioRobProject")
X = get_dataALL('combined_csv.csv')                                               # Returns X and Y. X holding sleep states and Y holding HR data
#X = get_dataone('Sleep_HR_0.csv')   
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


np.savetxt("mainPrediction.csv", predictionOutput, delimiter=",")                      # saves as a .csv 


print('Score:')
print('\n')
remodel.score(X)
display(remodel.score(X))
print('Means:')
display(remodel.means_)
print('transmat:')
display(remodel.transmat_)