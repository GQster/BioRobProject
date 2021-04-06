from functions import *     #imports our functions


#Fixdata()                                              # Fixes all our data



#Hidden Mrkov Model

# https://towardsdatascience.com/hidden-markov-model-implemented-from-scratch-72865bda430e                  example library
# https://jonathan-hui.medium.com/machine-learning-hidden-markov-model-hmm-31660d217a61                     example code
# https://ghassemi.xyz/static/documents/Ghassemi_NIH(HMM)_2014.pdf                                          example paper


# df_merged = pd.read_csv('Data/Merged/{}'.format(sleep_file))
# df_merged.columns =['','TimeSec','SleepLVL','HR']

# X = sleep
# Y = HR
# N = number of states the hidden variable (sleep) can be in at any time t. (0-5)
X, Y = get_data()                                                                # Returns X and Y. X holding sleep states and Y holding HR data
#display(X, Y)


model = hmm.GaussianHMM(n_components=2, covariance_type="full")
n_components = 2                # number of states in the model                 (awake, asleep)
model.fit(X)
# model.startprob_ = np.array([0.8, 0.1, 0.025, 0.025, 0.025, 0.025])             # starting probabilities, most likely start awake, maybe in state N1 bc of -1 removal
# model.transmat_ = np.array([[0.8, 0.1, 0.025, 0.025, 0.025, 0.025],
#                           [0.1, 0.7, 0.1, 0.034, 0.033, 0.033],
#                            [0.033, 0.1, 0.7, 0.1, 0.034, 0.033],
#                            [0.033, 0.034, 0.1, 0.7, 0.1, 0.033],
#                            [0.033, 0.033, 0.034, 0.1, 0.7, 0.1],
#                            [0.025, 0.025, 0.025, 0.025, 0.1, 0.8]])

remodel = hmm.GaussianHMM(n_components=6, covariance_type="full")
remodel.fit(X)



