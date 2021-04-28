from functions import *     #imports our functions
#from main import *          #imports HR_list and Sleep_list

#X = get_dataALL('combined_csv.csv') 
X = get_data('Sleep_HR_14.csv')                                              # Returns X and Y. X holding sleep states and Y holding HR data
print("fitting to HMM and decoding ...", end="")

# Make an HMM instance and execute fit
model = hmm.GMMHMM(n_components=6, n_iter=100)
#.fit(X)
#model = hmm.GaussianHMM(n_components=6, n_iter = 100, covariance_type="full").fit(X)
#model.startprob_ = np.array([0.8, 0.1, 0.025, 0.025, 0.025, 0.025])         # starting probabilities, most likely start awake, maybe in state N1 bc of -1 removal
#model.transmat_ = np.array([[0.8, 0.1, 0.025, 0.025, 0.025, 0.025],
#                            [0.1, 0.7, 0.1, 0.034, 0.033, 0.033],
#                            [0.033, 0.1, 0.7, 0.1, 0.034, 0.033],
#                            [0.033, 0.034, 0.1, 0.7, 0.1, 0.033],
#                            [0.033, 0.033, 0.034, 0.1, 0.7, 0.1],
#                            [0.025, 0.025, 0.025, 0.025, 0.1, 0.8]])
#model.means_ = np.array([[ 3, 59.7419355 ],
#    [ 1, 58.3814433 ],
#    [0, 61.46195652],
#     [2, 57.96015403],
#    [5, 62.67856466],
#    [4, 57.19449757]])
#model.covars_ ="full"
model.fit(X)

# Predict the optimal sequence of internal hidden state
#hidden_states = model.predict(X)

print("Transition matrix")
print(model.transmat_)
print()

print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    #print("var = ", np.diag(model.covars_[i]))



display(model.score(X))