#from functions import *     #imports our functions
from functions import *     #imports our functions

#this Def needs to be in the CNN file
def evalModel(epochs = 10, verbose = 1):
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=epochs, verbose=verbose)
    # evaluate the model
    train_mse = model.evaluate(trainX, trainy, verbose=0)
    test_mse = model.evaluate(testX, testy, verbose=0)
    print("Loss Values:")
    print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
    return history #end of evalModel

# Get Data
X = get_data('combined_csv.csv')
y = get_data('combined_csv.csv')

# split into input (X) and output (y) variables
del X['SleepLVL'] 
del y['HR'] 

# split into train and test
n_train = int(X.shape[0]*0.80)                      # Sets 80% of the data as train
trainX, testX = np.array_split(X,[n_train])
trainy, testy = np.array_split(y,[n_train])


# define model
model = Sequential()
model.add(Dense(64, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(5, activation='softmax'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='mean_squared_logarithmic_error', optimizer=opt)

#changing number of epochs doesnt seem to increase accuracy 
evalModel(epochs = 3)                                   #trains the model

# Getting Predictions
pred = model.predict(testX)                             
predy = np.argmax(pred, axis = 1)

# Converts Pandas to numpy Array
testY = np.asarray(testy)

# Calc accuracies
display(Accuracy_of_awake(testY, predy))
display(Accuracy_of_actual(testY, predy))