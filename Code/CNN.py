#from functions import *     #imports our functions
from functions import *     #imports our functions

#this Def needs to be in the CNN file
def evalModel(epochs = 10, verbose = 0):
    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=epochs, verbose=verbose)
    # evaluate the model
    train_mse = model.evaluate(trainX, trainy, verbose=0)
    test_mse = model.evaluate(testX, testy, verbose=0)
    print("Loss Values:")
    print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
    return history #end of evalModel




#X = get_data('Sleep_HR_0.csv')
#y = get_data('Sleep_HR_0.csv')

X = get_data('combined_csv.csv')
y = get_data('combined_csv.csv')

# split into input (X) and output (y) variables
del X['SleepLVL'] 
del y['HR'] 

# split into train and test
n_train = int(X.shape[0]*0.80)                      # Sets 80% of the data as train
trainX, testX = np.array_split(X,[n_train])
trainy, testy = np.array_split(y,[n_train])
#display("TrainX", trainX)
#display("Trainy", trainy)

#display("TestX", testX)
#display("Testy", testy)

def modelTest():
# define model
    model = Sequential()
    model.add(Dense(64, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(6, activation='linear'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=opt)

    #evalModel(epochs = 5, verbose = 1)

    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=2, verbose=0)
    # evaluate the model
    train_mse = model.evaluate(trainX, trainy, verbose=0)
    test_mse = model.evaluate(testX, testy, verbose=0)
    #print("Loss Values:")
    print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
    return train_mse, test_mse
train = 0
test = 0
n = 5
for x in range(n):
    traintemp, testtemp = modelTest()
    train +=traintemp
    test += testtemp
print('AVG: Train: %.3f, Test: %.3f' % (train/n, test/n))








#print("plots")
# plot loss during training
#pyplot.title('Loss / Mean Squared Error')
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()