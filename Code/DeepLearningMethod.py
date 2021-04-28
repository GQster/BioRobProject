#from functions import *     #imports our functions
from functions import *     #imports our functions

# first neural network with keras tutorial
#from numpy import loadtxt


# load the dataset
#dataset = loadtxt('/workspace/BioRobProject/Data/Merged/Sleep_HR_0.csv', delimiter=',')


# define the keras model
#The model expects rows of data with 1 variables (the input_dim=1 argument)
#The first hidden layer has 12 nodes and uses the relu activation function.
#The second hidden layer has 8 nodes and uses the relu activation function.
#̶T̶h̶e̶ ̶o̶u̶t̶p̶u̶t̶ ̶l̶a̶y̶e̶r̶ ̶h̶a̶s̶ ̶o̶n̶e̶ ̶n̶o̶d̶e̶ ̶a̶n̶d̶ ̶u̶s̶e̶s̶ ̶t̶h̶e̶ ̶s̶i̶g̶m̶o̶i̶d̶ ̶a̶c̶t̶i̶v̶a̶t̶i̶o̶n̶ ̶f̶u̶n̶c̶t̶i̶o̶n̶.̶
#model = Sequential()
#model.add(Dense(12, input_dim=1, activation='relu'))
#model.add(Dense(8, activation='relu'))
#   model.add(Dense(1, activation='sigmoid'))       #sigmoid to ensure output is between 0 and 1








# mlp for regression with mse loss function
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot


# generate regression dataset
#X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)
# standardize dataset
#X = StandardScaler().fit_transform(X)
#y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]



#dataset = get_dataone('Sleep_HR_0.csv')
#display(dataset)
X = get_dataone('Sleep_HR_0.csv')
y = get_dataone('Sleep_HR_0.csv')

# split into input (X) and output (y) variables
del X['SleepLVL'] 
del y['HR'] 

#display("X\n:", X)
#display("y:\n", y)


# split into train and test
n_train = int(X.shape[0]*0.80)                      # Sets 80% of the data as train
#print(n_train)
trainX, testX = np.array_split(X,[n_train])
#display(trainX)
#display(testX)

trainy, testy = np.array_split(y,[n_train])
#display(trainy)
#display(testy)

display("TX", trainX)
display("Ty", trainy)

#X.reset_index().values
#y.reset_index().values
Xnp = X.to_numpy()
ynp = y.to_numpy()
trainX, testX = np.array_split(Xnp,[n_train])
trainy, testy = np.array_split(ynp,[n_train])

display("TsetX", trainX)
display("Testy", trainy)












#trainX, testX = X[:n_train, :], X[n_train:, :]
#trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=opt)
# fit model
#history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
history = model.fit(X, y, epochs=10, validation_data=(testX, testy))
# evaluate the model
train_mse = model.evaluate(trainX, trainy, verbose=0)
test_mse = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
# plot loss during training
pyplot.title('Loss / Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()