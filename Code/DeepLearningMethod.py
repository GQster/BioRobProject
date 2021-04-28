#from functions import *     #imports our functions
from functions import *     #imports our functions

# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
#dataset = loadtxt('/workspace/BioRobProject/Data/Merged/Sleep_HR_0.csv', delimiter=',')



#dataset = get_dataone('Sleep_HR_0.csv')
#display(dataset)
X = get_dataone('Sleep_HR_0.csv')
y = get_dataone('Sleep_HR_0.csv')

# split into input (X) and output (y) variables
del X['SleepLVL'] 
del y['HR'] 

#display("X\n:", X)
#display("y:\n", y)


# define the keras model
#The model expects rows of data with 1 variables (the input_dim=1 argument)
#The first hidden layer has 12 nodes and uses the relu activation function.
#The second hidden layer has 8 nodes and uses the relu activation function.
#̶T̶h̶e̶ ̶o̶u̶t̶p̶u̶t̶ ̶l̶a̶y̶e̶r̶ ̶h̶a̶s̶ ̶o̶n̶e̶ ̶n̶o̶d̶e̶ ̶a̶n̶d̶ ̶u̶s̶e̶s̶ ̶t̶h̶e̶ ̶s̶i̶g̶m̶o̶i̶d̶ ̶a̶c̶t̶i̶v̶a̶t̶i̶o̶n̶ ̶f̶u̶n̶c̶t̶i̶o̶n̶.̶
model = Sequential()
model.add(Dense(12, input_dim=1, activation='relu'))
model.add(Dense(8, activation='relu'))
#   model.add(Dense(1, activation='sigmoid'))       #sigmoid to ensure output is between 0 and 1
