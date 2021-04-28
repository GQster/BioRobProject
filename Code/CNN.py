#From Lab 4
from functions import *     #imports our functions
#import matplotlib.pyplot as plt
#import datetime
#import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
#import seaborn as sn
#from keras.utils import to_categorical


df = get_dataone('Sleep_HR_0.csv')       #Returns Data frame

#df_FILT = FilterDF(df)

#sliding Window
window = sliding_window(df, window_size = 5 , stride = 1)
x, y = format_cnn_data(window)
#display(x.shape)



# We can now create our keras model for our simple CNN architecture
#create model
filt_model = Sequential()
#add model layers
filt_model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(5001,16,1)))
filt_model.add(Conv2D(32, kernel_size=3, activation='relu'))
filt_model.add(Flatten())
#model.add(Dense(256, activation='relu'))
filt_model.add(Dense(13, activation='softmax'))

# Let's look at a summary of our model
filt_model.summary()