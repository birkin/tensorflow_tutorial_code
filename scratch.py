# this is a comment
# we import packages at the top of your python scripts :)
import numpy as np # importing numpy package
import keras # importing the keras API
from keras.models import Sequential # allows us to define our models in Keras
from keras.layers import Dense # layers function
from keras.optimizers import SGD # optimizer for minimizing cost function
from sklearn import preprocessing # used for normalizing our data
from keras.layers import Dropout # dropout a method used to penalize for overfitting

# loading the housing data stored as a numpy file into training, validation and testing matricies
(x_train, y_train), (x_val, y_val), (x_test, y_test) = np.load( './housing.npy', encoding='bytes' )

print ("Shape of your training matrix and labels:")
print (np.shape(x_train)) # (350, 13): 350 rows/samples and 13 columns/features
print (np.shape(y_train)) # (350,): 350 labels correspoding to the training features
print(' ')
print ("Shape of your validation matrix and labels:")
print (np.shape(x_val)) # (50, 13): 50 rows/samples and 13 columns/features
print (np.shape(y_val)) # (50,): 50 labels correspoding to the validation features
print(' ')
print ("Shape of your test matrix and labels:")
print (np.shape(x_test)) # (106, 13): 106 rows/samples and 13 columns/features
print (np.shape(y_test)) # (106,): 106 labels correspoding to the test features
print(' ')

# preprocessing your data
# normalizing feature matrix
x_scaler = preprocessing.StandardScaler().fit(x_train) # we normalize with the training features
x_train = x_scaler.transform(x_train) # normalizing training features
x_val = x_scaler.transform(x_val) # normalizing validation features
x_test = x_scaler.transform(x_test) # normalizing test features

# normalizing target labels
y_scaler = preprocessing.StandardScaler().fit(y_train.reshape(-1,1)) # we normalize with the training labels
y_train = y_scaler.transform(y_train.reshape(-1,1)) # normalizing training labels
y_val = y_scaler.transform(y_val.reshape(-1,1)) # normalizing validaiton labels
y_test = y_scaler.transform(y_test.reshape(-1,1)) # normalizing test labels

# defining our model as Sequential (a linear stack of layers)
first_model = Sequential()
# adding our first layer to the neural network
first_model.add(Dense(13, input_dim=13, kernel_initializer='uniform', activation='relu'))


import pprint
print( '\n---\nfirst_model.__dict__, ```%s```\n---\n' % pprint.pformat(first_model.__dict__) )


# adding our ouptut layer to the neural network
first_model.add(Dense(1, kernel_initializer='uniform'))

# defining our optimizer sgd (stochastic gradient descent) and giving it a learning rate
sgd = SGD(lr=0.03)

# compiling our model we want to minimize the mean squared error and will use sgd to minimize this error
first_model.compile(loss='mean_squared_error', optimizer=sgd)
print( '\n---\nfirst_model.compile() just run\n---\n' )

# we train our model with the train data
# validate the model with the validation data every epoch
# an epoch is an entire iteration through the dataset
# batch size is how much data we are feeding into the model for a sgd update
first_model.fit(x_train, y_train, batch_size=5, validation_data=(x_val, y_val), epochs=30)
print( '\n---\nfirst_model.fit() just run\n---\n' )

