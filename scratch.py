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
