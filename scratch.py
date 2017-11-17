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
