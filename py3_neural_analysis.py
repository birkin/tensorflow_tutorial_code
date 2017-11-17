"""Script for loading and analyzing animal data in Keras.

The goal is to load your data, visualize it, build a model, and see
how Keras gives you the opportunity to easily swap out modeling operations
to explore your hypotheses.

Check the Keras documentation to learn about the tools available to you:
https://keras.io
"""


# 0. Import packages, giving you the capability of running machine learning
# models and visualing your results
import numpy as np
from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt


# 1. Load your data
# Make sure that you launch python is in the directory where you downloaded
# the dropbox data
data_file = 'neuro_data.npy'  # Name of the datafile
(train_nrl, train_kmtcs), (val_nrl, val_kmtcs) = np.load(data_file, encoding="bytes")  # Load
# data into memory. Note that the syntax above means that we are loading a
# "numpy" data file, which contains two tuples. Each tuple stores two matrices,
# held in the parentheticals. By loading data in this way we split those tuples
# into their constituent variables.

# 2. Visualize your data
f = plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(train_nrl.transpose(), aspect='auto')
plt.ylabel('Neurons 1...63')
plt.title('Neural data')
plt.subplot(2, 1, 2)
plt.plot(train_kmtcs)
plt.ylabel('Height of the knee')
plt.xlabel('Time')
plt.title('Kinematic data')
plt.show()
plt.close(f)

# 3. Create a model in Keras (note that the model is not optimized at this
# point.
# It is not yet predictive of the dataset.)
model = Sequential()  # This tells keras to turn the variable 'model' into a
# container for our ML model architecture.

input_shape = train_nrl.shape
model.add(Dense(units=1, input_shape=(input_shape[-1],), kernel_initializer='uniform', activation='linear'))  # We add a layer of weights, which relate the
# kinematic data with the neural data (for predictions).

model.compile(loss='mse', optimizer='adam')  # Compile prepares the model for
# optimization. "MSE" is short for mean squared error, which measures goodness
# of model fit. "Adam" is a flexible optimizer. Let's you "set it and forget it."

model.fit(x=train_nrl, y=train_kmtcs, epochs=10, batch_size=16, verbose=1)  # Putting the pieces together:
# model.fit trains the weights (line 36) to decode X_train into y_train.
# "epochs" is the amount of times we will cycle through the entire dataset.
# Within each epoch are iterations, in which "batches" of the dataset are used
# to train the model. These batches are "batch_size" sized. Verbose tells you
# what keras is doing throughout.

preds = model.predict(val_nrl)  # model.predict gives you model predictions.
score = np.corrcoef(preds.ravel(), val_kmtcs.ravel())[0, 1]  # And now we
# correlate them with the actual behavior that elicited the neural activity
# is at predicting neural data *outside of train_nrl*. This is called
# generalization, and is critical for machine learning.

# Visualize your predictions
f = plt.figure()
plt.plot(preds)
plt.plot(val_kmtcs)
plt.xlabel('Time')
plt.ylabel('Height of knee')
plt.legend()
plt.title('Correlation is: %s' % score)
plt.show()
plt.close(f)
print ('*' * 30)
print ('Your correlation at step (3) is %s' % score)
print ('*' * 30)

###########

# Datathon and beyond?
# 5. Build a deeper model
model = Sequential()
model.add(Dense(32, input_shape=(input_shape[-1],), kernel_initializer='uniform', activation='relu'))  # We add a new layer with a different activation. This
# Will build a more powerful model. Read more: https://keras.io/activations/
model.add(Dense(1, kernel_initializer='uniform', activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(train_nrl, train_kmtcs, epochs=10, batch_size=16, verbose=1)
preds = model.predict(val_nrl)
score = np.corrcoef(preds.ravel(), val_kmtcs.ravel())[0, 1]
print ('*' * 30)
print ('Your correlation at step (5) is %s' % score)
print ('*' * 30)

# 6. Making it easier to learn
model = Sequential()
model.add(Dense(32, input_shape=(input_shape[-1],), kernel_initializer='uniform', activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # We use a "regularizer" to help our model
# learn from the data. This is as if your friend told you, "This teacher
# is correct 90% of the time, but full of it the other 10%." You would be
# somewhat skeptical, right? The regularizer is how skeptable our model is (0.01).
model.add(Dense(1, kernel_initializer='uniform', activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(train_nrl, train_kmtcs, epochs=10, batch_size=16, verbose=1)
preds = model.predict(val_nrl)
score = np.corrcoef(preds.ravel(), val_kmtcs.ravel())[0, 1]
print ('*' * 30)
print ('Your correlation at step (6) is %s' % score)
print ('*' * 30)
