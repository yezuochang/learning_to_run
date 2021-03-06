

from keras.models import Sequential
from keras.layers.core import Dense, Activation
import random

class Model:
  def __init__(self, n_observation, n_action):
    model = Sequential()
    n = 32
    input_shape = ((n_observation+n_action)*2,)
    model.add(Dense(n, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Dense(n))
    model.add(Activation('relu'))
    model.add(Dense(n))
    model.add(Activation('relu'))
    model.add(Dense(n))
    model.add(Activation('tanh'))        
    model.add(Dense(n_observation))    
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    self.model = model

# n = 1000
# X_train = [None for _ in range(n)]
# y_train = [None for _ in range(n)]

# for c in range(n):
#   X_train[c] = [random.random() for _ in range(5)]
#   y_train[c] = [random.random() for _ in range(5)]


#model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
#score = model.evaluate(X_test, X_test, batch_size=16)
