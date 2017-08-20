import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
import dqn
DQNAgent = dqn.DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# gp_kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))

# gpr = GaussianProcessRegressor(kernel=gp_kernel)

ENV_NAME = 'CartPole-v0'



# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
nsteps = 5000
ntrain = 4000
dqn.fit(env, nb_steps=nsteps, visualize=False, verbose=2)

import fit 
ff = fit.Model(n_observation=4, n_action=1)
# X = np.array(dqn.X)
# Y = np.array(dqn.Y)
dqn.X = np.array(dqn.X)
dqn.Y = np.array(dqn.Y)

X_train = dqn.X[1:ntrain]
Y_train = dqn.Y[1:ntrain]
X_test = dqn.X[ntrain+1:nsteps]
Y_test = dqn.Y[ntrain+1:nsteps]

#gpr.fit(X_train, Y_train)
#print np.array(Y_test)
#print gpr.predict(X_test)
meanY = np.mean(Y_train, 0)
stdY = np.std(Ytrain)
Y_train = np.subtract(Y_train, meanY)
Y_train = np.divide(Y_train, stdY)

Y_test = np.subtract(Y_test, meanY)
Y_test = np.divide(Y_test, stdY)


ff.model.fit(X_train, Y_train, nb_epoch=50, batch_size=32)
for i in range(5):
    print X_test[i]
# print np.array(X_test), "\n"
print np.array(Y_test), "\n"

y = ff.model.predict(X_test)
print y , "\n"
print y-Y_test, "\n"
# After training is done, we save the final weights.
# dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
# dqn.test(env, nb_episodes=5, visualize=False)
