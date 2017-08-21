# Derived from keras-rl
import opensim as osim
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam

import numpy as np

# from rl.agents import DDPGAgent
import acm

from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from osim.env import *
from osim.http.client import Client

from keras.optimizers import RMSprop

import argparse
import math

from rl.callbacks import Callback

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--steps', dest='steps', action='store', default=10000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--token', dest='token', action='store', required=False)
args = parser.parse_args()

# Load walking environment
env = RunEnv(args.visualize)
env.reset()

nb_actions = env.action_space.shape[0]

# Total number of steps in training
nallsteps = args.steps

# Create networks for DDPG
# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(32))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('sigmoid'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
inp = concatenate([action_input, flattened_observation])
x = Dense(64)(inp)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())


inp = concatenate([action_input, flattened_observation])
x = Dense(64)(inp)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(env.observation_space.shape[0])(x)
x = Activation('tanh')(x)
env_model = Model(inputs=[action_input, observation_input], outputs=x)

class CB(Callback):
    def on_step_end(self, step, logs):
        print '\nstep', step
        print logs

# Set up the agent for training
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.noutput)
agent = acm.ACMAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  delta_clip=1.)
# agent = ContinuousddpgAgent(nb_actions=env.noutput, V_model=V_model, L_model=L_model, mu_model=mu_model,
#                            memory=memory, nb_steps_warmup=1000, random_process=random_process,
#                            gamma=.99, target_model_update=0.1)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
if args.train:
    agent.fit(env, nb_steps=nallsteps, visualize=False, 
        verbose=1, nb_max_episode_steps=env.timestep_limit, 
        log_interval=10000, callbacks=[CB()])
    # After training is done, we save the final weights.
    agent.save_weights(args.model, overwrite=True)
    import fit 
    ff = fit.Model(
        n_observation=env.observation_space.shape[0], 
        n_action=nb_actions)
    # X = np.array(ddpg.X)
    # Y = np.array(ddpg.Y)
    nsteps = nallsteps
    ntrain = int(nsteps*0.8)

    agent.X = np.array(agent.X)
    agent.Y = np.array(agent.Y)

    X_train = agent.X[1:ntrain]
    Y_train = agent.Y[1:ntrain]
    X_test = agent.X[ntrain+1:nsteps]
    Y_test = agent.Y[ntrain+1:nsteps]

    #gpr.fit(X_train, Y_train)
    #print np.array(Y_test)
    #print gpr.predict(X_test)
    meanY = np.mean(Y_train, 0)
    stdY = np.std(Y_train, 0)
    for k in range(len(stdY)):
        if stdY[k] == 0: 
            stdY[k] = 1.0
    Y_train = np.subtract(Y_train, meanY)
    Y_train = np.divide(Y_train, stdY)

    Y_test = np.subtract(Y_test, meanY)
    Y_test = np.divide(Y_test, stdY)


    ff.model.fit(X_train, Y_train, epochs=10000, batch_size=32)
    file = open('output.txt','w')
    for i in range(5):
        print X_test[i]
    # print np.array(X_test), "\n"
    print >>file, np.array(Y_test), "\n"
    y = ff.model.predict(X_test)
    print >>file, y , "\n"
    print >>file, y-Y_test, "\n"

# If TEST and TOKEN, submit to crowdAI
if not args.train and args.token:
    agent.load_weights(args.model)
    # Settings
    remote_base = 'http://grader.crowdai.org:1729'
    client = Client(remote_base)

    # Create environment
    observation = client.env_create(args.token)

    # Run a single step
    # The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
    while True:
        v = np.array(observation).reshape((env.observation_space.shape[0]))
        action = agent.forward(v)
        [observation, reward, done, info] = client.env_step(action.tolist())
        if done:
            observation = client.env_reset()
            if not observation:
                break

    client.submit()

# If TEST and no TOKEN, run some test experiments
if not args.train and not args.token:
    agent.load_weights(args.model)
    # Finally, evaluate our algorithm for 1 episode.
    agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=500)
