from __future__ import division
from collections import deque
import os
import warnings

import numpy as np
import keras.backend as K
import keras.optimizers as optimizers

from rl.core import Agent
from rl.random import OrnsteinUhlenbeckProcess
from rl.util import *


# Derived from keras-rl
import opensim as osim
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam

import numpy as np

# from rl.agents import DDPGAgent
import ddpg
DDPGAgent = ddpg.DDPGAgent

from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from osim.env import *
from osim.http.client import Client

from keras.optimizers import RMSprop

import argparse
import math

# Load walking environment
env = RunEnv(False)
env.reset()

nb_actions = env.action_space.shape[0]

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
inp = concatenate([action_input, flattened_observation])
x = inp
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

out = critic([action_input, observation_input])
dx = K.gradients(out, action_input)
print dx
