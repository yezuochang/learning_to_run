
import numpy as np
from rl.agents import DDPGAgent
from keras.callbacks import Callback

class ACMCallback(Callback):
    def on_step_end(self, step, logs):
        print 'step', step, logs

class ACMAgent(DDPGAgent):
    def __int__(self, *args, **kwargs):
        super(ACMAgent, self).__init__(*args, **kwargs)
        self.X = []
        self.Y = []
        self.x = []

    def on_epsode_begin(self, step, logs):
        print 'step', step

    def on_step_end(self, step, logs):
        print 'step', step

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
        visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
        nb_max_episode_steps=None):
        callbacks = [] if not callbacks else callbacks[:]
        callbacks.append(ACMCallback())
        super(ACMAgent, self).fit(env, nb_steps, action_repetition, 
            callbacks, verbose, visualize, nb_max_start_steps, 
            start_step_policy, log_interval, nb_max_episode_steps)