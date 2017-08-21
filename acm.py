
import numpy as np
from rl.agents import DDPGAgent

class ACMAgent(DDPGAgent):
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory,
                gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                random_process=None, custom_model_objects={}, target_model_update=.001, **kwargs):
        super(ACMAgent, self).__init__(**kwargs)
        self.X = []
        self.Y = []
        self.x = []
    
    def on_step_end(self, step, logs):
        print 'step', step

    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
        visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
        nb_max_episode_steps=None):
        callbacks = [] if not callbacks else callbacks[:]
        callbacks.append(self.on_step_end)
        super(ACMAgent, self).fit(self, env, nb_steps, action_repetition, 
        callbacks=callbacks, verbose=1, visualize=False, nb_max_start_steps=0, 
        start_step_policy=None, log_interval=10000, 
        nb_max_episode_steps=None)