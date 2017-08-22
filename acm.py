
import numpy as np
from rl.agents import DDPGAgent
from rl.callbacks import Callback
class CB(Callback):
  def on_step_end(self, step, logs):
    self.model.on_step_end(step, logs)

class ACMAgent(DDPGAgent):
  def __init__(self, nb_actions, actor, critic, critic_action_input, memory,
        gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
        train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
        random_process=None, custom_model_objects={}, target_model_update=.001, **kwargs):
    super(ACMAgent, self).__init__(nb_actions, actor, critic, critic_action_input, memory,
        gamma, batch_size, nb_steps_warmup_critic, nb_steps_warmup_actor,
        train_interval, memory_interval, delta_range, delta_clip,
        random_process, custom_model_objects, target_model_update, **kwargs)
    self.X = []
    self.Y = []
    self.x = []

  def on_episode_begin(self, episode, logs={}):

  
  def on_step_end(self, step, logs):
    # state = self.memory.get_recent_state(observation)    
    print 'step', step, logs['observation']
    # if self.recent_observation is not None:
    #   # print self.recent_observation
    #   # print observation
    #   # print action
    #   for k in range(len(observation)):
    #     self.x.append(self.recent_observation[k])
    #   for k in range(len(action)):
    #     self.x.append(action[k])
    #   # self.x += self.recent_observation+action
    #   if len(self.x) == (len(observation)+len(action))*2:
    #     self.X.append(self.x)
    #     y = [None]*len(observation) 
    #     for k in range(len(observation)):
    #       y[k] = observation[k] - self.recent_observation[k]
    #     self.Y.append(y)
    #     # self.Y.append(observation-self.recent_observation)
    #     self.x = self.x[len(observation)+len(action):]


  def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
    visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
    nb_max_episode_steps=None):      
    callbacks = [] if not callbacks else callbacks[:]
    callbacks.append(CB())
    super(ACMAgent, self).fit(env, nb_steps, action_repetition, 
      callbacks, verbose, visualize, nb_max_start_steps, 
      start_step_policy, log_interval, 
      nb_max_episode_steps)