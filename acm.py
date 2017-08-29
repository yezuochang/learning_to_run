
import numpy as np
from rl.agents import DDPGAgent
from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate


class ACMCallback(Callback):
    def __init__(self, agent):
        self.agent = agent
    def on_step_end(self, step, logs):
        self.agent.on_step_end(step, logs)
        # print 'step', step
        # print 'memory', self.memory

class ACMAgent(DDPGAgent):
    def __init__(self, nb_actions, actor, critic, env_model, critic_action_input, memory,
                gamma=.99, batch_size=32, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,
                random_process=None, custom_model_objects={}, target_model_update=.001, **kwargs):
        super(ACMAgent, self).__init__(nb_actions, actor, critic, critic_action_input, 
                memory, gamma, batch_size, nb_steps_warmup_critic, nb_steps_warmup_actor,
                train_interval, memory_interval, delta_range, delta_clip,
                random_process, custom_model_objects, target_model_update, **kwargs)
        self.env_model = env_model
        self.env_model.compile(loss='mean_squared_error', optimizer='rmsprop')
        self.callback = ACMCallback(self)
        
        
    def on_epsode_begin(self, step, logs):
        # print 'step', step
        pass

    def on_step_end(self, step, logs):
        print '\nstep', step, self.nb_steps_warmup_actor
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            state1_batch = np.array(state1_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                # print e.state1, e.state1.shape
                targets = state1_batch.reshape(self.batch_size, len(e.state1[0]))
                state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                self.env_model.train_on_batch(state0_batch_with_action, targets)
                if self.step % 10 == 0:
                    y = self.env_model.predict_on_batch(state0_batch_with_action)
                    print targets
                    print y-targets
                    d1 = np.linalg.norm(targets)
                    d2 = np.linalg.norm(y-targets)
                    print d1, d2
                    print 'Error: ', d2/d1


    def fit(self, env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
        visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
        nb_max_episode_steps=None):
        callbacks = [] if not callbacks else callbacks[:]
        callbacks.append(self.callback)
        super(ACMAgent, self).fit(env, nb_steps, action_repetition, 
            callbacks, verbose, visualize, nb_max_start_steps, 
            start_step_policy, log_interval, nb_max_episode_steps)