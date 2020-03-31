""" DDPG HighLevel implementation in Code

DDPG Code as in the book Deep Reinforcement Learning, Chapter 13.

Runtime: Python 3.6.5
Dependencies: keras, keras-rl, gym
DocStrings: GoogleStyle

Author : Mohit Sewak (p20150023@goa-bits-pilani.ac.in)
Inspired from: DDPG example implementation on Keras-RL github repository (keras-rl/keras-rl/blob/master/examples)
"""
# make general imports
import numpy as np
import os, logging
# Make keras specific imports
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
# Make reinforcement learning specific imports
import gym
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
# Configuring logging and setting logger to stream logs at DEBUG level
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class DDPG:
    """Deep Deterministic Policy Gradient Class

        This is an implementation of DDPG for continuous control tasks made using the high level keras-rl library.

        Args:
            env_name (str): Name of the gym environment
            weights_dir (str): Dir for storing model weights (for both actors and critic as separate files)
            actor_layers (list(int)): A list of int representing neurons in each subsequent the hidden layer in actor
            critic_layers (list(int)): A list of int representing neurons in each subsequent the hidden layer in actor
            n_episodes (int): Maximum training eprisodes
            visualize (bool): Whether a popup window with the environment view is required
    """
    def __init__(self,env_name = 'MountainCarContinuous-v0', weights_dir = "model_weights",
                 actor_layers = [64,64,32], critic_layers = [128,128,64], n_episodes=200, visualize=True):
        self.env_name=env_name
        self.env = gym.make(env_name)
        np.random.seed(123)
        self.env.seed(123)
        self.actor_layers = actor_layers
        self.critic_layers = critic_layers
        self.n_episodes = n_episodes
        self.visualize=visualize
        self.n_actions = self.env.action_space.shape[0]
        self.n_states = self.env.observation_space.shape
        self.weights_file = os.path.join(weights_dir,'ddpg_{}_weights.h5f'.format(self.env_name))
        self.actor = None
        self.critic = None
        self.agent = None
        self.action_input = None

    def _make_actor(self):
        """Internal helper function to create an actor custom model
        """
        self.actor = Sequential()
        self.actor.add(Flatten(input_shape=(1,) + self.n_states))
        for size in self.actor_layers:
            self.actor.add(Dense(size,activation='relu'))
        self.actor.add(Dense(self.n_actions,activation='linear'))
        self.actor.summary()

    def _make_critic(self):
        """Internal helper function to create an actor custom model
        """
        action_input = Input(shape=(self.n_actions,), name='action_input')
        observation_input = Input(shape=(1,) + self.n_states, name='observation_input')
        flattened_observation = Flatten()(observation_input)
        input_layer = Concatenate()([action_input, flattened_observation])
        hidden_layers = Dense(self.critic_layers[0], activation='relu')(input_layer)
        for size in self.critic_layers[1:]:
            hidden_layers = Dense(size, activation='relu')(hidden_layers)
        output_layer = Dense(1, activation='linear')(hidden_layers)
        self.critic = Model(inputs=[action_input, observation_input], outputs=output_layer)
        self.critic.summary()
        self.action_input = action_input

    def _make_agent(self):
        """Internal helper function to create an actor-critic custom agent model
        """
        if self.actor is None:
            self._make_actor()
        if self.critic is None:
            self._make_critic()
        memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=self.n_actions, theta=.15, mu=0., sigma=.3)
        self.agent = DDPGAgent(nb_actions=self.n_actions, actor=self.actor, critic=self.critic,
                               critic_action_input=self.action_input, memory=memory,
                               nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                               random_process=random_process, gamma=.99, target_model_update=1e-3)
        self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    def _load_or_make_agent(self):
        """Internal helper function to load an agent model, creates a new if no model weights exists
        """
        if self.agent is None:
            self._make_agent()
        if os.path.exists(self.weights_file):
            logger.info("Found existing weights for the model for this environment. Loading...")
            self.agent.load_weights(self.weights_file)

    def train(self):
        """Train the DDPG agent
        """
        self._load_or_make_agent()
        self.agent.fit(self.env, nb_steps=50000, visualize=self.visualize, verbose=1, nb_max_episode_steps=self.n_episodes)
        self.agent.save_weights(self.weights_file, overwrite=True)

    def test(self, nb_episodes=5):
        """Test the DDPG agent
        """
        logger.info("Testing the agents with {} episodes...".format(nb_episodes))
        self.agent.test(self.env, nb_episodes=nb_episodes, visualize=self.visualize, nb_max_episode_steps=200)


if __name__ == "__main__":
    """Main function for testing the A3C Master code's implementation
    """
    agent = DDPG()
    agent.train()
    agent.test()