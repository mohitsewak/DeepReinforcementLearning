""" A3C in Code - A3C Worker

A3C Code as in the book Deep Reinforcement Learning, Chapter 12.

Runtime: Python 3.6.5
Dependencies: numpy, matplotlib, tensorflow (/ tensorflow-gpu), gym
DocStrings: GoogleStyle

Author : Mohit Sewak (p20150023@goa-bits-pilani.ac.in)

"""
import logging
# making general imports
import threading
import os
import numpy as np
# making deep learning and env related imports
import tensorflow as tf
import gym
# making imports of custom modules
from experience_replay import SimpleListBasedMemory
from actorcritic_model import ActorCriticModel
# Configuring logging and Creating logger, setting the log to streaming, and level as DEBUG
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class A3C_Worker(threading.Thread):
    """A3C Worker Class

        A3C Worker implemented as a thread (extends threading.Thread). The function computes the gradient of the policy
        and value networks' updates and then update the global network parameters of a similar policy and value
        networks after every some steps or after completion of a worker's episode.

    """
    global_constant_max_episodes_across_all_workers = 10000
    global_constant_total_steps_before_sync_for_any_workers = 10
    global_shared_best_episode_score = 0
    global_shared_total_episodes_across_all_workers = 0
    global_shared_semaphore = threading.Lock()
    global_shared_training_stats = []
    global_shared_episode_reward = 0

    def __init__(self, central_a3c_model, optimizer, worker_id, env_name, model_dir, discounting_factor=0.99):
        """Initialize the A3C worker instance

            Args:
                central_a3c_model (ActorCriticModel): An instance of the ActorCriticModel or similar model shared by the
                                                    A3C master
                optimizer (tf.train.Optimizer): An instance of the Optimizer object as used in the A3C_Master to update
                                                    its network parameters.
                worker_id (int): An integer representing the id of the instantiated worker.
                model_dir (str): dir for saving the model. Should be the same location from where the A3C_Master will
                                                    retrieve the trained model for playing.
                discounting_factor (float): Value of gamma, the discounting factor for future rewards.
        """
        super(A3C_Worker, self).__init__()
        self.central_a3c_model = central_a3c_model
        self.optimizer = optimizer
        self.worker_id = worker_id
        self.env_name = env_name
        self.env = gym.make(env_name).unwrapped
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.gamma = discounting_factor
        self.worker_model = ActorCriticModel(self.n_actions)
        self.memory = SimpleListBasedMemory()
        self.model_dir = model_dir
        self.this_episode_loss = 0
        self.this_episode_steps = 0
        self.this_episode_reward = 0
        self.this_episode_discountedreward = 0
        self.total_steps = 0
        self.steps_since_last_sync = 0
        logger.debug("Instantiating env for worker id: {}".format(self.worker_id))

    def run(self):
        """Thread's run function

            This is the default function that is executed when a the start() function of a class instance that extends
            threading.Thread class is called
            This function has the majority of the logic for the worker's functioning.

        """
        logger.debug("Starting execution of thread for worker id: {}".format(self.worker_id))
        while A3C_Worker.global_shared_total_episodes_across_all_workers < A3C_Worker.global_constant_max_episodes_across_all_workers:
            A3C_Worker.global_shared_total_episodes_across_all_workers += 1
            logger.info("Starting episode {}/{} using worker {}".format(
                A3C_Worker.global_shared_total_episodes_across_all_workers,
                A3C_Worker.global_constant_max_episodes_across_all_workers, self.worker_id))
            done = False
            current_state = self._reset_episode_stats()
            while not done:
                self._increment_all_steps()
                policy_logits, values = self.worker_model(tf.convert_to_tensor(np.random.random((1, self.n_states)), dtype=tf.float32))
                stochastic_action_probabilities = tf.nn.softmax(policy_logits)
                stochastic_policy_driven_action = np.random.choice(self.n_actions, p=stochastic_action_probabilities.numpy()[0])
                action = stochastic_policy_driven_action
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                self.this_episode_reward += reward
                self.memory.store(current_state, action, reward)

                if self.steps_since_last_sync >= A3C_Worker.global_constant_total_steps_before_sync_for_any_workers or done:
                    self._sync_worker_gradient_updates_with_global_model(done, new_state)
                if done:
                    A3C_Worker.global_shared_training_stats.append((self.worker_id, A3C_Worker.global_shared_total_episodes_across_all_workers,
                        self.this_episode_steps, self.this_episode_reward, self.this_episode_discountedreward, self.this_episode_loss))
                    if self.this_episode_reward > A3C_Worker.global_shared_best_episode_score:
                        self._update_best_model()

    def _update_best_model(self):
        """Rewrite the saved model with a beteer performing one

            This function rewrites the existing model (if any) saved in the model_dir, if any worker thread happens
            to obtain a better score in any of the episodes than the laste best score for an episode by any of the
            workers.

        """
        A3C_Worker.global_shared_best_episode_score = self.this_episode_reward
        with A3C_Worker.global_shared_semaphore:
            logger.info("Saving best model - worker:{}, episode:{}, episode-steps:{}, "
                        "episode-reward: {}, episode-discounted-reward:{}, episode-loss:{}".
                        format(self.worker_id,
                               A3C_Worker.global_shared_total_episodes_across_all_workers,
                               self.this_episode_steps, self.this_episode_reward,
                               self.this_episode_discountedreward, self.this_episode_loss))
            self.central_a3c_model.save_weights(os.path.join(self.model_dir, 'model_{}.h5'.format(self.env_name)))

    def _reset_episode_stats(self):
        """Internal helper function to reset the episodal statistics
        """
        self.this_episode_steps = 0
        self.this_episode_loss = 0
        self.this_episode_reward = 0
        self.this_episode_discountedreward = 0
        self.memory.clear()
        return self.env.reset()

    def _increment_all_steps(self):
        """Internal helper function to increment the step counts in a workers execution.
        """
        self.total_steps += 1
        self.steps_since_last_sync += 1
        self.this_episode_steps += 1

    def _sync_worker_gradient_updates_with_global_model(self, done, new_state):
        """Internal helper function to sync the gradient updates of the worker with the master

            This function is called whenever either an episodes ends or a pecified number of steps have elapsed since
            a particular worker synced with the master.
            In this process the losses for the policy and values are computed and the loss function is differentiated
            to fund the gradient. The so obtained gradient is used to update the weights of the master (global network)
            model parameters. Then the worker copies the updated weights of the master and resumes training.

        """
        with tf.GradientTape() as tape:
            total_loss = self._compute_loss(done, new_state)
        self.this_episode_loss += total_loss
        # Calculate local gradients
        grads = tape.gradient(total_loss, self.worker_model.trainable_weights)
        # Push local gradients to global model
        self.optimizer.apply_gradients(zip(grads, self.central_a3c_model.trainable_weights))
        # Update local model with new weights
        self.worker_model.set_weights(self.central_a3c_model.get_weights())
        self.memory.clear()
        self.steps_since_last_sync = 0

    def _compute_loss(self, done, new_state):
        """Function to compute the loss

            This method compute the loss as required by the _sync_worker_gradient_updates_with_global_model
            method to compute the gradients

        """
        if done:
          reward_sum = 0.  # terminal
        else:
          reward_sum = self.worker_model(tf.convert_to_tensor(new_state[None, :],dtype=tf.float32))[-1].numpy()[0]
        # Get discounted rewards
        discounted_rewards = []
        for reward in self.memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + self.gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        self.this_episode_discountedreward=np.float(discounted_rewards[0])
        # logger.info("Reward episode:{},step:{} = {}".format(A3C_Worker.global_shared_total_episodes_across_all_workers, self.this_episode_steps, self.memory.rewards[::-1]))
        # logger.info("Discounted-Reward episode:{},step:{} = {}".format(A3C_Worker.global_shared_total_episodes_across_all_workers, self.this_episode_steps, discounted_rewards))
        logits, values = self.worker_model(tf.convert_to_tensor(np.vstack(self.memory.states), dtype=tf.float32))
        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32) - values
        # Value loss
        value_loss = advantage ** 2
        # Calculate our policy loss
        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.memory.actions, logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss


if __name__ == "__main__":
    raise NotImplementedError("This class needs to be imported and instantiated from a Reinforcement Learning "
                              "agent class and does not contain any invokable code in the main function")