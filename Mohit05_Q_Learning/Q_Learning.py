""" Q Learning in Code

Q Learning Code (on custom environment as created in Chapter 2) as in the book Deep Reinforcement Learning, Chapter 5.

Runtime: Python 3.6.5
Dependencies: numpy, matplotlib (optional for plotting, else the plotting function can be commented)
DocStrings: GoogleStyle

Author : Mohit Sewak (p20150023@goa-bits-pilani.ac.in)

"""

#including necessary imports
import logging
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
# import custom exceptions that we coded to receive for more meaningful messages
from rl_exceptions import PolicyDoesNotExistException
# Import the custom environment we built in Chapter 02. We will use the same environment here.
from envs.gridworld import GridWorldEnv

# Configure logging for the project
# Create file logger, to be used for deployment
# logging.basicConfig(filename="Chapter05.log", format='%(asctime)s %(message)s', filemode='w')
logging.basicConfig()
# Creating a stream logger for receiving inline logs
logger = logging.getLogger()
# Setting the logging threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


class BehaviorPolicy:
    """Behavior Policy Class
    Class for different behavior policies for use with an Off-Policy Reinforcement Learning agent.

    Args:
        n_actions (int): the cardinality of the action space
        policy_type (str): type of behavior policy to be implemented.
            The current implementation contains only the "epsilon_greedy" policy.
        policy_parameters (dict) : A dict of relevant policy parameters for the requested policy.
            The epsilon-greedy policy as implemented requires only the value of the "epsilon" as float.

    Returns:
        None

    """

    def __init__(self, n_actions, policy_type = "epsilon_greedy", policy_parameters = {"epsilon":0.1}):
        self.policy = policy_type
        self.n_actions = n_actions
        self.policy_type = policy_type
        self.policy_parameters = policy_parameters

    def getPolicy(self):
        """Get the requested behavior policy

        This function returns a function corresponding to the requested behavior policy

        Args:
            None

        Returns:
            function: A function of the requested behavior policy type.

        Raises:
            PolicyDoesNotExistException: When a policy corresponding to the parameter policy_type is not implemented.
        """

        if self.policy_type == "epsilon_greedy":
            self.epsilon = self.policy_parameters["epsilon"]
            return self.return_epsilon_greedy_policy()
        else:
            raise PolicyDoesNotExistException("The selected policy does not exists! The implemented policies are "
                                              "epsilon-greedy.")

    def return_epsilon_greedy_policy(self):
        """Epsilon-Greedy Policy Implementation

        This is the implementation of the Epsilon-Greedy policy as returned by the getPolicy method when "epsilon-greedy"
        policy type is selected.

        Args:
            None

        Returns:
            function: a function that could be directly called for selecting the recommended action as per e-greedy.

        """

        def choose_action_by_epsilon_greedy(values_of_all_possible_actions):
            """Action-Selection by epsilon-Greedy

            This function chooses the action as the epsilon-greedy policy

            Args:
                values_of_all_possible_actions (list): A list of values of all actions in the current state

            Returns:
                int: the index of the action recommended by the policy

            """
            logger.debug("Taking e-greedy action for action values"+str(values_of_all_possible_actions))
            prob_taking_best_action_only = 1 - self.epsilon
            prob_taking_any_random_action = self.epsilon / self.n_actions
            action_probability_vector = [prob_taking_any_random_action] * self.n_actions
            exploitation_action_index = np.argmax(values_of_all_possible_actions)
            action_probability_vector[exploitation_action_index] += prob_taking_best_action_only
            chosen_action = np.random.choice(np.arange(self.n_actions), p=action_probability_vector)
            return chosen_action
        return choose_action_by_epsilon_greedy


class QLearning:
    """Q Learning Agent

    Class for training a Q Learning agent on any custom environment.

    Args:
        env (Object): An object instantiation of a custom env class like the GridWorld() environment
        number_episodes (int): The maximum number of episodes to be executed for training the agent
        discounting_factor (float): The discounting factor (gamma) used to discount the future rewards to current step
        behavior_policy (str): The behavior policy chosen (as q learning is off policy). Example "epsilon-greedy"
        epsilon (float): The value of epsilon, a parameters that defines the probability of taking a random action
        learning_rate (float): The learning rate (alpha) used to update the q values in each step

    Examples:
        q_agent = QLearning()

    """

    def __init__(self, env=GridWorldEnv(), number_episodes=500, discounting_factor=0.9,
                 behavior_policy="epsilon_greedy", epsilon=0.1, learning_rate=0.5):
        self.env = env
        self.n_states = env.get_statespace_len()
        self.n_actions = env.get_actionspace_len()
        self.stateDict = self.env.stateDict
        self.n_episodes = number_episodes
        self.gamma = discounting_factor
        self.alpha = learning_rate
        self.policy = BehaviorPolicy(n_actions=self.n_actions, policy_type=behavior_policy).getPolicy()
        self.policyParameter = epsilon
        self.episodes_completed = 0
        self.trainingStats_steps_in_each_episode = []
        self.trainingStats_rewards_in_each_episode = []
        self.q_table = np.zeros((self.n_states,self.n_actions),dtype = float)

    def train_agent(self):
        """Train the Q Learning Agent

        This is the main function to be called to start the training of the Q Learning agent in the given environment
        and with the given parameters.

        Args:
            None

        Returns:
            list: list (int) of steps used in each training episode
            list: list (float) of rewards received in each training episode

        Examples:
            training_statistics = q_agent.train_agent()

        """

        logger.debug("Number of States: {}".format(str(self.n_states)))
        logger.debug("Number of Actions: {}".format(str(self.n_actions)))
        logger.debug("Initial Q Table: {}".format(str(self.q_table)))
        for episode in range(self.n_episodes):
            logger.debug("Starting episode {}".format(episode))
            self.start_new_episode()
        return self.trainingStats_steps_in_each_episode, self.trainingStats_rewards_in_each_episode

    def start_new_episode(self):
        """Starts New Episode

        Function to Starts New Episode for training the agent. It also resets the environment.

        Args:
            None

        Returns:
            None

        """

        current_state = self.env.reset()
        logger.debug("Env reset, state received: {}".format(current_state))
        cumulative_this_episode_reward = 0
        for iteration in count():
            current_state_index = self.stateDict.get(current_state)
            policy_defined_action = self.policy(self.q_table[current_state_index])
            next_state, reward, done, _ = self.env.step(policy_defined_action)
            next_state_index = self.stateDict.get(next_state)
            logger.debug("Action Taken in Episode {}, Iteration {}: next_state={}, reward={}, done={}".
                         format(self.episodes_completed, iteration, next_state, reward, done))
            if done:
                self.trainingStats_rewards_in_each_episode.append(cumulative_this_episode_reward)
                self.trainingStats_steps_in_each_episode.append(iteration)
                self.episodes_completed += 1
                break
            cumulative_this_episode_reward += reward
            self.update_q_table(current_state_index, policy_defined_action, reward, next_state_index)
            current_state = next_state

    def update_q_table(self, current_state_index, action, reward, next_state_index):
        """Update Q Table

        Function to update the value of the q table

        Args:
            current_state_index (int): Index of the current state
            action (int): Index of the action taken in the current state
            reward (float): The instantaneous reward received by the agent by taking the action
            next_state_index (int): The index of the next state reached by taking the action

        Returns:
            None

        """

        target_q = reward + self.gamma * np.max(self.q_table[next_state_index])
        current_q = self.q_table[current_state_index, action]
        q_difference = target_q - current_q
        q_update = self.alpha * q_difference
        self.q_table[current_state_index,action] += q_update

    def plot_statistics(self):
        """Plot Training Statistics

        Function to plot training statistics of the Q Learning agent's training. This function plots the dual axis plot,
        with the episode count on the x axis and the steps and rewards in each episode on the y axis.

        Args:
            None

        Returns:
            None

        Examples:
            q_agent.plot_statistics()

        """

        trainingStats_episodes = np.arange(len(self.trainingStats_steps_in_each_episode))
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Episodes (e)')
        ax1.set_ylabel('Steps To Episode Completion', color="red")
        ax1.plot(trainingStats_episodes, self.trainingStats_steps_in_each_episode, color="red")
        ax2 = ax1.twinx()
        ax2.set_ylabel('Reward in each Episode', color="blue")
        ax2.plot(trainingStats_episodes, self.trainingStats_rewards_in_each_episode, color="blue")
        fig.tight_layout()
        plt.show()


if __name__ =="__main__":
    """Main function
    
    A sample implementation of the above classes (BehaviorPolicy and QLearning) for testing purpose.
    This function is executed when this file is run from the command propt directly or by selection.
    
    """

    logger.info("Q Learning - Creating the agent")
    q_agent = QLearning()
    logger.info("Q Learning - Training the agent")
    training_statistics = q_agent.train_agent()
    logger.info("Q Learning - Plotting training statistics")
    q_agent.plot_statistics()
