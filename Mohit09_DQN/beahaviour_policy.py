""" DQN in Code - BehaviorPolicy

DQN Code as in the book Deep Reinforcement Learning, Chapter 9.

Runtime: Python 3.6.5
Dependencies: numpy
DocStrings: GoogleStyle

Author : Mohit Sewak (p20150023@goa-bits-pilani.ac.in)

"""

# General imports
import logging
import numpy as np
# Import of custom exception classes implemented to make the error more understandable
from rl_exceptions import PolicyDoesNotExistException, InsufficientPolicyParameters, FunctionNotImplemented

# Configure logging for the project
# Create file logger, to be used for deployment
# logging.basicConfig(filename="Chapter09_BPolicy.log", format='%(asctime)s %(message)s', filemode='w')
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
        if "epsilon" not in policy_parameters:
            raise InsufficientPolicyParameters("epsilon not available")
        self.epsilon = self.policy_parameters["epsilon"]
        self.min_epsilon = None
        self.epsilon_decay_rate = None
        logger.debug("Policy Type {}, Parameters Received {}".format(policy_type, policy_parameters))

    def getPolicy(self):
        """Get the requested behavior policy

        This function returns a function corresponding to the requested behavior policy

            Args:
                None

            Returns:
                function: A function of the requested behavior policy type.

            Raises:
                PolicyDoesNotExistException: When a policy corresponding to the parameter policy_type is not
                                            implemented.
                InsufficientPolicyParameters: When a required policy parameter is not available
                                            (or key spelled incorrectly).

        """

        if self.policy_type == "epsilon_greedy":
            return self.return_epsilon_greedy_policy()
        elif self.policy_type == "epsilon_decay":
            self.epsilon = self.policy_parameters["epsilon"]
            if "min_epsilon" not in self.policy_parameters:
                raise InsufficientPolicyParameters("EpsilonDecay policy also requires the min_epsilon parameter")
            if "epsilon_decay_rate" not in self.policy_parameters:
                raise InsufficientPolicyParameters("EpsilonDecay policy also requires the epsilon_decay_rate parameter")
            self.min_epsilon = self.policy_parameters["min_epsilon"]
            self.epsilon_decay_rate = self.policy_parameters["epsilon_decay_rate"]
            return self.return_epsilon_decay_policy()
        else:
            raise PolicyDoesNotExistException("The selected policy does not exists! The implemented policies are "
                                              "epsilon-greedy and epsilon-decay")

    def return_epsilon_decay_policy(self):
        """Epsilon-Decay Policy Implementation

            This is the implementation of the Epsilon-Decay policy as returned by the getPolicy method when
            "epsilon-decay" policy type is selected.

            Returns:
                function: a function that could be directly called for selecting the recommended action as per e-decay.

        """

        def choose_action_by_epsilon_decay(values_of_all_possible_actions):
            """Action selection by epsilon_decay policy

                This is the base function that is actually invoked in each iteration to return the recommended action
                index as per the desired e_decay policy.

                Args:
                    values_of_all_possible_actions (array): A float array of the action values from which the
                                                            recommended action has to be chosen

                Returns:
                     int: The index of the recommended action as per the policy

            """
            logger.debug("Taking e-decay action for action values"+str(values_of_all_possible_actions))
            prob_taking_best_action_only = 1 - self.epsilon
            prob_taking_any_random_action = self.epsilon / self.n_actions
            action_probability_vector = [prob_taking_any_random_action] * self.n_actions
            exploitation_action_index = np.argmax(values_of_all_possible_actions)
            action_probability_vector[exploitation_action_index] += prob_taking_best_action_only
            chosen_action = np.random.choice(np.arange(self.n_actions), p=action_probability_vector)
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay_rate
            logger.debug("Decayed epsilon value after the current iteration: {}".format(self.epsilon))
            return chosen_action
        return choose_action_by_epsilon_decay

    def return_epsilon_greedy_policy(self):
        """Epsilon-Greedy Policy Implementation

            This is the implementation of the Epsilon-Greedy policy as returned by the getPolicy method when
            "epsilon-greedy" policy type is selected.

            Returns:
                function: a function that could be directly called for selecting the recommended action as per e-greedy.

        """

        def choose_action_by_epsilon_greedy(values_of_all_possible_actions):
            """Action selection by epsilon-Greedy policy

                This is the base function that is actually invoked in each iteration to return the recommended action
                index as per the desired e_decay policy.

                Args:
                    values_of_all_possible_actions (array): A float array of the action values from which the
                                                            recommended action has to be chosen

                Returns:
                     int: The index of the recommended action as per the policy

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


if __name__ == "__main__":
    raise FunctionNotImplemented("This class needs to be imported and instantiated from a Reinforcement Learning"
                                 "agent class and does not contain any invokable code in the main function")