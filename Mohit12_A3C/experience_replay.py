""" A3C in Code - ExperienceReplayMemory

A3C Code as in the book Deep Reinforcement Learning, Chapter 12.

Runtime: Python 3.6.5
DocStrings: GoogleStyle

Author : Mohit Sewak (p20150023@goa-bits-pilani.ac.in)

"""

# General Imports
import logging
import random
# Import for data structure for different types of memory
from collections import deque

# Configure logging for the project
# Create file logger, to be used for deployment
# logging.basicConfig(filename="Chapter09_BPolicy.log", format='%(asctime)s %(message)s', filemode='w')
logging.basicConfig()
# Creating a stream logger for receiving inline logs
logger = logging.getLogger()
# Setting the logging threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


class ExperienceReplayMemory:
    """Base class for all the extended versions for the ExperienceReplayMemory class implementation
    """
    pass


class SimpleListBasedMemory(ExperienceReplayMemory):
    """Simple Memory Implementation for A3C Workers

    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        """Stores the state, action and reward for the A3C
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        """Resets the memory
        """
        self.__init__()


class SequentialDequeMemory(ExperienceReplayMemory):
    """Extension of the ExperienceReplayMemory class with deque based Sequential Memory

           Args:
                queue_capacity (int): The maximum capacity (in terms of the number of  experience tuples) of the memory
                buffer.

    """

    def __init__(self, queue_capacity=2000):
        self.queue_capacity = 2000
        self.memory = deque(maxlen=self.queue_capacity)

    def add_to_memory(self,experience_tuple):
        """Add an experience tuple to the memory buffer

            Args:
                experience_tuple (tuple): A tuple of experience for training. In case of Q learning this tuple could be
                (S, A, R, S) with optional done_flag and in case of SARSA it could have an additional action element.

        """
        self.memory.append(experience_tuple)

    def get_random_batch_for_replay(self, batch_size=64):
        """Get a random mini-batch for replay from the Sequential memory buffer

            Args:
                batch_size (int): The size of the batch required

            Returns:
                list: list of the required number of experience tuples

        """
        return random.sample(self.memory,batch_size)

    def get_memory_size(self):
        """Get the size of the occupied buffer

            Returns:
                int: The number of the experience tuples already in memory
        """
        return len(self.memory)


if __name__ == "__main__":
    raise NotImplementedError("This class needs to be imported and instantiated from a Reinforcement Learning "
                              "agent class and does not contain any invokable code in the main function")