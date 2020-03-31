""" DQN in Code - Custom RL Exceptions

DQN Code as in the book Deep Reinforcement Learning, Chapter 9.

Runtime: Python 3.6.5
DocStrings: None

Author : Mohit Sewak (p20150023@goa-bits-pilani.ac.in)

"""

class PolicyDoesNotExistException(Exception):
    pass


class InsufficientPolicyParameters(Exception):
    pass


class FunctionNotImplemented(Exception):
    pass
