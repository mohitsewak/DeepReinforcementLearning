""" A3C in Code - Custom RL Exceptions

A3C Code as in the book Deep Reinforcement Learning, Chapter 12.

Runtime: Python 3.6.5
DocStrings: None

Author : Mohit Sewak (p20150023@goa-bits-pilani.ac.in)

"""


class PolicyDoesNotExistException(Exception):
    pass


class InsufficientPolicyParameters(Exception):
    pass

