"""
Collection of activation functions for networks.
Author: dulwin
Date: 25/04/2016
Version: 0.1.0
"""

import numpy as np


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1.0 - np.power(tanh(z), 2)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z):
    return np.array([max(0, np.max(x)) for x in z])


def relu_prime(z):
    return (z > 0).astype(float)
