"""Class representing a single neuron object of the perceptron"""

import random
from micrograd.nn.abstract_nn import AbstractNeuralNet
from micrograd.value import Value

class Neuron(AbstractNeuralNet):
    """Implemenst the neuron of the perceptron
    
    Args:
        nin (int): number of parameter for the Neuron
        nonlin (bool):  
    """

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x) -> Value:
        """

        Args:
            x (_type_): _description_

        Returns:
            Value: _description_
        """
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return activation.relu() if self.nonlin else activation

    def parameters(self) -> list[Value]:
        return self.w + [self.b]
