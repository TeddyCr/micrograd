"""
Mixin class for neural net operation in Neuron, Layer
"""

from abc import ABC, abstractmethod

from micrograd.value import Value

class AbstractNeuralNet(ABC):
    """
    Abstract class for component of the neural net (neuron,
    layer and multi layer perceptron class)
    """

    def zero_grad(self):
        """Set all parameter gradient to 0"""
        for p in self.parameters:
            p.grad = 0

    @abstractmethod
    def parameters(self) -> list[Value]:
        """Parameter method to be implemented by child classes

        Returns:
            list[Value]

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("Class must be implemented by children")
