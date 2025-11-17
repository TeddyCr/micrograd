"""Class representing a single layer object of the perceptron"""

from micrograd.nn.abstract_nn import AbstractNeuralNet
from micrograd.nn.neuron import Neuron
from micrograd.value import Value


class Layer(AbstractNeuralNet):
    """Implemenst the neuron of the perceptron
    
    Args:
        nin (int): number of parameter linked to the neurons
        nout (int): number of neurons to generate
    """

    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [
            p for n in self.neurons
            for p in n.parameters()
        ]
