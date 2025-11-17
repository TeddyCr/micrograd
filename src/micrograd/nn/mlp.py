"""Class representing a multi-layer perceptron"""

from micrograd.nn.abstract_nn import AbstractNeuralNet
from micrograd.nn.layer import Layer
from micrograd.value import Value


class MultiLayerPerceptron(AbstractNeuralNet):
    """Instantiate our multi layer perceptron model

    Args:
        nin (int): number of input in the input layer
        nouts (list[int]): each element reprents a a layer where the 
            last element is the output layer and the element from n -> n-1
            represent the hidden layer
    """

    def __init__(self, nin: int, nouts: list[int]):
        sz = [nin] + nouts
        self.layers = [
            Layer(
                sz[i],
                sz[i+1],
                nonlin=i!=len(nouts)-1
            )
            for i in range(len(nouts))
        ]

    def __call__(self, x: list[Value]) -> Value:
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self) -> list[Value]:
        return [
            p for layer in self.layers
            for p in layer.parameters()
        ]
