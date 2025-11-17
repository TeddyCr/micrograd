"""Test implementation of the MLP object"""

import enum
from _pytest.python_api import approx
from micrograd.nn.mlp import MultiLayerPerceptron
from micrograd.value import Value

def test_simple_nn():
    """simple test with 1 input 1 neuron 1 output
    
    We should have 2 layers. our hidden layer neuron should
    hhave 1 layer
    """

    model = MultiLayerPerceptron(1, [1,1])
    assert len(model.layers) == 2

    for layer in model.layers:            
        assert len(layer.neurons) == 1
        assert len(layer.neurons[0].w) == 1


def test_mlp_with_two_hidden_layers():
    """2 hidden layer 1 input 2 neurons 2 output"""
    model = MultiLayerPerceptron(1, [2,2,1])

    assert len(model.layers) == 3

    for i, layer in enumerate(model.layers):
        assert len(layer.neurons) == 2 if i < 2 else len(layer.neurons) == 1
        assert len(layer.neurons[0].w) == 1 if i < 1 else len(layer.neurons[0].w) == 2


def test_simple_mlp():
    """Test simple nn object returns the expected values"""    
    model = MultiLayerPerceptron(1, [1,1])
    model.layers[0].neurons[0].w = [Value(1.3)]
    model.layers[0].neurons[0].b = Value(4)
    model.layers[1].neurons[0].w = [Value(2.4)]
    model.layers[1].neurons[0].b = Value(2)

    y_hat = model([2.0])
    assert approx(y_hat[0].data) == 17.84

    loss = (8.84 - y_hat[0])**2
    assert approx(loss.data) == 81
    loss.backward()

    expected_grad = {
        "1.3": 86.4,
        "4": 43.2,
        "2.4": 118.8,
        "2": 18,
    }

    for p in model.parameters():
        assert expected_grad.get(str(p.data)) == approx(p.grad)
