"""
This file holds the logic representing a scalar value with a gradient.
"""

class Value:
    """Class representing a scalar value with its gradient"""

    def __init__(self, data, children=(), _op=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = _op
        self.topology = list()

    def __add__(self, other):
        """

        Args:
            other (_type_): _description_

        Returns:
            _type_: _description_
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out =  Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad # either 0 or out

        out._backward = _backward

        return out

    def backward(self):
        self.topology: list[Value] = []
        visited = set()

        def build_topology(val):
            if val not in visited:
                visited.add(val)
                for child in val._prev:
                    build_topology(child)

                self.topology.append(val)

        build_topology(self)

        self.grad = 1
        for val in reversed(self.topology):
            val._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"
