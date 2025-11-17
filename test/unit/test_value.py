"""
Test implementation for Value object
"""

from micrograd.value import Value
from pytest import approx

def test_value():
    """Simple test for value"""
    a = Value(2.0)
    y = Value(8.84)
    w00 = Value(1.3)
    w01 = Value(2.4)
    bh0 = Value(4)
    byhat = Value(2)

    expected_h0 = Value(6.6)
    expected_yhat = Value(17.84)
    expected_C0 = Value(81)

    expected_w00_grad = 86.4
    expected_w01_grad = 118.8
    expected_bh0_grad = 43.2

    actual_h0 = a * w00 + bh0
    assert actual_h0.data == expected_h0.data
    actual_yhat  = actual_h0 * w01 + byhat
    assert approx(actual_yhat.data) == expected_yhat.data

    actual_C0 = (actual_yhat - y)**2
    assert approx(actual_C0.data) == expected_C0.data

    actual_C0.backward()
    actual_w00_grad = _get_value_from_topo(actual_C0, 1.3)
    assert approx(actual_w00_grad.grad) == expected_w00_grad

    actual_w01_grad = _get_value_from_topo(actual_C0, 2.4)
    assert approx(actual_w01_grad.grad) == expected_w01_grad

    actual_bh0_grad = _get_value_from_topo(actual_C0, 4)
    assert approx(actual_bh0_grad.grad) == expected_bh0_grad


def _get_value_from_topo(val: Value, filter_: float) -> Value | None:
    return next(
        (v for v in val.topology if v.data == filter_),
        None
    )
