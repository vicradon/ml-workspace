import numpy as np

def mse(pred, actual):
    assert len(pred) == len(actual)
    n = len(pred)
    summation = 0.0

    for i in range(n):
        diff_squared = (actual[i] - pred[i])**2
        summation += diff_squared
        
    return summation/n


def d_of_mse(pred, actual, x):
    assert len(pred) == len(actual)
    n = len(pred)
    summation = 0.0

    for i in range(n):
        summation += (pred[i] - actual[i]) * x[i]

    return (2/n) * summation


eps = 1e-6
w = 2.0
x = [1.0, 2.0, 3.0]
actual = [2.0, 4.0, 6.0]

pred = [w * xi for xi in x]
pred_eps = [(w + eps) * xi for xi in x]

numeric_grad = (mse(pred_eps, actual) - mse(pred, actual)) / eps
analytic_grad = d_mse_dw(pred, actual, x)