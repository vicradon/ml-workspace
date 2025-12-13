import numpy as np
from gradient_descent.mse_function import mse

# 1, 2, 0.5
x = np.array([-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5])
y_actual = np.array([ 3.5,  1. , -0.5, -1. , -0.5,  1. ,  3.5,  7. , 11.5, 17. , 23.5])

assert len(x) == len(y_actual)

n = len(x)

lr = 0.005

param0 = 0.0
param1 = 0.0
param2 = 0.0

param_new = lambda param, lr, gradient: param - (lr * gradient)


def compute_gradient():
    d0 = d1 = d2 = 0.0
    differential_coef = -2/n

    # summation in mse
    for i in range(n):
        y_hat = param0 + param1*x[i] + param2*((x[i])**2)
        error = y_actual[i] - y_hat

        d0 += error
        d1 += error * x[i]
        d2 += error * x[i]**2

    d0 *= differential_coef
    d1 *= differential_coef
    d2 *= differential_coef

    return d0, d1, d2

for step in range(1000):
    d0, d1, d2 = compute_gradient()

    # parameter updates
    param0 = param_new(param0, lr, d0)
    param1 = param_new(param1, lr, d1)
    param2 = param_new(param2, lr, d2)

print(param0)
print(param1)
print(param2)