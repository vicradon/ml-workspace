import numpy as np
import torch

f_of_x = lambda x: x**2 - 5

iter_count = 1000
threshold = 10**-5
learning_rate = 0.01

starting_point = -3.0

x = torch.tensor([starting_point], requires_grad=True)

for i in range(iter_count):
    loss = f_of_x(x)
    loss.backward()

    grad_value = x.grad.item()
    if abs(grad_value) < threshold:
        print("Computed after", i, "iterations")
        break

    with torch.no_grad():
        x -= learning_rate * x.grad

    x.grad.zero_()

print("final values: {:.4f} {:.4f}".format(x.item(), f_of_x(x).item()))