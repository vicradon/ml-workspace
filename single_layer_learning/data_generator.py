import numpy as np

y = lambda x: 1 + 2*x + 0.5*x**2

x_range = np.arange(-20, 21, 1)
y_range = y(x_range)

print(x_range.tolist())
print(y_range.tolist())