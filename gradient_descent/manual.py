import numpy as np
from matplotlib import pyplot as plt
from utils import make_gif_from_plots, plot_state

f_of_x = lambda x: x**2 - 5
gradient_of_f = lambda x: 2*x

x_range = np.arange(-4, 4, 0.1)
y_range = f_of_x(x_range)


plt.plot(x_range, y_range)
output_file = "plots/plot.png"
plt.savefig(output_file, dpi=200, bbox_inches="tight")

iter_count = 1000
threshold = 10**-5
learning_rate = 0.01

gradient_descent = lambda x: x - learning_rate * gradient_of_f(x)

computed_pairs = []

x_start = -3
y_start = 4

for i in range(iter_count):
    x_new = gradient_descent(x_start)
    y_new = f_of_x(x_new)

    computed_pairs.append((x_new, y_new))

    if abs(x_new - x_start) >= threshold:
        plot_state(
            label="f(x) = x^2 - 5",
            iteration=i,
            skip_after=30,
            x_val=x_new,
            y_val=y_new,
            x_range=x_range,
            y_range=y_range
        )

    if abs(x_new - x_start) <= threshold:
        break

    x_start = x_new

print(computed_pairs[-3:])



print("Found mininum after", len(computed_pairs), "iterations")
make_gif_from_plots("plots")