from dataclasses import dataclass
import numpy as np
from gradient_descent.mse_function import mse
import torch
from .data_generator import generate_polynomial_data


@dataclass
class LinearApproxResult:
    param0: float
    param1: float
    param2: float
    learned_at_step: int
    final_loss: float

class LinearApproximator:
    def __init__(self, x, y_actual, learning_rate=0.005, max_iterations=10_000):
        assert len(x) == len(y_actual)

        self.x = torch.tensor(x, dtype=torch.float32)/50
        self.y_actual = torch.tensor(y_actual, dtype=torch.float32)/100
        self.max_iterations = max_iterations

        self.lr = learning_rate
        self.threshold = 10**-5

        temp_params = torch.randn(3)

        self.param0 = torch.tensor(temp_params[0], requires_grad=True)
        self.param1 = torch.tensor(temp_params[1], requires_grad=True)
        self.param2 = torch.tensor(temp_params[2], requires_grad=True)




    def predict(self):
        return self.param0 + self.param1 * self.x + self.param2 * self.x**2

    def loss_fn(self):
        """ Thi is the mean squared error """
        y_pred = self.predict()
        error = self.y_actual - y_pred
        return torch.mean(error**2)


    def learn(self):
        prev_loss = float("inf")
        iteration_count = 0

        for iteration in range(self.max_iterations):
            current_loss = self.loss_fn()
            loss_value = current_loss.item()

            if abs(prev_loss - loss_value) < self.threshold:
                return LinearApproxResult(
                    self.param0.item(), 
                    self.param1.item(), 
                    self.param2.item(), 
                    iteration,  # ← Use 'iteration' not 'iteration_count'
                    loss_value
                )

            current_loss.backward()

            with torch.no_grad():
                self.param0 -= self.lr * self.param0.grad
                self.param1 -= self.lr * self.param1.grad
                self.param2 -= self.lr * self.param2.grad

                self.param0.grad.zero_()
                self.param1.grad.zero_()
                self.param2.grad.zero_()

            prev_loss = loss_value

        final_loss = self.loss_fn().item()

        return LinearApproxResult(
            self.param0.item(), 
            self.param1.item(), 
            self.param2.item(), 
            self.max_iterations,
            final_loss
        )


class ApproximatorTrainer:
    def __init__(self, datasets):
        self.datasets = datasets
        self.results = {}

    def trainall(self):
        for name in self.datasets:
            x, y, hyperparams = self.datasets[name]
            instance = LinearApproximator(x, y, **hyperparams)
            self.results[name] = instance.learn()

    def __repr__(self):
        lines = []
        for name, res in self.results.items():
            lines.append(
                f"{name}: param0={res.param0:.5f}, param1={res.param1:.5f}, "
                f"param2={res.param2:.5f} (learned at step {res.learned_at_step})"
            )
        return "\n".join(lines)


# Generate datasets using the data generator
x_small, y_small = generate_polynomial_data(num_samples=10, x_min=-5, x_max=4)
x_medium, y_medium = generate_polynomial_data(num_samples=11, x_min=-5, x_max=5)
x_large, y_large = generate_polynomial_data(num_samples=41, x_min=-20, x_max=20)

datasets = {
    "small_range": (x_small, y_small, {"learning_rate": 0.005, "max_iterations": 10_000}),
    "medium_range": (x_medium, y_medium, {"learning_rate": 0.005, "max_iterations": 10_000}),
    "large_range": (x_large, y_large, {"learning_rate": 0.0001, "max_iterations": 10_000}),
    "large_range_noisy": (*generate_polynomial_data(num_samples=41, x_min=-20, x_max=20, noise_std=5.0),
                          {"learning_rate": 0.0001, "max_iterations": 10_000}),
}


approx_trainer = ApproximatorTrainer(datasets)
approx_trainer.trainall()
print(approx_trainer)