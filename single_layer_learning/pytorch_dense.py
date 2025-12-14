import torch
from dataclasses import dataclass
from collections import OrderedDict
from .data_generator import generate_polynomial_data

@dataclass
class LinearApproximatorResult:
    final_loss: float
    state_dict: OrderedDict
    max_iterations_reached: int

    def __repr__(self):
        return f"""final loss: {self.final_loss}
max iterations reached: {self.max_iterations_reached}
state dictionary: {self.state_dict}
        """

class LinearApproximator:
    def __init__(self, learning_rate=0.005, max_iterations=50_000):
        self.max_iterations = max_iterations
        self.lr = learning_rate
        self.threshold = 10**-5

        self.dense = torch.nn.Linear(2, 1, bias=True)
        self.optimizer = torch.optim.Adam(self.dense.parameters(), lr=self.lr)

        self.x = None
        self.y_actual = None

    def predict(self):
        return self.dense.forward(self.x)
        

    def loss_fn(self):
        """This is the mean squared error."""
        y_pred = self.predict()
        error = self.y_actual - y_pred
        return torch.mean(error**2)


    def learn(self, x_data, y_data):
        """
        Train the model on the provided data

        Args:
            x_data (array-like): Input x values
            y_data (array-like): Target y values
        """
        assert len(x_data) == len(y_data), "x_data and y_data must have the same length"

        # Prepare data
        _x = torch.tensor(x_data, dtype=torch.float32)
        _x_squared = _x ** 2
        self.x = torch.stack([_x, _x_squared], dim=-1)
        self.y_actual = torch.tensor(y_data, dtype=torch.float32)[:, None]

        prev_loss = float("inf")

        for iteration in range(self.max_iterations):
            self.optimizer.zero_grad()
            current_loss = self.loss_fn()
            loss_value = current_loss.item()

            if abs(prev_loss - current_loss) < self.threshold:
                return LinearApproximatorResult(
                    state_dict=self.dense.state_dict(),
                    final_loss=current_loss,
                    max_iterations_reached=iteration
                )

            current_loss.backward()
            self.optimizer.step()

            if current_loss != current_loss:
                break

            prev_loss = loss_value

        final_loss = self.loss_fn().item()

        return LinearApproximatorResult(
            state_dict=self.dense.state_dict(),
            final_loss=final_loss,
            max_iterations_reached=iteration
        )

    def inference(self, x):
        x = torch.tensor([x, x**2], dtype=torch.float32)[None, :]

        return self.dense.forward(x)


linear_approximator = LinearApproximator(learning_rate=1e-3)

x_data, y_data = generate_polynomial_data(num_samples=41, x_min=-20, x_max=20, noise_std=0.0)
res = linear_approximator.learn(x_data, y_data)
print(res)
print("Prediction at x=-20:", linear_approximator.inference(-20).item())

x_noisy, y_noisy = generate_polynomial_data(num_samples=41, x_min=-20, x_max=20, noise_std=5.0)
res_noisy = linear_approximator.learn(x_noisy, y_noisy)
print(res_noisy)
print("Noisy prediction at x=-20:", linear_approximator.inference(-20).item())
