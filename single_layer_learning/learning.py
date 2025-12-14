from dataclasses import dataclass
import numpy as np
from gradient_descent.mse_function import mse

# 1, 2, 0.5
# 1 + 2x + 0.5x**2

x1 = [ -5, -4, -3, -2, -1,  0,  1,  2,  3,  4]
y1_actual = [ 3.5,  1. , -0.5, -1. , -0.5,  1. ,  3.5,  7. , 11.5, 17.]

x2 = [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5]
y2_actual = [ 3.5,  1. , -0.5, -1. , -0.5,  1. ,  3.5,  7. , 11.5, 17. , 23.5]

x3 = [-20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
y3_actual = [161.0, 143.5, 127.0, 111.5, 97.0, 83.5, 71.0, 59.5, 49.0, 39.5, 31.0, 23.5, 17.0, 11.5, 7.0, 3.5, 1.0, -0.5, -1.0, -0.5, 1.0, 3.5, 7.0, 11.5, 17.0, 23.5, 31.0, 39.5, 49.0, 59.5, 71.0, 83.5, 97.0, 111.5, 127.0, 143.5, 161.0, 179.5, 199.0, 219.5, 241.0]


@dataclass
class LinearApproxResult:
    param0: float
    param1: float
    param2: float
    param3: float
    param4: float
    learned_at_step: int

class LinearApproximator:
    def __init__(self, x, y_actual, learning_rate=0.005, iter_count=10_000):
        assert len(x) == len(y_actual)
        self.x = np.array(x)
        self.y_actual = np.array(y_actual)
        self.iter_count = iter_count

        self.n = len(x)

        self.lr = learning_rate
        self.threshold = 10**-5

        self.param0 = 0.0
        self.param1 = 0.0
        self.param2 = 0.0
        self.param3 = 0.0
        self.param4 = 0.0
    
    def compute_gradient(self):
        d0 = d1 = d2 = 0.0
        diff_coef = -2/self.n

        # summation in mse
        for i in range(self.n):
            y_hat = self.param0 + self.param1*self.x[i] + self.param2*((self.x[i])**2) 
            error = self.y_actual[i] - y_hat

            d0 += error
            d1 += error * self.x[i]
            d2 += error * self.x[i]**2

        d0 *= diff_coef
        d1 *= diff_coef
        d2 *= diff_coef

        max_grad = 10.0
        d0 = np.clip(d0, -max_grad, max_grad)
        d1 = np.clip(d1, -max_grad, max_grad)
        d2 = np.clip(d2, -max_grad, max_grad)

        return d0, d1, d2

    def learn(self):
        prev_loss = float("inf")
        iteration_count = 0

        for step in range(self.iter_count):
            d0, d1, d2 = self.compute_gradient()

            # parameter updates param_new = param_current - lr *  gradient
            self.param0 -= self.lr*d0
            self.param1 -= self.lr*d1
            self.param2 -= self.lr*d2

            y_pred = self.param0 + self.param1*self.x + self.param2*(self.x**2)
            loss = mse(self.y_actual, y_pred)

            if abs(prev_loss - loss) <= self.threshold:
                iteration_count = step
                break

            prev_loss = loss

        return LinearApproxResult(self.param0, self.param1, self.param2, iteration_count)


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


datasets = {
    "la1": (x1, y1_actual, {}),
    "la2": (x2, y2_actual, {}),
    "la3": (x3, y3_actual, {"learning_rate": 0.0001, "iter_count": 40_000}), # not gonna learn
}


approx_trainer = ApproximatorTrainer(datasets)
approx_trainer.trainall()
print(approx_trainer)