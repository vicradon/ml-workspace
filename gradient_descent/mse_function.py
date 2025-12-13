import numpy as np

def mse(pred, actual):
    assert len(pred) == len(actual)
    n = len(pred)
    summation = 0.0

    for i in range(n):
        diff_squared = (actual[i] - pred[i])**2
        summation += diff_squared
        
    return summation/n


if __name__ == "__main__":
    pred1 = np.array([1.5, 2.5, 3.5])
    actual1 = np.array([1.0, 2.0, 3.0])

    pred2 = np.array([1.5, 2.5, 3.5])
    actual2 = np.array([1.3, 2.3, 3.3])

    error1 = mse(pred1, actual1)
    error2 = mse(pred2, actual2)

    print(error1)
    print(error2)


