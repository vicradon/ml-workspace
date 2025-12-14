import numpy as np


def generate_polynomial_data(num_samples=41, x_min=-20, x_max=20, noise_std=0.0):
    """
    Generate polynomial data for the function y = 1 + 2*x + 0.5*x^2

    Args:
        num_samples (int): Number of data points to generate
        x_min (int): Minimum x value
        x_max (int): Maximum x value
        noise_std (float): Standard deviation of Gaussian noise to add

    Returns:
        tuple: (x_values, y_values) as numpy arrays
    """
    # Generate x values evenly spaced
    x_values = np.linspace(x_min, x_max, num_samples)

    # Calculate y values using polynomial function
    y_values = 1 + 2*x_values + 0.5*x_values**2

    # Add noise if specified
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, y_values.shape)
        y_values += noise

    return x_values, y_values


def get_quadratic_features(x):
    """
    Transform 1D x into quadratic features [x, x^2] for PyTorch dense layer

    Args:
        x (array-like): Input x values

    Returns:
        np.array: Features matrix with shape (n_samples, 2)
    """
    x = np.array(x)
    return np.column_stack([x, x**2])


if __name__ == "__main__":
    # Example usage
    x_range, y_range = generate_polynomial_data(num_samples=41, x_min=-20, x_max=20, noise_std=0.0)
    print("X values:", x_range.tolist())
    print("Y values:", y_range.tolist())

    # Example with noise
    x_noisy, y_noisy = generate_polynomial_data(num_samples=41, x_min=-20, x_max=20, noise_std=2.0)
    print("\nNoisy Y values:", y_noisy.tolist())