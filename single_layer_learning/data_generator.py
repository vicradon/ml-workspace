import numpy as np


class DataGenerator:
    """
    A class that can generate different types of data for machine learning tasks.
    """

    def __init__(self, num_samples=41, x_min=-20, x_max=20, noise_std=0.0):
        """
        Initialize the DataGenerator with default parameters.

        Args:
            num_samples (int): Number of data points to generate
            x_min (int): Minimum x value
            x_max (int): Maximum x value
            noise_std (float): Standard deviation of Gaussian noise to add
        """
        self.num_samples = num_samples
        self.x_min = x_min
        self.x_max = x_max
        self.noise_std = noise_std
        self._x_values = None

    def _generate_x_values(self):
        """Generate x values evenly spaced between x_min and x_max."""
        if self._x_values is None:
            self._x_values = np.linspace(self.x_min, self.x_max, self.num_samples)
        return self._x_values

    def _add_noise(self, y_values):
        """Add Gaussian noise to y values if noise_std > 0."""
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, y_values.shape)
            y_values += noise
        return y_values

    def quadratic_polynomial(self, a0=1.0, a1=2.0, a2=0.5):
        """
        Generate quadratic polynomial data: y = a0 + a1*x + a2*x^2

        Args:
            a0 (float): Constant term
            a1 (float): Linear coefficient
            a2 (float): Quadratic coefficient

        Returns:
            tuple: (x_values, y_values) as numpy arrays
        """
        x_values = self._generate_x_values()
        y_values = a0 + a1*x_values + a2*x_values**2
        y_values = self._add_noise(y_values)
        return x_values, y_values

    def sine_wave(self, amplitude=1.0, frequency=1.0, phase=0.0, offset=0.0):
        """
        Generate sine wave data: y = amplitude * sin(frequency * x + phase) + offset

        Args:
            amplitude (float): Amplitude of the sine wave
            frequency (float): Frequency of the sine wave
            phase (float): Phase shift
            offset (float): Vertical offset

        Returns:
            tuple: (x_values, y_values) as numpy arrays
        """
        x_values = self._generate_x_values()
        y_values = amplitude * np.sin(frequency * x_values + phase) + offset
        y_values = self._add_noise(y_values)
        return x_values, y_values

    def square_wave(self, amplitude=1.0, frequency=1.0, phase=0.0, offset=0.0, duty_cycle=0.5):
        """
        Generate square wave data

        Args:
            amplitude (float): Amplitude of the square wave
            frequency (float): Frequency of the square wave
            phase (float): Phase shift
            offset (float): Vertical offset
            duty_cycle (float): Fraction of period where signal is high (0.0 to 1.0)

        Returns:
            tuple: (x_values, y_values) as numpy arrays
        """
        x_values = self._generate_x_values()
        # Calculate the phase-adjusted x values
        x_phase = frequency * x_values + phase
        # Normalize to [0, 2π] range
        x_phase = x_phase % (2 * np.pi)
        # Create square wave: high when in duty cycle portion, low otherwise
        y_values = np.where(x_phase < 2 * np.pi * duty_cycle, amplitude, -amplitude) + offset
        y_values = self._add_noise(y_values)
        return x_values, y_values

    def get_quadratic_features(self, x):
        """
        Transform 1D x into quadratic features [x, x^2] for PyTorch dense layer

        Args:
            x (array-like): Input x values

        Returns:
            np.array: Features matrix with shape (n_samples, 2)
        """
        x = np.array(x)
        return np.column_stack([x, x**2])


# Backward compatibility functions
def generate_polynomial_data(num_samples=41, x_min=-20, x_max=20, noise_std=0.0):
    """
    Generate polynomial data for the function y = 1 + 2*x + 0.5*x^2

    This is a backward compatibility function that uses the DataGenerator class.

    Args:
        num_samples (int): Number of data points to generate
        x_min (int): Minimum x value
        x_max (int): Maximum x value
        noise_std (float): Standard deviation of Gaussian noise to add

    Returns:
        tuple: (x_values, y_values) as numpy arrays
    """
    generator = DataGenerator(num_samples, x_min, x_max, noise_std)
    return generator.quadratic_polynomial()


def get_quadratic_features(x):
    """
    Transform 1D x into quadratic features [x, x^2] for PyTorch dense layer

    This is a backward compatibility function.

    Args:
        x (array-like): Input x values

    Returns:
        np.array: Features matrix with shape (n_samples, 2)
    """
    generator = DataGenerator()
    return generator.get_quadratic_features(x)


if __name__ == "__main__":
    # Example usage with DataGenerator class

    # Create a data generator
    generator = DataGenerator(num_samples=41, x_min=-20, x_max=20, noise_std=0.0)

    # Generate different types of data
    print("=== Quadratic Polynomial ===")
    x_quad, y_quad = generator.quadratic_polynomial(a0=1.0, a1=2.0, a2=0.5)
    print("First 5 X values:", x_quad[:5].tolist())
    print("First 5 Y values:", y_quad[:5].tolist())

    print("\n=== Sine Wave ===")
    x_sine, y_sine = generator.sine_wave(amplitude=2.0, frequency=0.5, phase=0.0, offset=1.0)
    print("First 5 X values:", x_sine[:5].tolist())
    print("First 5 Y values:", y_sine[:5].tolist())

    print("\n=== Square Wave ===")
    x_square, y_square = generator.square_wave(amplitude=1.5, frequency=1.0, phase=0.0, offset=0.0, duty_cycle=0.3)
    print("First 5 X values:", x_square[:5].tolist())
    print("First 5 Y values:", y_square[:5].tolist())

    print("\n=== With Noise ===")
    noisy_generator = DataGenerator(num_samples=10, x_min=-5, x_max=5, noise_std=0.5)
    x_noisy, y_noisy = noisy_generator.sine_wave(amplitude=1.0, frequency=1.0)
    print("Noisy sine wave Y values:", y_noisy.tolist())

    # Backward compatibility example
    print("\n=== Backward Compatibility ===")
    x_old, y_old = generate_polynomial_data(num_samples=5, x_min=-2, x_max=2, noise_std=0.0)
    print("Old function Y values:", y_old.tolist())