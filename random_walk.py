import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple, Callable
from datetime import datetime, date


@dataclass
class RandomWalk:
    """
    A class to simulate random walk paths with drift and volatility.

    Attributes:
        S0 (float): Initial value
        mu (float): Drift parameter (must be >= 0)
        sigma (float): Volatility parameter (must be > 0)
        delta_t (float): Time step (must be > 0)
        steps (int): Number of steps to simulate
        normal_dist_func (Callable[[], float], optional): Custom normal distribution function
    """

    S0: float
    mu: float
    sigma: float
    delta_t: float
    steps: int
    normal_dist_func: Optional[Callable[[], float]] = None

    def __post_init__(self):
        """Validate input parameters."""
        if self.S0 <= 0:
            raise ValueError("Initial value (S0) must be positive")
        if self.mu < 0:
            raise ValueError("Drift (mu) must be non-negative")
        if self.sigma <= 0:
            raise ValueError("Volatility (sigma) must be positive")
        if self.delta_t <= 0:
            raise ValueError("Time step (delta_t) must be positive")
        if self.steps <= 0:
            raise ValueError("Number of steps must be positive")
        if self.normal_dist_func is not None:
            # Test the custom function
            try:
                test_value = self.normal_dist_func()
                if not isinstance(test_value, (int, float)):
                    raise ValueError(
                        "Custom normal distribution function must return a number")
            except Exception as e:
                raise ValueError(
                    f"Invalid custom normal distribution function: {str(e)}")

    def _generate_normal_samples(self, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """
        Generate normal distribution samples using either default or custom function.

        Args:
            size: Size of the output array

        Returns:
            np.ndarray: Array of normal distribution samples
        """
        if self.normal_dist_func is None:
            return np.random.normal(0, 1, size)
        else:
            # Generate samples using the custom function
            if isinstance(size, int):
                return np.array([self.normal_dist_func() for _ in range(size)])
            else:
                return np.array([[self.normal_dist_func() for _ in range(size[1])]
                                 for _ in range(size[0])])

    def generate_path(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a single random walk path.

        Args:
            random_seed (int, optional): Seed for random number generation

        Returns:
            np.ndarray: Array of path values
        """
        if random_seed is not None and self.normal_dist_func is None:
            np.random.seed(random_seed)

        # Initialize path array
        path = np.zeros(self.steps + 1)
        path[0] = self.S0

        # Generate random normal variables
        phi = self._generate_normal_samples(self.steps)

        # Generate path using the formula:
        # S_{i+1} = S_i * (1 + mu * delta_t + sigma * phi_i * sqrt(delta_t))
        for t in range(self.steps):
            path[t + 1] = path[t] * (1 + self.mu * self.delta_t +
                                     self.sigma * phi[t] * np.sqrt(self.delta_t))

        return path

    def generate_multiple_paths(self,
                                num_paths: int,
                                random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate multiple random walk paths.

        Args:
            num_paths (int): Number of paths to generate
            random_seed (int, optional): Seed for random number generation

        Returns:
            np.ndarray: Array of shape (num_paths, steps + 1) containing paths
        """
        if random_seed is not None and self.normal_dist_func is None:
            np.random.seed(random_seed)

        # Initialize paths array
        paths = np.zeros((num_paths, self.steps + 1))
        paths[:, 0] = self.S0

        # Generate all random normal variables at once
        phi = self._generate_normal_samples((num_paths, self.steps))

        # Generate all paths efficiently using broadcasting
        for t in range(self.steps):
            paths[:, t + 1] = paths[:, t] * (1 + self.mu * self.delta_t +
                                             self.sigma * phi[:, t] * np.sqrt(self.delta_t))

        return paths

    def plot_paths(self,
                   paths: np.ndarray,
                   title: str = "Random Walk Paths",
                   show_mean: bool = True,
                   confidence_interval: Optional[float] = 0.95,
                   figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot multiple random walk paths with optional statistics.

        Args:
            paths (np.ndarray): Array of paths to plot
            title (str): Plot title
            show_mean (bool): Whether to show mean path
            confidence_interval (float, optional): Confidence interval to show
            figsize (tuple): Figure size
        """
        time_points = np.arange(0, (self.steps + 1) *
                                self.delta_t, self.delta_t)

        plt.figure(figsize=figsize)

        # Plot individual paths with low alpha
        plt.plot(time_points, paths.T, alpha=0.1, color='blue')

        if show_mean:
            mean_path = np.mean(paths, axis=0)
            plt.plot(time_points, mean_path, 'r--',
                     label='Mean Path', linewidth=2)

        if confidence_interval is not None:
            alpha = (1 - confidence_interval) / 2
            lower = np.percentile(paths, alpha * 100, axis=0)
            upper = np.percentile(paths, (1 - alpha) * 100, axis=0)
            plt.fill_between(time_points, lower, upper,
                             alpha=0.2, color='gray',
                             label=f'{confidence_interval:.0%} Confidence Interval')

        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    def get_path_statistics(self, paths: np.ndarray) -> dict:
        """
        Calculate statistics for the simulated paths.

        Args:
            paths (np.ndarray): Array of simulated paths

        Returns:
            dict: Dictionary containing path statistics
        """
        final_values = paths[:, -1]
        all_values = paths.flatten()

        stats = {
            'mean_final': np.mean(final_values),
            'std_final': np.std(final_values),
            'min_final': np.min(final_values),
            'max_final': np.max(final_values),
            'mean_overall': np.mean(all_values),
            'std_overall': np.std(all_values),
            'min_overall': np.min(all_values),
            'max_overall': np.max(all_values),
            'skewness': (np.mean(final_values - np.mean(final_values))**3 /
                         np.std(final_values)**3),
            'kurtosis': (np.mean(final_values - np.mean(final_values))**4 /
                         np.std(final_values)**4 - 3)
        }

        return stats


if __name__ == "__main__":
    # Example usage with default normal distribution
    rw_default = RandomWalk(
        S0=100,          # Initial value
        mu=0.05,         # 5% drift
        sigma=0.2,       # 20% volatility
        delta_t=1/252,   # Daily steps (assuming 252 trading days)
        steps=252        # One year simulation
    )

    # Example with custom normal distribution function
    from scipy.stats import norm

    def custom_normal():
        """Custom normal distribution function with different parameters."""
        return norm.rvs(loc=0, scale=1.2)  # Higher volatility

    rw_custom = RandomWalk(
        S0=100,
        mu=0.05,
        sigma=0.2,
        delta_t=1/252,
        steps=252,
        normal_dist_func=custom_normal
    )

    # Generate paths with both distributions
    n_paths = 1000
    paths_default = rw_default.generate_multiple_paths(n_paths, random_seed=42)
    paths_custom = rw_custom.generate_multiple_paths(n_paths)

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Default distribution
    ax1.plot(paths_default.T, alpha=0.1, color='blue')
    ax1.plot(np.mean(paths_default, axis=0), 'r--',
             label='Mean Path', linewidth=2)
    ax1.set_title('Paths with Default Normal Distribution')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Custom distribution
    ax2.plot(paths_custom.T, alpha=0.1, color='green')
    ax2.plot(np.mean(paths_custom, axis=0), 'r--',
             label='Mean Path', linewidth=2)
    ax2.set_title('Paths with Custom Normal Distribution')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()
