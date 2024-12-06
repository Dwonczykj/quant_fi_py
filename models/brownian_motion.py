import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime, date


@dataclass
class BrownianMotion:
    """
    A class to simulate and visualize Brownian motion paths.

    Attributes:
        years (float): Time horizon in years
        steps_per_year (int): Number of steps per year
        drift (float): Annual drift term (mu)
        volatility (float): Annual volatility (sigma)
        initial_value (float): Starting value for the process
    """

    years: float
    steps_per_year: int = 252  # Default to daily steps (trading days)
    drift: float = 0.0
    volatility: float = 1.0
    initial_value: float = 0.0

    def __post_init__(self):
        self.dt = 1 / self.steps_per_year
        self.total_steps = int(self.years * self.steps_per_year)
        self.time_points = np.linspace(0, self.years, self.total_steps + 1)

    def generate_path(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a single Brownian motion path.

        Args:
            random_seed (int, optional): Seed for random number generation

        Returns:
            np.ndarray: Array of path values
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Generate random increments
        dW = np.random.normal(0, np.sqrt(self.dt), self.total_steps)

        # Calculate the path using the Euler-Maruyama method
        W = np.zeros(self.total_steps + 1)
        W[0] = self.initial_value

        for t in range(self.total_steps):
            W[t + 1] = W[t] + self.drift * self.dt + self.volatility * dW[t]

        return W

    def generate_multiple_paths(self,
                                num_paths: int,
                                random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate multiple Brownian motion paths.

        Args:
            num_paths (int): Number of paths to generate
            random_seed (int, optional): Seed for random number generation

        Returns:
            np.ndarray: Array of shape (num_paths, total_steps + 1) containing paths
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        paths = np.zeros((num_paths, self.total_steps + 1))
        paths[:, 0] = self.initial_value

        # Generate all random increments at once
        dW = np.random.normal(0, np.sqrt(self.dt),
                              (num_paths, self.total_steps))

        # Calculate all paths efficiently using broadcasting
        for t in range(self.total_steps):
            paths[:, t + 1] = paths[:, t] + self.drift * self.dt + \
                self.volatility * dW[:, t]

        return paths

    def plot_paths(self,
                   paths: np.ndarray,
                   title: str = "Brownian Motion Paths",
                   show_mean: bool = True,
                   confidence_interval: Optional[float] = 0.95,
                   figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot multiple Brownian motion paths with optional statistics.

        Args:
            paths (np.ndarray): Array of paths to plot
            title (str): Plot title
            show_mean (bool): Whether to show mean path
            confidence_interval (float, optional): Confidence interval to show
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)

        # Plot individual paths with low alpha
        plt.plot(self.time_points, paths.T, alpha=0.1, color='blue')

        if show_mean:
            mean_path = np.mean(paths, axis=0)
            plt.plot(self.time_points, mean_path, 'r--',
                     label='Mean Path', linewidth=2)

        if confidence_interval is not None:
            alpha = (1 - confidence_interval) / 2
            lower = np.percentile(paths, alpha * 100, axis=0)
            upper = np.percentile(paths, (1 - alpha) * 100, axis=0)
            plt.fill_between(self.time_points, lower, upper,
                             alpha=0.2, color='gray',
                             label=f'{confidence_interval:.0%} Confidence Interval')

        plt.title(title)
        plt.xlabel('Time (years)')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Example usage

    # Create a Brownian motion instance
    bm = BrownianMotion(
        years=5,
        steps_per_year=252,  # Daily steps
        drift=0.05,          # 5% annual drift
        volatility=0.2,      # 20% annual volatility
        initial_value=100    # Starting value
    )

    # Generate and plot multiple paths
    n_paths = 1000
    paths = bm.generate_multiple_paths(n_paths, random_seed=42)

    # Plot with all optional features
    bm.plot_paths(
        paths,
        title=f"{n_paths} Brownian Motion Paths\n"
        f"(μ={bm.drift:.1%}, σ={bm.volatility:.1%})",
        show_mean=True,
        confidence_interval=0.95
    )

    # Plot just the paths without statistical overlays
    bm.plot_paths(
        paths,
        title="Raw Brownian Motion Paths",
        show_mean=False,
        confidence_interval=None
    )

    # Generate and plot a single path
    single_path = bm.generate_path(random_seed=42)
    plt.figure(figsize=(12, 6))
    plt.plot(bm.time_points, single_path)
    plt.title("Single Brownian Motion Path")
    plt.xlabel("Time (years)")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.show()
