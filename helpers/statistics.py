import numpy as np
from typing import Union, List, Optional, Tuple
from scipy import linalg
from scipy.stats import norm, t
import sympy as sp
from dataclasses import dataclass
from functools import reduce


@dataclass
class FinancialStatistics:
    """
    A comprehensive class for financial and mathematical statistics calculations.
    """

    def __init__(self, data: Optional[Union[List, np.ndarray]] = None):
        """
        Initialize with optional data.

        Args:
            data: Input data for calculations
        """
        self.data = np.array(data) if data is not None else None

    @staticmethod
    def arithmetic_mean(data: Union[List, np.ndarray]) -> float:
        """Calculate arithmetic mean."""
        return np.mean(data)

    @staticmethod
    def geometric_mean(data: Union[List, np.ndarray]) -> float:
        """Calculate geometric mean."""
        return np.exp(np.mean(np.log(np.abs(data))))

    @staticmethod
    def harmonic_mean(data: Union[List, np.ndarray]) -> float:
        """Calculate harmonic mean."""
        return len(data) / np.sum(1.0 / np.array(data))

    @staticmethod
    def sample_variance(data: Union[List, np.ndarray], ddof: int = 1) -> float:
        """
        Calculate sample variance with degrees of freedom adjustment.

        Args:
            data: Input data
            ddof: Delta degrees of freedom (default=1 for sample variance)
        """
        return np.var(data, ddof=ddof)

    @staticmethod
    def sample_std(data: Union[List, np.ndarray], ddof: int = 1) -> float:
        """Calculate sample standard deviation."""
        return np.std(data, ddof=ddof)

    @staticmethod
    def skewness(data: Union[List, np.ndarray]) -> float:
        """Calculate skewness."""
        data = np.array(data)
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        return (n * np.sum((data - mean) ** 3)) / ((n - 1) * (n - 2) * std ** 3)

    @staticmethod
    def kurtosis(data: Union[List, np.ndarray]) -> float:
        """Calculate excess kurtosis."""
        data = np.array(data)
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        return (n * (n + 1) * np.sum((data - mean) ** 4)) / \
               ((n - 1) * (n - 2) * (n - 3) * std ** 4) - 3

    @staticmethod
    def covariance_matrix(data: np.ndarray) -> np.ndarray:
        """Calculate covariance matrix."""
        return np.cov(data, rowvar=False)

    @staticmethod
    def correlation_matrix(data: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix."""
        return np.corrcoef(data, rowvar=False)

    @staticmethod
    def eigen_decomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform eigen decomposition of a matrix.

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        return linalg.eigh(matrix)

    @staticmethod
    def cholesky_decomposition(matrix: np.ndarray) -> np.ndarray:
        """Perform Cholesky decomposition."""
        return linalg.cholesky(matrix)

    @staticmethod
    def taylor_series(func: str, x0: float, n: int = 4) -> str:
        """
        Generate Taylor series expansion.

        Args:
            func: String representation of function
            x0: Point around which to expand
            n: Number of terms

        Returns:
            String representation of Taylor series
        """
        x = sp.Symbol('x')
        expr = sp.sympify(func)
        series = sp.series(expr, x, x0, n).removeO()
        return str(series)

    @staticmethod
    def matrix_rank(matrix: np.ndarray) -> int:
        """Calculate matrix rank."""
        return np.linalg.matrix_rank(matrix)

    @staticmethod
    def matrix_condition_number(matrix: np.ndarray) -> float:
        """Calculate matrix condition number."""
        return np.linalg.cond(matrix)

    @staticmethod
    def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve linear system Ax = b."""
        return np.linalg.solve(A, b)

    def confidence_interval(self,
                            confidence: float = 0.95,
                            distribution: str = 'normal') -> Tuple[float, float]:
        """
        Calculate confidence interval.

        Args:
            confidence: Confidence level (default=0.95)
            distribution: 'normal' or 't' distribution

        Returns:
            Tuple of (lower bound, upper bound)
        """
        if self.data is None:
            raise ValueError(
                "No data provided for confidence interval calculation")

        mean = np.mean(self.data)
        std = np.std(self.data, ddof=1)
        n = len(self.data)

        if distribution == 'normal':
            z = norm.ppf((1 + confidence) / 2)
            margin = z * std / np.sqrt(n)
        elif distribution == 't':
            t_stat = t.ppf((1 + confidence) / 2, df=n-1)
            margin = t_stat * std / np.sqrt(n)
        else:
            raise ValueError("Distribution must be 'normal' or 't'")

        return mean - margin, mean + margin

    @staticmethod
    def moving_average(data: Union[List, np.ndarray],
                       window: int,
                       type: str = 'simple') -> np.ndarray:
        """
        Calculate moving average.

        Args:
            data: Input data
            window: Window size
            type: 'simple' or 'exponential'

        Returns:
            Array of moving averages
        """
        data = np.array(data)
        if type == 'simple':
            weights = np.ones(window)
        elif type == 'exponential':
            weights = np.exp(np.linspace(-1., 0., window))
        else:
            raise ValueError("Type must be 'simple' or 'exponential'")

        weights = weights / weights.sum()
        return np.convolve(data, weights, mode='valid')

    @staticmethod
    def quantile(data: Union[List, np.ndarray], q: float) -> float:
        """Calculate quantile."""
        return np.percentile(data, q * 100)

    @staticmethod
    def value_at_risk(returns: Union[List, np.ndarray],
                      confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return -np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def conditional_value_at_risk(returns: Union[List, np.ndarray],
                                  confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = -np.percentile(returns, (1 - confidence) * 100)
        return -np.mean(returns[returns <= -var])


if __name__ == "__main__":
    # Example usage
    stats = FinancialStatistics()

    # Generate sample data
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)

    # Basic statistics
    print(f"Arithmetic Mean: {stats.arithmetic_mean(data):.4f}")
    print(f"Geometric Mean: {stats.geometric_mean(np.abs(data)):.4f}")
    print(f"Sample Variance: {stats.sample_variance(data):.4f}")
    print(f"Sample Std Dev: {stats.sample_std(data):.4f}")
    print(f"Skewness: {stats.skewness(data):.4f}")
    print(f"Kurtosis: {stats.kurtosis(data):.4f}")

    # Matrix operations
    matrix = np.array([[1, 0.5], [0.5, 1]])
    eigenvals, eigenvecs = stats.eigen_decomposition(matrix)
    print("\nEigenvalues:", eigenvals)
    print("Eigenvectors:\n", eigenvecs)

    # Taylor series example
    taylor = stats.taylor_series('exp(x)', 0, 4)
    print(f"\nTaylor series of exp(x) around 0: {taylor}")

    # Moving averages
    prices = np.cumsum(np.random.normal(0.001, 0.02, 100))
    ma = stats.moving_average(prices, 20)
    ema = stats.moving_average(prices, 20, type='exponential')

    # Risk metrics
    returns = np.random.normal(0.001, 0.02, 1000)
    var = stats.value_at_risk(returns)
    cvar = stats.conditional_value_at_risk(returns)
    print(f"\nValue at Risk (95%): {var:.4f}")
    print(f"Conditional VaR (95%): {cvar:.4f}")

    # Plotting
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(prices, label='Prices')
    plt.plot(np.arange(19, len(prices)), ma, label='20-day MA')
    plt.plot(np.arange(19, len(prices)), ema, label='20-day EMA')
    plt.title('Price Series with Moving Averages')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
