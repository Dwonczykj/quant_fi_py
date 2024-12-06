import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Union, Optional, List
from datetime import datetime, date


@dataclass
class TimeSeriesReturns:
    """
    A class to calculate and analyze returns from a time series.

    Attributes:
        values (Union[List, np.ndarray, pd.Series]): Time series of values
        period (int): Number of periods to use for return calculation
    """

    values: Union[List, np.ndarray, pd.Series]
    period: int = 1  # Default to 1-period returns

    def __post_init__(self):
        # Convert input to numpy array if it isn't already
        self.values = np.array(self.values)

        # Validate inputs
        if len(self.values) < self.period + 1:
            raise ValueError(
                "Time series length must be greater than period length")
        if self.period < 1:
            raise ValueError("Period must be positive")

    def simple_returns(self) -> np.ndarray:
        """
        Calculate simple returns: (P_{t+n} - P_t) / P_t
        where n is the period length.

        Returns:
            np.ndarray: Array of simple returns
        """
        # Shift values by period length
        values_shifted = self.values[self.period:]
        values_original = self.values[:-self.period]

        # Calculate returns
        returns = (values_shifted - values_original) / values_original

        return returns

    def log_returns(self) -> np.ndarray:
        """
        Calculate logarithmic returns: ln(P_{t+n} / P_t)
        where n is the period length.

        Returns:
            np.ndarray: Array of log returns
        """
        # Shift values by period length
        values_shifted = self.values[self.period:]
        values_original = self.values[:-self.period]

        # Calculate log returns
        returns = np.log(values_shifted / values_original)

        return returns

    def return_statistics(self, log: bool = False) -> dict:
        """
        Calculate basic statistics for the returns.

        Args:
            log (bool): Whether to use log returns instead of simple returns

        Returns:
            dict: Dictionary containing return statistics
        """
        returns = self.log_returns() if log else self.simple_returns()

        # Annualization factor (assuming daily data)
        annual_factor = np.sqrt(252 / self.period)

        stats = {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'annualized_mean': np.mean(returns) * (252 / self.period),
            'annualized_std': np.std(returns) * annual_factor,
            'skewness': pd.Series(returns).skew(),
            'kurtosis': pd.Series(returns).kurtosis(),
            'min': np.min(returns),
            'max': np.max(returns),
            'positive_returns': np.mean(returns > 0),
            'negative_returns': np.mean(returns < 0),
        }

        return stats

    def rolling_returns(self, window: int) -> np.ndarray:
        """
        Calculate rolling returns over a specified window.

        Args:
            window (int): Window size for rolling calculation

        Returns:
            np.ndarray: Array of rolling returns
        """
        returns = self.simple_returns()
        rolling_returns = np.array([
            np.mean(returns[i:i+window])
            for i in range(len(returns) - window + 1)
        ])

        return rolling_returns

    def cumulative_returns(self, log: bool = False) -> np.ndarray:
        """
        Calculate cumulative returns.

        Args:
            log (bool): Whether to use log returns instead of simple returns

        Returns:
            np.ndarray: Array of cumulative returns
        """
        returns = self.log_returns() if log else self.simple_returns()

        if log:
            cum_returns = np.exp(np.cumsum(returns)) - 1
        else:
            cum_returns = np.cumprod(1 + returns) - 1

        return cum_returns


if __name__ == "__main__":
    # Example usage

    # Generate sample price data
    np.random.seed(42)
    initial_price = 100
    days = 252  # One trading year
    prices = initial_price * \
        np.exp(np.random.normal(0.0002, 0.01, days).cumsum())

    # Create returns calculator instance
    returns_calc = TimeSeriesReturns(prices)

    # Calculate different types of returns
    simple_rets = returns_calc.simple_returns()
    log_rets = returns_calc.log_returns()

    # Get return statistics
    stats = returns_calc.return_statistics()

    print("Return Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

    # Plot cumulative returns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    cum_returns = returns_calc.cumulative_returns()
    plt.plot(cum_returns, label='Cumulative Returns')
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # Plot rolling returns
    plt.figure(figsize=(12, 6))
    rolling_rets = returns_calc.rolling_returns(window=20)
    plt.plot(rolling_rets, label='20-day Rolling Returns')
    plt.title('Rolling Returns Over Time')
    plt.xlabel('Trading Days')
    plt.ylabel('Rolling Return')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
