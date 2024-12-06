import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class EuropeanOption:
    """
    A class to price European options using the Black-Scholes model.

    Attributes:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate (annual)
        sigma (float): Volatility of the underlying asset (annual)
        option_type (OptionType): Type of option (call or put)
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType = OptionType.CALL

    def _calculate_d1(self) -> float:
        """Calculate d1 parameter for Black-Scholes formula."""
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def _calculate_d2(self) -> float:
        """Calculate d2 parameter for Black-Scholes formula."""
        return self._calculate_d1() - self.sigma * np.sqrt(self.T)

    def price(self) -> float:
        """
        Calculate the theoretical price of the European option.

        Returns:
            float: Option price
        """
        d1 = self._calculate_d1()
        d2 = self._calculate_d2()

        if self.option_type == OptionType.CALL:
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def delta(self) -> float:
        """
        Calculate the delta of the option (first derivative with respect to stock price).

        Returns:
            float: Option delta
        """
        d1 = self._calculate_d1()

        if self.option_type == OptionType.CALL:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def gamma(self) -> float:
        """
        Calculate the gamma of the option (second derivative with respect to stock price).

        Returns:
            float: Option gamma
        """
        d1 = self._calculate_d1()
        return norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self) -> float:
        """
        Calculate the vega of the option (derivative with respect to volatility).

        Returns:
            float: Option vega
        """
        d1 = self._calculate_d1()
        return self.S * np.sqrt(self.T) * norm.pdf(d1)

    def theta(self) -> float:
        """
        Calculate the theta of the option (derivative with respect to time).

        Returns:
            float: Option theta
        """
        d1 = self._calculate_d1()
        d2 = self._calculate_d2()

        if self.option_type == OptionType.CALL:
            return (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) -
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            return (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) +
                    self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))

    def rho(self) -> float:
        """
        Calculate the rho of the option (derivative with respect to interest rate).

        Returns:
            float: Option rho
        """
        d2 = self._calculate_d2()

        if self.option_type == OptionType.CALL:
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)


if __name__ == "__main__":
    # Example usage
    option = EuropeanOption(
        S=100,  # Current stock price
        K=100,  # Strike price
        T=1,    # Time to maturity (1 year)
        r=0.05,  # Risk-free rate (5%)
        sigma=0.2,  # Volatility (20%)
        option_type=OptionType.CALL
    )

    # Calculate option price
    price = option.price()
    print(f"Option Price: {price:.2f}")

    # Calculate Greeks
    print(f"Delta: {option.delta():.4f}")
    print(f"Gamma: {option.gamma():.4f}")
    print(f"Vega: {option.vega():.4f}")
    print(f"Theta: {option.theta():.4f}")
    print(f"Rho: {option.rho():.4f}")
