import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime, date


@dataclass
class Bond:
    """
    A class to represent and price a bond.

    Attributes:
        face_value (float): The par/face value of the bond
        coupon_rate (float): Annual coupon rate (as a decimal)
        years_to_maturity (float): Time to maturity in years
        payments_per_year (int): Number of coupon payments per year
        market_rate (float): Current market interest rate (as a decimal)
    """

    face_value: float
    coupon_rate: float
    years_to_maturity: float
    market_rate: float
    payments_per_year: int = 2  # Semi-annual payments by default

    def __post_init__(self):
        if self.market_rate is None:
            self.market_rate = self.coupon_rate

    @property
    def coupon_payment(self) -> float:
        """Calculate the coupon payment amount."""
        return (self.face_value * self.coupon_rate) / self.payments_per_year

    def price(self, market_rate: Optional[float] = None) -> float:
        """
        Calculate the present value (price) of the bond.

        Args:
            market_rate (float, optional): Market interest rate to use for discounting.
                                         If None, uses the bond's market_rate.

        Returns:
            float: Present value of the bond
        """
        if market_rate is None:
            market_rate = self.market_rate

        # Convert annual rate to per-period rate
        period_rate = market_rate / self.payments_per_year

        # Total number of payments
        n_payments = int(self.years_to_maturity * self.payments_per_year)

        # Calculate present value of coupon payments
        coupon_pv = self.coupon_payment * \
            (1 - (1 + period_rate)**-n_payments) / period_rate

        # Calculate present value of face value
        face_pv = self.face_value / (1 + period_rate)**n_payments

        return coupon_pv + face_pv

    def yield_to_maturity(self, bond_price: float, precision: float = 0.0001) -> float:
        """
        Calculate the yield to maturity using numerical method.

        Args:
            bond_price (float): Current market price of the bond
            precision (float): Desired precision for the calculation

        Returns:
            float: Yield to maturity (as a decimal)
        """
        # Initial guesses for rates
        rate_low = 0.0
        rate_high = 1.0

        while True:
            rate_mid = (rate_low + rate_high) / 2
            price_mid = self.price(rate_mid)

            if abs(price_mid - bond_price) < precision:
                return rate_mid

            if price_mid > bond_price:
                rate_low = rate_mid
            else:
                rate_high = rate_mid

    def duration(self) -> float:
        """
        Calculate the Macaulay duration of the bond.

        Returns:
            float: Duration in years
        """
        period_rate = self.market_rate / self.payments_per_year
        n_payments = int(self.years_to_maturity * self.payments_per_year)

        duration = 0
        for t in range(1, n_payments + 1):
            # PV of each cash flow
            if t < n_payments:
                cf_pv = self.coupon_payment / (1 + period_rate)**t
            else:
                cf_pv = (self.coupon_payment + self.face_value) / \
                    (1 + period_rate)**t

            # Weight by time
            duration += t * cf_pv

        return duration / (self.price() * self.payments_per_year)

    def modified_duration(self) -> float:
        """
        Calculate the modified duration of the bond.

        Returns:
            float: Modified duration
        """
        return self.duration() / (1 + self.market_rate / self.payments_per_year)

    def convexity(self) -> float:
        """
        Calculate the convexity of the bond.

        Returns:
            float: Convexity
        """
        period_rate = self.market_rate / self.payments_per_year
        n_payments = int(self.years_to_maturity * self.payments_per_year)

        convexity = 0
        for t in range(1, n_payments + 1):
            # PV of each cash flow
            if t < n_payments:
                cf_pv = self.coupon_payment / (1 + period_rate)**t
            else:
                cf_pv = (self.coupon_payment + self.face_value) / \
                    (1 + period_rate)**t

            # Weight by time squared
            convexity += t * (t + 1) * cf_pv

        convexity = convexity / \
            (self.price() * (1 + period_rate)**2 * self.payments_per_year**2)
        return convexity


if __name__ == "__main__":
    # Example usage
    bond = Bond(
        face_value=1000,        # $1000 par value
        coupon_rate=0.05,       # 5% annual coupon rate
        years_to_maturity=10,   # 10-year bond
        payments_per_year=2,    # Semi-annual payments
        market_rate=0.06        # 6% market rate
    )

    # Calculate bond price
    price = bond.price()
    print(f"Bond Price: ${price:.2f}")

    # Calculate yield to maturity for a given price
    ytm = bond.yield_to_maturity(980)
    print(f"Yield to Maturity: {ytm:.4%}")

    # Calculate duration measures
    duration = bond.duration()
    mod_duration = bond.modified_duration()
    print(f"Duration: {duration:.2f} years")
    print(f"Modified Duration: {mod_duration:.2f}")

    # Calculate convexity
    convexity = bond.convexity()
    print(f"Convexity: {convexity:.4f}")
