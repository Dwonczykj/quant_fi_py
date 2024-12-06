import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Union
from datetime import datetime, date


@dataclass
class InterestRateSwap:
    """
    A class to represent and price an interest rate swap.

    Attributes:
        notional (float): The notional amount of the swap
        fixed_rate (float): The fixed rate of the swap (as a decimal)
        years (float): Total years of the swap
        payments_per_year (int): Number of payments per year
        floating_rate (float): Current floating rate (as a decimal)
        discount_curve (Union[float, List[float]]): Either a flat rate or list of discount factors
    """

    notional: float
    fixed_rate: float
    years: float
    payments_per_year: int = 2
    floating_rate: float = None
    discount_curve: Union[float, List[float]] = None

    def __post_init__(self):
        if self.floating_rate is None:
            self.floating_rate = self.fixed_rate
        if self.discount_curve is None:
            self.discount_curve = self.floating_rate

    @property
    def number_of_payments(self) -> int:
        """Calculate total number of payments."""
        return int(self.years * self.payments_per_year)

    def get_discount_factor(self, period: int) -> float:
        """
        Get discount factor for a specific period.

        Args:
            period (int): The payment period number

        Returns:
            float: Discount factor for that period
        """
        if isinstance(self.discount_curve, (int, float)):
            # If discount_curve is a single rate, calculate discount factor
            rate = self.discount_curve / self.payments_per_year
            return 1 / (1 + rate) ** period
        else:
            # If discount_curve is a list of discount factors
            return self.discount_curve[period - 1]

    def fixed_leg_payment(self) -> float:
        """Calculate the fixed payment amount."""
        return (self.notional * self.fixed_rate) / self.payments_per_year

    def floating_leg_payment(self) -> float:
        """Calculate the floating payment amount."""
        return (self.notional * self.floating_rate) / self.payments_per_year

    def fixed_leg_value(self) -> float:
        """
        Calculate the present value of the fixed leg.

        Returns:
            float: Present value of fixed payments
        """
        fixed_payment = self.fixed_leg_payment()
        pv = 0

        for period in range(1, self.number_of_payments + 1):
            discount_factor = self.get_discount_factor(period)
            pv += fixed_payment * discount_factor

        return pv

    def floating_leg_value(self) -> float:
        """
        Calculate the present value of the floating leg.

        Returns:
            float: Present value of floating payments
        """
        if isinstance(self.discount_curve, (int, float)):
            # For flat rate curve, floating leg PV equals notional
            return self.notional
        else:
            floating_payment = self.floating_leg_payment()
            pv = 0

            for period in range(1, self.number_of_payments + 1):
                discount_factor = self.get_discount_factor(period)
                pv += floating_payment * discount_factor

            return pv

    def value(self) -> float:
        """
        Calculate the present value of the swap from fixed rate payer's perspective.

        Returns:
            float: Swap value
        """
        return self.floating_leg_value() - self.fixed_leg_value()

    def par_rate(self) -> float:
        """
        Calculate the par swap rate that makes the swap value zero.

        Returns:
            float: Par swap rate
        """
        sum_df = 0
        for period in range(1, self.number_of_payments + 1):
            sum_df += self.get_discount_factor(period)

        return (self.payments_per_year * (1 - self.get_discount_factor(self.number_of_payments))) / sum_df

    def dv01(self) -> float:
        """
        Calculate the DV01 (dollar value of 1bp change in rates).

        Returns:
            float: DV01 value
        """
        bump = 0.0001  # 1 basis point
        original_value = self.value()

        # Bump up the fixed rate
        self.fixed_rate += bump
        bumped_value = self.value()
        self.fixed_rate -= bump  # Reset the rate

        return (bumped_value - original_value) / bump * 0.01  # Scale to 1bp


if __name__ == "__main__":
    # Example usage
    swap = InterestRateSwap(
        notional=1000000,      # $1M notional
        fixed_rate=0.05,       # 5% fixed rate
        years=5,               # 5-year swap
        payments_per_year=2,   # Semi-annual payments
        floating_rate=0.045,   # 4.5% floating rate
        discount_curve=0.05    # 5% discount rate (flat curve)
    )

    # Calculate swap value
    value = swap.value()
    print(f"Swap Value: ${value:,.2f}")

    # Calculate par swap rate
    par_rate = swap.par_rate()
    print(f"Par Swap Rate: {par_rate:.4%}")

    # Calculate DV01
    dv01 = swap.dv01()
    print(f"DV01: ${dv01:,.2f}")

    # Calculate individual leg values
    fixed_value = swap.fixed_leg_value()
    floating_value = swap.floating_leg_value()
    print(f"Fixed Leg Value: ${fixed_value:,.2f}")
    print(f"Floating Leg Value: ${floating_value:,.2f}")
