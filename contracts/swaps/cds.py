from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
from datetime import date
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from scipy.stats import norm, t
from dataclasses import dataclass
from enum import Enum

from services.market_data_adaptor import CreditSpreadPoint, DiscountFactorPoint, MarketDataManager


class PricingModel(str, Enum):
    GAUSSIAN_COPULA = "gaussian"
    STUDENT_T_COPULA = "student_t"


class CDSSpec(BaseModel):
    """Specification for a single name CDS"""
    reference_name: str
    start_date: date
    end_date: date
    notional: float = Field(gt=0)
    recovery_rate: float = Field(ge=0, le=1)
    payment_frequency_months: int = Field(ge=1, le=12)

    @property
    def payment_dates(self) -> List[date]:
        """Generate payment dates based on frequency"""
        dates = pd.date_range(
            self.start_date,
            self.end_date,
            freq=f"{self.payment_frequency_months}M"
        )
        return [d.date() for d in dates]


class BasketCDSSpec(BaseModel):
    """Specification for a basket CDS"""
    reference_cdss: List[CDSSpec]
    k: int = Field(gt=0)  # k-th to default
    model: PricingModel
    correlation_tenor: float = Field(ge=0)  # tenor for correlation calculation
    # degrees of freedom for Student's t
    student_t_df: Optional[int] = Field(gt=0)

    @validator('k')
    def validate_k(cls, v: int, values: Dict) -> int:
        if 'reference_cdss' in values and v > len(values['reference_cdss']):
            raise ValueError(
                "k cannot be larger than number of reference entities")
        return v


# class HazardRateCurve:
#     """Represents a hazard rate curve bootstrapped from CDS spreads"""

#     def __init__(self, spreads: List[CreditSpreadPoint], recovery_rate: float):
#         self.spreads = sorted(spreads, key=lambda x: x.tenor_years)
#         self.recovery_rate = recovery_rate
#         self._bootstrap_hazard_rates()

#     def _bootstrap_hazard_rates(self):
#         """Bootstrap hazard rates from CDS spreads"""
#         self.hazard_rates = []
#         prev_rate = 0

#         for i, point in enumerate(self.spreads):
#             if i == 0:
#                 # First point - simple calculation
#                 self.hazard_rates.append(
#                     point.value / (1 - self.recovery_rate)
#                 )
#             else:
#                 # Bootstrap using previous rates
#                 prev_survival = np.exp(-sum(self.hazard_rates)
#                                        * self.spreads[i-1].tenor_years)
#                 current_spread = point.value

#                 # Solve numerically for hazard rate
#                 def objective(h):
#                     survival = np.exp(-h * (point.tenor_years -
#                                       self.spreads[i-1].tenor_years))
#                     return current_spread - (1 - survival) * (1 - self.recovery_rate) / point.tenor_years

#                 from scipy.optimize import newton
#                 rate = newton(objective, x0=prev_rate)
#                 self.hazard_rates.append(rate)
#                 prev_rate = rate

#     def survival_probability(self, t: float) -> float:
#         """Calculate survival probability to time t"""
#         if t <= 0:
#             return 1.0

#         # Find relevant hazard rate segment
#         idx = 0
#         while idx < len(self.spreads) and self.spreads[idx].tenor_years < t:
#             idx += 1

#         if idx == 0:
#             return np.exp(-self.hazard_rates[0] * t)

#         # Piecewise calculation
#         result = 1.0
#         current_t = 0.0

#         for i in range(idx):
#             next_t = min(
#                 t, self.spreads[i+1].tenor_years if i+1 < len(self.spreads) else t)
#             segment_length = next_t - current_t
#             result *= np.exp(-self.hazard_rates[i] * segment_length)
#             current_t = next_t

#         return result


class DefaultTimeSimulator:
    """Simulates default times using copula approach"""

    def __init__(
        self,
        hazard_curves: Dict[str, HazardRateCurve],
        correlation_matrix: np.ndarray,
        model: PricingModel,
        student_t_df: Optional[int]
    ):
        self.hazard_curves = hazard_curves
        self.correlation_matrix = correlation_matrix
        self.model = model
        self.student_t_df = student_t_df or 4

        if model == PricingModel.STUDENT_T_COPULA and student_t_df is None:
            raise ValueError("Student's t degrees of freedom required")

        # Cholesky decomposition for correlation
        self.cholesky = np.linalg.cholesky(correlation_matrix)

    def simulate_default_times(
        self,
        n_scenarios: int,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Simulate default times for all reference entities"""
        if seed is not None:
            np.random.seed(seed)

        n_entities = len(self.hazard_curves)

        # Generate correlated uniform variables
        if self.model == PricingModel.GAUSSIAN_COPULA:
            # Standard normal variates
            Z = np.random.standard_normal((n_scenarios, n_entities))
            # Correlate using Cholesky
            X = Z @ self.cholesky.T
            # Transform to uniform using normal CDF
            U = norm.cdf(X)
        else:  # Student's t copula
            # Generate independent t variates
            Z = np.random.standard_t(
                df=self.student_t_df, size=(n_scenarios, n_entities))
            # Correlate
            X = Z @ self.cholesky.T
            # Transform to uniform using t CDF
            U = t.cdf(X, df=self.student_t_df)

        # Convert to default times using inverse of survival function
        default_times = {}
        for i, (name, curve) in enumerate(self.hazard_curves.items()):
            u = U[:, i]
            # Solve S(τ) = u for τ
            # Simplified
            default_times[name] = - \
                np.log(u) / np.array([curve.hazard_rates[0]])

        return default_times


class BasketCDSPricer:
    """Prices k-th to default basket CDS"""

    def __init__(self, spec: BasketCDSSpec, market_data_manager: MarketDataManager):
        self.spec = spec
        self.market_data = market_data_manager

    def price(
        self,
        pricing_date: date,
        n_scenarios: int = 10000,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """Price the basket CDS using Monte Carlo simulation"""
        # Get market data
        data = self.market_data.get_reference_data(
            entity_names=[
                cds.reference_name for cds in self.spec.reference_cdss],
            pricing_date=pricing_date,
            correlation_tenor=self.spec.correlation_tenor
        )

        # Bootstrap hazard curves
        hazard_curves = {
            cds.reference_name: HazardRateCurve(
                data["spreads"][cds.reference_name]['current_spreads'],
                cds.recovery_rate,

            )
            for cds in self.spec.reference_cdss
        }

        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(
            data['spreads'])

        # Setup default time simulator
        simulator = DefaultTimeSimulator(
            hazard_curves=hazard_curves,
            correlation_matrix=correlation_matrix,
            model=self.spec.model,
            student_t_df=self.spec.student_t_df
        )

        # Simulate default times
        default_times = simulator.simulate_default_times(n_scenarios, seed)

        # Calculate legs
        premium_leg = self._calculate_premium_leg(
            default_times, data['discount_factors'])
        protection_leg = self._calculate_protection_leg(
            default_times, data['discount_factors'])

        # Calculate fair spread
        fair_spread = protection_leg / \
            premium_leg if premium_leg > 0 else float('nan')

        return {
            'fair_spread': fair_spread,
            'premium_leg': premium_leg,
            'protection_leg': protection_leg
        }

    def _calculate_correlation_matrix(self, market_data: Dict) -> np.ndarray:
        """Calculate correlation matrix from historical spread data"""
        # Extract historical spreads and calculate returns
        spreads = {}
        for cds in self.spec.reference_cdss:
            hist_data = market_data[cds.reference_name]['historical_spreads']
            spreads[cds.reference_name] = pd.Series(
                [p.value for p in hist_data],
                index=[p.date for p in hist_data]
            )

        # Calculate log returns
        returns = pd.DataFrame({
            name: np.log(s / s.shift(1)) for name, s in spreads.items()
        }).dropna()

        # Calculate correlation matrix
        return returns.corr().values

    def _calculate_premium_leg(
        self,
        default_times: Dict[str, np.ndarray],
        discount_factors: List[DiscountFactorPoint]
    ) -> float:
        """Calculate present value of premium leg payments up to kth default"""
        n_scenarios = len(next(iter(default_times.values())))

        # Convert default times to array and sort for each scenario
        default_array = np.array([times for times in default_times.values()]).T
        sorted_defaults = np.sort(default_array, axis=1)

        # Get kth default time for each scenario
        kth_default_times = sorted_defaults[:, self.spec.k - 1]

        # Setup payment schedule
        payment_schedule = []
        for cds in self.spec.reference_cdss:
            payment_schedule.extend(cds.payment_dates)
        payment_schedule = sorted(
            list(set(payment_schedule)))  # Remove duplicates

        # Convert to years from pricing date
        payment_years = [(d - payment_schedule[0]).days /
                         365 for d in payment_schedule]

        # Interpolate discount factors
        df_dates = [p.date for p in discount_factors]
        df_values = [p.value for p in discount_factors]
        payment_dfs = np.interp(payment_years,
                                [(d - df_dates[0]).days / 365 for d in df_dates],
                                df_values)

        # Calculate premium payments
        total_pv = 0.0
        year_fraction = payment_years[1] - \
            payment_years[0]  # Assume regular payments

        for i, payment_time in enumerate(payment_years):
            # Count scenarios where payment is made (before kth default)
            valid_scenarios = np.sum(kth_default_times > payment_time)

            # Add PV of payment
            total_pv += payment_dfs[i] * year_fraction * valid_scenarios

        # Return average across scenarios
        return total_pv / n_scenarios

    def _calculate_protection_leg(
        self,
        default_times: Dict[str, np.ndarray],
        discount_factors: List[DiscountFactorPoint]
    ) -> float:
        """Calculate present value of protection payment at kth default"""
        n_scenarios = len(next(iter(default_times.values())))

        # Get kth default time and corresponding entity for each scenario
        default_array = np.array([times for times in default_times.values()]).T
        sorted_indices = np.argsort(default_array, axis=1)
        kth_indices = sorted_indices[:, self.spec.k - 1]
        kth_default_times = np.take_along_axis(default_array,
                                               np.expand_dims(
                                                   sorted_indices[:, self.spec.k - 1], 1),
                                               axis=1).squeeze()

        # Get recovery rates for defaulting entities
        recovery_rates = np.array(
            [cds.recovery_rate for cds in self.spec.reference_cdss])
        kth_recoveries = recovery_rates[kth_indices]

        # Interpolate discount factors for default times
        df_dates = [p.date for p in discount_factors]
        df_values = [p.value for p in discount_factors]
        df_years = [(d - df_dates[0]).days / 365 for d in df_dates]

        default_dfs = np.interp(kth_default_times,
                                df_years,
                                df_values,
                                left=df_values[0],
                                right=df_values[-1])

        # Calculate protection payments
        protection_payments = default_dfs * (1 - kth_recoveries)

        # Remove scenarios where no kth default occurs within horizon
        max_tenor = max(df_years)
        valid_scenarios = kth_default_times <= max_tenor
        protection_payments = protection_payments[valid_scenarios]

        if len(protection_payments) == 0:
            return 0.0

        # Return average across valid scenarios
        return float(np.mean(protection_payments))


class HazardRateCurve:
    """Represents a hazard rate curve bootstrapped from CDS spreads"""

    def __init__(self, spreads: List[CreditSpreadPoint], recovery_rate: float):
        self.spreads = sorted(spreads, key=lambda x: x.tenor_years)
        self.recovery_rate = recovery_rate
        self._bootstrap_hazard_rates()

    def _bootstrap_hazard_rates(self):
        """Bootstrap hazard rates from CDS spreads using iterative method"""
        self.hazard_rates = []
        tenors = [s.tenor_years for s in self.spreads]

        for i, point in enumerate(self.spreads):
            if i == 0:
                # First point - simple approximation
                self.hazard_rates.append(
                    point.value / ((1 - self.recovery_rate)
                                   * point.tenor_years)
                )
            else:
                # Solve iteratively for hazard rate
                def objective(h):
                    # Previous hazard rates
                    prev_rates = self.hazard_rates.copy()
                    prev_rates.append(h)

                    # Calculate survival probabilities
                    total_prob = 0
                    current_t = 0

                    for j in range(len(prev_rates)):
                        next_t = tenors[j]
                        dt = next_t - current_t

                        # Probability of survival to current time * default in interval
                        survival = np.exp(-sum(r * t for r, t in zip(prev_rates[:j],
                                                                     [tenors[k] - (0 if k == 0 else tenors[k-1])
                                                                      for k in range(j)])))
                        default_prob = (1 - np.exp(-prev_rates[j] * dt))

                        total_prob += survival * default_prob
                        current_t = next_t

                    # Expected loss should match spread
                    expected_loss = total_prob * (1 - self.recovery_rate)
                    return expected_loss - point.value * point.tenor_years

                # Solve using Newton's method with previous rate as initial guess
                from scipy.optimize import newton
                rate = newton(objective, x0=self.hazard_rates[-1])
                self.hazard_rates.append(rate)

    def default_probability(self, t: float) -> float:
        """Calculate cumulative default probability to time t"""
        return 1 - self.survival_probability(t)

    def survival_probability(self, t: float) -> float:
        """Calculate survival probability to time t using piecewise hazard rates"""
        if t <= 0:
            return 1.0

        total_hazard = 0.0
        current_t = 0.0

        for i, rate in enumerate(self.hazard_rates):
            next_t = self.spreads[i].tenor_years

            if t <= next_t:
                # Add final partial segment
                total_hazard += rate * (t - current_t)
                break
            else:
                # Add full segment
                total_hazard += rate * (next_t - current_t)
                current_t = next_t

            if i == len(self.hazard_rates) - 1 and t > next_t:
                # Beyond last point - extrapolate using last hazard rate
                total_hazard += rate * (t - next_t)

        return np.exp(-total_hazard)


if __name__ == "__main__":
    from datetime import date, timedelta

    # Setup pricing date and market data
    pricing_date = date(2024, 12, 10)
    market_data_manager = MarketDataManager("./market_data")

    # Create CDS specifications
    reference_cdss = [
        CDSSpec(
            reference_name=name,
            start_date=pricing_date,
            end_date=pricing_date + timedelta(days=365*5),  # 5Y CDS
            notional=1_000_000,  # 1MM
            recovery_rate=0.4,
            payment_frequency_months=3  # Quarterly
        )
        for name in ["DEUTSCHE_BANK", "BARCLAYS", "JPMORGAN", "GOLDMAN_SACHS", "RBS"]
    ]

    # Create basket CDS specification
    basket_spec = BasketCDSSpec(
        reference_cdss=reference_cdss,
        k=3,  # 3rd to default
        model=PricingModel.GAUSSIAN_COPULA,
        correlation_tenor=5.0,  # 5Y for correlation calculation
        student_t_df=4
    )

    # Create pricer and calculate price
    pricer = BasketCDSPricer(basket_spec, market_data_manager)

    try:
        result = pricer.price(
            pricing_date=pricing_date,
            n_scenarios=10000,
            seed=42
        )

        print("\nBasket CDS Pricing Results:")
        print(f"Fair spread: {result['fair_spread']*10000:.1f} bps")
        print(f"Premium leg PV: ${result['premium_leg']:,.2f}")
        print(f"Protection leg PV: ${result['protection_leg']:,.2f}")

    except Exception as e:
        print(f"Error in pricing: {str(e)}")
