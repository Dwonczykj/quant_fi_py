from typing import Dict, List, Optional, TypedDict, Union
from datetime import date, datetime
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MarketDataPoint(BaseModel):
    """Base class for market data points with validation"""
    date: date
    value: float = Field(gt=0)

    @validator('value')
    def validate_value(cls, v: float) -> float:
        if np.isnan(v) or np.isinf(v):
            raise ValueError("Value must be a finite number")
        return v


class CreditSpreadPoint(MarketDataPoint):
    """Credit spread data point with tenor information"""
    tenor_years: float = Field(gt=0)

    @validator('tenor_years')
    def validate_tenor(cls, v: float) -> float:
        if v <= 0 or v > 50:  # Assuming max 50 year tenors
            raise ValueError("Tenor must be between 0 and 50 years")
        return v


class DiscountFactorPoint(MarketDataPoint):
    """Discount factor data point"""
    @validator('value')
    def validate_discount_factor(cls, v: float) -> float:
        if v > 1:
            raise ValueError("Discount factor must be <= 1")
        return v


class MarketDataReader:
    """Base class for reading market data from various sources"""

    def __init__(self, data_path: Union[str, Path]):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} does not exist")


class CreditSpreadReader(MarketDataReader):
    """Reads CDS spread data for reference entities"""

    def read_current_spreads(self, entity_name: str) -> List[CreditSpreadPoint]:
        """Read current term structure of CDS spreads"""
        try:
            # Assuming CSV format: date,tenor,spread
            df = pd.read_csv(self.data_path / f"{entity_name}_spreads.csv")

            return [
                CreditSpreadPoint(
                    date=pd.to_datetime(row['date']).date(),
                    tenor_years=row['tenor'],
                    value=row['spread']
                )
                for _, row in df.iterrows()
            ]
        except Exception as e:
            logger.error(f"Error reading spreads for {entity_name}: {str(e)}")
            raise

    def read_historical_spreads(
        self,
        entity_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        tenor_years: float = 5  # Default to 5Y CDS for correlation
    ) -> List[CreditSpreadPoint]:
        """Read historical spread data for correlation calculation"""
        try:
            df = pd.read_csv(self.data_path / f"{entity_name}_historical.csv")
            df['date'] = pd.to_datetime(df['date']).dt.date

            # Filter by dates if provided
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]

            # Filter by tenor
            df = df[df['tenor'] == tenor_years]

            return [
                CreditSpreadPoint(
                    date=row['date'],
                    tenor_years=tenor_years,
                    value=row['spread']
                )
                for _, row in df.iterrows()
            ]
        except Exception as e:
            logger.error(f"Error reading historical spreads for {
                         entity_name}: {str(e)}")
            raise


class DiscountCurveReader(MarketDataReader):
    """Reads SONIA discount curve data"""

    def read_curve(self, curve_date: date) -> List[DiscountFactorPoint]:
        """Read discount factors for a specific date"""
        try:
            # Assuming CSV format: date,tenor_years,discount_factor
            df = pd.read_csv(self.data_path /
                             f"sonia_{curve_date.isoformat()}.csv")

            return [
                DiscountFactorPoint(
                    date=curve_date +
                    pd.DateOffset(years=row['tenor_years']).date(),
                    value=row['discount_factor']
                )
                for _, row in df.iterrows()
            ]
        except Exception as e:
            logger.error(f"Error reading discount curve for {
                         curve_date}: {str(e)}")
            raise


class MarketDataCache:
    """Cache for market data to avoid repeated file reads"""

    def __init__(self):
        self._spread_cache: Dict[str, List[CreditSpreadPoint]] = {}
        self._historical_cache: Dict[str, List[CreditSpreadPoint]] = {}
        self._discount_cache: Dict[date, List[DiscountFactorPoint]] = {}

    def get_spreads(
        self,
        reader: CreditSpreadReader,
        entity_name: str,
        refresh: bool = False
    ) -> List[CreditSpreadPoint]:
        """Get current spreads, using cache unless refresh requested"""
        if refresh or entity_name not in self._spread_cache:
            self._spread_cache[entity_name] = reader.read_current_spreads(
                entity_name)
        return self._spread_cache[entity_name]

    def get_historical_spreads(
        self,
        reader: CreditSpreadReader,
        entity_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        tenor_years: float = 5,
        refresh: bool = False
    ) -> List[CreditSpreadPoint]:
        """Get historical spreads, using cache unless refresh requested"""
        cache_key = f"{entity_name}_{tenor_years}"
        if refresh or cache_key not in self._historical_cache:
            self._historical_cache[cache_key] = reader.read_historical_spreads(
                entity_name, start_date, end_date, tenor_years
            )
        return self._historical_cache[cache_key]

    def get_discount_factors(
        self,
        reader: DiscountCurveReader,
        curve_date: date,
        refresh: bool = False
    ) -> List[DiscountFactorPoint]:
        """Get discount factors, using cache unless refresh requested"""
        if refresh or curve_date not in self._discount_cache:
            self._discount_cache[curve_date] = reader.read_curve(curve_date)
        return self._discount_cache[curve_date]


class ReferenceDataSpreadsDict(TypedDict):
    current_spreads: List[CreditSpreadPoint]
    historical_spreads: List[CreditSpreadPoint]


class ReferenceDataPointDict(TypedDict):
    discount_factors: List[DiscountFactorPoint]
    spreads: Dict[str, ReferenceDataSpreadsDict]


class MarketDataManager:
    """High-level interface for market data access"""

    def __init__(self, data_path: Union[str, Path]):
        self.spread_reader = CreditSpreadReader(data_path)
        self.discount_reader = DiscountCurveReader(data_path)
        self.cache = MarketDataCache()

    def get_reference_data(
        self,
        entity_names: List[str],
        pricing_date: date,
        correlation_lookback_days: int = 750,  # ~3 years of trading days
        correlation_tenor: float = 5,
        use_cache: bool = True
    ) -> ReferenceDataPointDict:
        # ) -> Dict[str, Dict[str, List[Union[CreditSpreadPoint, DiscountFactorPoint]]]]:
        """Get all required market data for basket CDS pricing"""

        start_date = pricing_date - \
            pd.DateOffset(days=correlation_lookback_days)

        result: ReferenceDataPointDict = {
            "spreads": {},
            "discount_factors": []
        }
        for entity in entity_names:
            result["spreads"][entity] = {
                'current_spreads': self.cache.get_spreads(
                    self.spread_reader, entity, refresh=not use_cache
                ),
                'historical_spreads': self.cache.get_historical_spreads(
                    self.spread_reader, entity, start_date, pricing_date,
                    correlation_tenor, refresh=not use_cache
                )
            }

        # Add discount factors
        result['discount_factors'] = self.cache.get_discount_factors(
            self.discount_reader, pricing_date, refresh=not use_cache
        )

        return result


if __name__ == "__main__":
    # Example usage
    import datetime as dt

    # Initialize manager with data path
    manager = MarketDataManager("./market_data")

    # Define reference entities and pricing date
    entities = ["DEUTSCHE_BANK", "BARCLAYS",
                "JPMORGAN", "GOLDMAN_SACHS", "RBS"]
    pricing_date = dt.date(2024, 12, 10)

    # Fetch all required market data
    try:
        market_data = manager.get_reference_data(
            entity_names=entities,
            pricing_date=pricing_date,
            correlation_lookback_days=750,  # 3 years
            correlation_tenor=5  # 5Y CDS for correlation
        )

        # Example of accessing data
        for entity in entities:
            current_spreads = market_data["spreads"][entity]['current_spreads']
            print(f"\nCurrent spreads for {entity}:")
            for spread in current_spreads[:3]:  # First 3 points
                print(f"Tenor {spread.tenor_years}Y: {spread.value:.2f} bps")

            hist_spreads = market_data["spreads"][entity]['historical_spreads']
            print(f"Historical data points: {len(hist_spreads)}")

        # Example of discount factors
        discount_factors = market_data['discount_factors']
        print("\nDiscount factors:")
        for df in discount_factors[:3]:  # First 3 points
            print(f"Date {df.date}: {df.value:.4f}")

    except Exception as e:
        print(f"Error fetching market data: {str(e)}")
