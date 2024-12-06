import yfinance as yf
import pandas as pd
from typing import Optional, List, Union, Dict
from datetime import datetime, date
from functools import lru_cache


class YahooFinanceService:
    """
    A singleton service class for fetching financial data from Yahoo Finance.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YahooFinanceService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self._cache = {}

    @lru_cache(maxsize=100)
    def get_ticker_info(self, ticker: str) -> Dict:
        """
        Get basic information about a ticker.

        Args:
            ticker (str): Ticker symbol

        Returns:
            Dict: Dictionary containing ticker information
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            return ticker_obj.info
        except Exception as e:
            raise ValueError(f"Error fetching info for ticker {
                             ticker}: {str(e)}")

    def get_historical_data(self,
                            ticker: str,
                            start_date: Optional[Union[str,
                                                       datetime, date]] = None,
                            end_date: Optional[Union[str,
                                                     datetime, date]] = None,
                            interval: str = '1d') -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.

        Args:
            ticker (str): Ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data
            interval (str): Data interval ('1d', '1wk', '1mo', etc.)

        Returns:
            pd.DataFrame: DataFrame containing historical price data
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            return df
        except Exception as e:
            raise ValueError(
                f"Error fetching historical data for ticker {ticker}: {str(e)}")

    def search_tickers(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for ticker symbols matching a query.

        Args:
            query (str): Search query
            limit (int): Maximum number of results to return

        Returns:
            List[Dict]: List of matching tickers with their information
        """
        try:
            tickers = yf.Tickers(query)
            results = []
            for ticker in tickers.tickers:
                try:
                    info = ticker.info
                    results.append({
                        'symbol': info.get('symbol'),
                        'name': info.get('longName'),
                        'exchange': info.get('exchange'),
                        'industry': info.get('industry'),
                        'sector': info.get('sector')
                    })
                    if len(results) >= limit:
                        break
                except:
                    continue
            return results
        except Exception as e:
            raise ValueError(f"Error searching tickers: {str(e)}")

    def get_multiple_tickers_data(self,
                                  tickers: List[str],
                                  start_date: Optional[Union[str,
                                                             datetime, date]] = None,
                                  end_date: Optional[Union[str,
                                                           datetime, date]] = None,
                                  interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple tickers.

        Args:
            tickers (List[str]): List of ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            interval (str): Data interval ('1d', '1wk', '1mo', etc.)

        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping tickers to their historical data
        """
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.get_historical_data(
                    ticker, start_date, end_date, interval)
            except Exception as e:
                print(f"Warning: Failed to fetch data for {ticker}: {str(e)}")
        return results

    def get_market_cap(self, ticker: str) -> float:
        """
        Get the market capitalization for a ticker.

        Args:
            ticker (str): Ticker symbol

        Returns:
            float: Market capitalization
        """
        info = self.get_ticker_info(ticker)
        return info.get('marketCap', None)

    def get_financial_ratios(self, ticker: str) -> Dict:
        """
        Get key financial ratios for a ticker.

        Args:
            ticker (str): Ticker symbol

        Returns:
            Dict: Dictionary containing financial ratios
        """
        info = self.get_ticker_info(ticker)
        ratios = {
            'PE_Ratio': info.get('trailingPE'),
            'Forward_PE': info.get('forwardPE'),
            'PB_Ratio': info.get('priceToBook'),
            'Dividend_Yield': info.get('dividendYield'),
            'ROE': info.get('returnOnEquity'),
            'ROA': info.get('returnOnAssets'),
            'Profit_Margin': info.get('profitMargins')
        }
        return ratios


if __name__ == "__main__":
    # Example usage
    yf_service = YahooFinanceService()

    # Search for tickers
    search_results = yf_service.search_tickers("AAPL", limit=5)
    print("\nSearch Results:")
    for result in search_results:
        print(result)

    # Get historical data
    historical_data = yf_service.get_historical_data(
        "AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    print("\nHistorical Data Sample:")
    print(historical_data.head())

    # Get multiple tickers data
    tickers_data = yf_service.get_multiple_tickers_data(
        ["AAPL", "MSFT", "GOOGL"],
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    print("\nMultiple Tickers Data Sample:")
    for ticker, data in tickers_data.items():
        print(f"\n{ticker}:")
        print(data.head())

    # Get financial ratios
    ratios = yf_service.get_financial_ratios("AAPL")
    print("\nFinancial Ratios:")
    for ratio, value in ratios.items():
        print(f"{ratio}: {value}")

    # Plot closing prices
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    for ticker, data in tickers_data.items():
        plt.plot(data.index, data['Close'], label=ticker)
    plt.title('Closing Prices Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
