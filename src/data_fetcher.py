"""Data fetching module for Simba Signals ML.

This module provides functions to fetch stock data from Alpha Vantage API
and process it for analysis.
"""

import os
from datetime import datetime, timedelta

from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv

from .utils import get_logger

# Set up logger
logger = get_logger('data_fetcher')

# Load environment variables from .env file
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")  # Retrieved from .env file


def get_stock_data(ticker: str, start: str = None, end: str = None, save_csv=True):
    """Fetch stock data from Alpha Vantage API.

    Args:
        ticker (str): Stock ticker symbol
        start (str, optional): Start date in YYYY-MM-DD format. Defaults to 5 years ago.
        end (str, optional): End date in YYYY-MM-DD format. Defaults to today.
        save_csv (bool, optional): Whether to save data to CSV. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing stock price data
    """
    logger.info(f"Fetching stock data for {ticker}")
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, meta = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
    data.sort_index(inplace=True)

    if not start:
        start_date = datetime.today() - timedelta(days=5 * 365)
        start = start_date.strftime('%Y-%m-%d')

    if end:
        data = data.loc[start:end]
    else:
        data = data.loc[start:]

    data = data.rename(
        columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. adjusted close': 'Adj Close',
            '6. volume': 'Volume',
            '7. dividend amount': 'Dividend',
            '8. split coefficient': 'Split Coefficient',
        }
    )

    if save_csv:
        os.makedirs("data", exist_ok=True)
        data.to_csv(
            f"data/{ticker}.csv", mode='w', index=True
        )  # Overwrite existing file
        logger.info(f"Saved data for {ticker} to data/{ticker}.csv")

    logger.info(f"Retrieved {len(data)} rows of data for {ticker}")
    return data


def main():
    """Fetch stock data for a predefined list of tickers."""
    # TODO: Replace with actual SPY tickers
    # TODO: Fetch SPY tickers dynamically if possible from a real source
    spy_tickers = [
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "FB",
        "TSLA",
        "BRK.B",
        "NVDA",
        "JPM",
        "JNJ",
    ]  # Add all SPY tickers here

    for ticker in spy_tickers:
        try:
            get_stock_data(ticker, start="2018-01-01")
            logger.info(f"Successfully fetched data for {ticker}")
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")


if __name__ == "__main__":
    main()
