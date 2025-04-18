import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")  # Retrieved from .env file


def get_stock_data(ticker: str, start: str = "2018-01-01", end: str = ""):
    file_path = f"data/{ticker}.csv"

    # Check if the data already exists locally
    if os.path.exists(file_path):
        print(f"Loading data for {ticker} from {file_path}")
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        print(f"Fetching data for {ticker} from Alpha Vantage")
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, meta = ts.get_daily(symbol=ticker, outputsize='full')
        data.sort_index(inplace=True)
        data = data.loc[start:end] if end else data.loc[start:]
        data = data.rename(
            columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume',
            }
        )

        # Save the data to a CSV file
        os.makedirs("data", exist_ok=True)
        data.to_csv(file_path)
        print(f"Data for {ticker} saved to {file_path}")


def main():
    # TODO: Replace with actual SPY tickers
    # TODO: Fetch SPY tickers dynamically if possible from a rea
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
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")


if __name__ == "__main__":
    main()
