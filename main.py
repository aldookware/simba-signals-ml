import os
import pandas as pd
from src.data_fetcher import get_stock_data
from src.features import add_technical_indicators
from src.labeler import label_signals
from src.model import train_model


def process_ticker(ticker):
    """
    Process a single ticker: fetch data, add indicators, label signals, and train
    the model.
    """
    # Check if the CSV file exists
    file_path = f"data/{ticker}.csv"
    if not os.path.exists(file_path):
        print(f"Fetching data for {ticker}...")
        df = get_stock_data(ticker)
    else:
        print(f"Loading data for {ticker} from {file_path}...")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Add technical indicators and label signals
    df = add_technical_indicators(df)
    df = label_signals(df)

    # Train the model
    train_model(df)
    print(f"Model training complete for {ticker} and saved as simba_model.pkl")


if __name__ == "__main__":
    # Specify a single ticker or a list of tickers
    tickers = ["AAPL", "MSFT", "GOOGL"]  # Replace with your list of tickers

    for ticker in tickers:
        try:
            process_ticker(ticker)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
