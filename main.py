import os
import pandas as pd
from src.data_fetcher import get_stock_data
from src.features import add_technical_indicators
from src.labeler import label_signals
from src.model import train_model, threshold_sweep
import joblib


def process_ticker(
    ticker, save_model=True, model_name=None, thresholds=None, default_thresh=0.7
):
    """
    Process a single ticker: fetch data, add indicators, label signals, and train
    the model.

    Args:
        ticker (str): Stock ticker symbol
        save_model (bool): Whether to save the model
        model_name (str, optional): Custom name for model file

    Returns:
        tuple: (model, f1_score, X_test, y_test) model, performance, and test data
    """
    try:
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

        # Train the model and get performance metrics
        best_model, X_test, y_test = train_model(
            df,
            default_thresh=default_thresh,
            class_threshs=thresholds,
            save_model=save_model,
        )  # Capture all the return values

        if save_model:
            model_filename = model_name if model_name else f"simba_model_{ticker}.pkl"
            joblib.dump(best_model, model_filename)
            print(f"Model saved as {model_filename}")

        # Get the model's test F1 score
        # This will be stored as an attribute of the model during training
        f1 = getattr(best_model, 'test_f1_score', 0)

        # Return the model, F1 score, and test data
        return best_model, f1, X_test, y_test

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None, 0, None, None


def train_best_model(tickers, thresholds=None, save_all=False):
    """
    Train models for all tickers and select the best one based on F1 score.

    Args:
        tickers (list): List of stock ticker symbols
        save_all (bool): Whether to save all individual models

    Returns:
        tuple: Best performing model, test features, and test labels
    """
    results = {}
    best_score = 0
    best_ticker = None
    best_model = None
    best_X_test = None
    best_y_test = None

    for ticker in tickers:
        print(f"\n{'='*50}\nProcessing {ticker}\n{'='*50}")
        model_result = process_ticker(
            ticker, save_model=save_all, thresholds=thresholds
        )
        if model_result is not None:
            model, score, X_test, y_test = model_result
            results[ticker] = score
            if score > best_score:
                best_score = score
                best_ticker = ticker
                best_model = model
                best_X_test = X_test
                best_y_test = y_test

    if best_model is not None:
        print(f"\nBest model is from {best_ticker} with F1 score of {best_score:.4f}")
        joblib.dump(best_model, "simba_model_best.pkl")
        print("Best model saved as simba_model_best.pkl")

        # Save performance summary
        perf_df = pd.DataFrame.from_dict(results, orient='index', columns=['F1 Score'])
        perf_df.index.name = 'Ticker'
        perf_df.sort_values('F1 Score', ascending=False, inplace=True)
        perf_df.to_csv('model_performance.csv')
        print("Performance summary saved as model_performance.csv")

        return best_model, best_X_test, best_y_test
    else:
        print("No successful models were trained")
        return None, None, None


if __name__ == "__main__":
    # List of tickers to process
    tickers = ["AAPL"]

    # Uncomment to process a single ticker
    # process_ticker("AAPL", save_model=True)
    thresholds = {'Buy': 0.55, 'Sell': 0.55}

    # Train models for all tickers and select the best one
    best_model, X_test, y_test = train_best_model(
        tickers, thresholds=thresholds, save_all=False
    )
    threshold_sweep(best_model, X_test, y_test)
