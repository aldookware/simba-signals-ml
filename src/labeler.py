"""Labeling module for Simba Signals ML.

This module provides functions to create target labels for stock price data
based on future price movements.
"""

from .utils import get_logger

# Set up logger
logger = get_logger('labeler')


def label_signals(df, lookahead=5, up_thresh=0.02, down_thresh=-0.02):
    """Create target labels for stock price data based on future returns.

    Args:
        df (pd.DataFrame): DataFrame containing stock price data
        lookahead (int, optional): Number of days to look ahead. Defaults to 5.
        up_thresh (float, optional): Threshold for Buy signal. Defaults to 0.02.
        down_thresh (float, optional): Threshold for Sell signal. Defaults to -0.02.

    Returns:
        pd.DataFrame: DataFrame with 'Signal' column added
    """
    logger.info(
        f"Creating signals with lookahead={lookahead}, up_thresh={up_thresh}, "
        f"down_thresh={down_thresh}"
    )

    # Calculate future returns
    future_returns = df["Close"].shift(-lookahead) / df["Close"] - 1

    # Create signals based on thresholds
    df["Signal"] = future_returns.apply(
        lambda x: "Buy" if x > up_thresh else ("Sell" if x < down_thresh else "Neutral")
    )

    # Remove rows with NaN values
    initial_rows = len(df)
    df.dropna(inplace=True)
    dropped_rows = initial_rows - len(df)

    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with NaN values")

    # Log distribution of signals
    signal_counts = df["Signal"].value_counts()
    logger.info(f"Signal distribution: {signal_counts.to_dict()}")

    return df
