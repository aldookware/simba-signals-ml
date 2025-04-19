"""Feature engineering module for Simba Signals ML.

This module contains functions for calculating technical indicators
and other features used for stock market prediction.
"""

import numpy as np
import pandas_ta as ta

from .utils import get_logger

# Set up logger
logger = get_logger('features')


def add_technical_indicators(df):
    """Add technical indicators to the stock price DataFrame.

    Parameters:
    df (pd.DataFrame): A DataFrame containing stock price data with at least a
        'Close' column.

    Returns:
    pd.DataFrame: The input DataFrame with additional columns for technical
        indicators.

    Technical Indicators:
    - RSI: Relative Strength Index, measures the speed and change of price movements.
    - SMA_50: Simple Moving Average over 50 periods, used to identify trends.
    - SMA_200: Simple Moving Average over 200 periods, used for long-term trends.
    - MACD: Moving Average Convergence Divergence, shows the relationship between
      two EMAs.
    - MACD_signal: Signal line for MACD, a 9-period EMA of the MACD line.
    - MACD_hist: Histogram showing the difference between MACD and MACD_signal.
    - OBV: On-Balance Volume, measures buying and selling pressure.
    - MFI: Money Flow Index, a volume-weighted RSI.
    - ATR: Average True Range, measures market volatility.
    - BB_upper: Upper Bollinger Band, indicates overbought conditions.
    - BB_lower: Lower Bollinger Band, indicates oversold conditions.
    - BB_middle: Middle Bollinger Band, a simple moving average.
    - Daily_Return: Percentage change in closing price from the previous day.
    - Log_Return: Logarithmic return of the closing price.
    - Rolling_Return_5: Percentage change in closing price over 5 days.
    - Day_of_Week: Day of the week as an integer (0=Monday, 6=Sunday).
    - Month: Month of the year as an integer (1=January, 12=December).
    - Is_Month_End: Indicates if the date is the last trading day of the month.
    - Had_Dividend: Indicates if a dividend was issued on the day.
    - Had_Split: Indicates if a stock split occurred on the day.
    """
    logger.info(f"Adding technical indicators to DataFrame with shape {df.shape}")

    try:
        # Replace incorrect MACD usage with pandas_ta MACD function
        logger.debug("Calculating MACD indicators")
        macd = ta.macd(df['Close'])
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['MACD_hist'] = macd['MACDh_12_26_9']

        # Add other technical indicators as needed
        logger.debug("Calculating momentum indicators (RSI)")
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)

        # Volume-based indicators
        logger.debug("Calculating volume-based indicators (OBV, MFI)")
        df['OBV'] = ta.obv(df['Close'], df['Volume'])

        # Fix for MFI dtype incompatibility warning
        mfi_values = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
        df['MFI'] = mfi_values.astype(
            float
        )  # Explicit conversion to float to fix dtype warning

        df['AD'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])

        # Volatility indicators
        logger.debug("Calculating volatility indicators (ATR, Bollinger Bands)")
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        bb = ta.bbands(df['Close'], length=20)
        df['BB_upper'] = bb['BBU_20_2.0']
        df['BB_lower'] = bb['BBL_20_2.0']
        df['BB_middle'] = bb['BBM_20_2.0']

        # Returns-based features
        logger.debug("Calculating returns-based features")
        df['Daily_Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Rolling_Return_5'] = df['Close'].pct_change(periods=5)

        # Calendar features
        logger.debug("Calculating calendar features")
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Is_Month_End'] = df.index.is_month_end.astype(int)

        # Corporate action flags
        logger.debug("Calculating corporate action flags")
        df['Had_Dividend'] = (df['Dividend'] > 0).astype(int)
        # Split line to avoid E501 (line too long)
        df['Had_Split'] = (df['Split Coefficient'] != 1.0).astype(int)

        initial_rows = len(df)
        df.dropna(inplace=True)
        dropped_rows = initial_rows - len(df)

        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with NaN values")
        logger.info(
            f"Technical indicators added successfully. "
            f"Final DataFrame shape: {df.shape}"
        )

        return df

    except Exception as e:
        logger.error(f"Error adding technical indicators: {str(e)}", exc_info=True)
        raise
