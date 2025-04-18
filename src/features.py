import pandas_ta as ta


def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame.
    """
    # Replace incorrect RSIIndicator usage with pandas_ta RSI function
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # Add other technical indicators as needed
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['SMA_200'] = ta.sma(df['Close'], length=200)

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df["Close"])
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df
