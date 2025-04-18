import pandas_ta as ta


def add_technical_indicators(df):
    df["RSI"] = ta.rsi(df["Close"], length=14)
    df.ta.macd(close='Close', append=True)
    df.ta.bbands(length=20, append=True)
    df["SMA50"] = ta.sma(df["Close"], length=50)

    df["SMA200"] = ta.sma(df["Close"], length=200)
    df.dropna(inplace=True)
    return df
