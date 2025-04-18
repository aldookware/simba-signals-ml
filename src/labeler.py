def label_signals(df, lookahead=5, up_thresh=0.02, down_thresh=-0.02):
    future_returns = df["Close"].shift(-lookahead) / df["Close"] - 1
    df["Signal"] = future_returns.apply(
        lambda x: "Buy" if x > up_thresh else ("Sell" if x < down_thresh else "Neutral")
    )
    df.dropna(inplace=True)
    return df
