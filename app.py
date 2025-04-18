import yfinance as yf
import logging

logging.basicConfig(level=logging.DEBUG)

ticker = yf.Ticker("AAPL")
data = ticker.history(
    start="2023-01-01", end="2024-01-01", auto_adjust=False, actions=False
)
print(data.head())
