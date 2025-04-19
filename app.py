"""Web application for Simba Signals ML model deployment.

This module provides a web interface for the stock market signals prediction model,
allowing users to request predictions for specific stocks.
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from src.features import add_technical_indicators

try:
    from flask import Flask, jsonify, request
except ImportError:
    print("Flask not installed, API functionality will not be available")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the best model at startup
MODEL_PATH = "simba_model_best.pkl"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "simba_model.pkl"  # Fallback to default model

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None


@app.route("/health", methods=["GET"])
def health_check():
    """Check the health of the application and model.

    Returns:
        flask.Response: JSON response indicating health status.
    """
    if model is not None:
        return jsonify({"status": "healthy", "model": MODEL_PATH}), 200
    else:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to make predictions on stock data.

    Expects JSON input with the following format:
    {
        "data": {
            "Date": ["2022-01-01", "2022-01-02", ...],
            "Open": [100.0, 101.0, ...],
            "High": [102.0, 103.0, ...],
            "Low": [99.0, 100.0, ...],
            "Close": [101.0, 102.0, ...],
            "Volume": [1000000, 1100000, ...],
            "Adj Close": [101.0, 102.0, ...],
            "Dividend": [0.0, 0.0, ...],
            "Split Coefficient": [1.0, 1.0, ...]
        }
    }

    Returns:
        flask.Response: JSON response with predictions and probabilities.
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Get data from request
        data = request.json.get("data")
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Set date as index if present
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

        # Add technical indicators
        try:
            df = add_technical_indicators(df)
        except Exception as e:
            return (
                jsonify({"error": f"Error adding technical indicators: {str(e)}"}),
                400,
            )

        # Make predictions
        X = df.drop(columns=["Signal"], errors='ignore')

        # Check if all required features are present
        missing_features = [
            feature for feature in model.feature_names_in_ if feature not in X.columns
        ]

        if missing_features:
            return (
                jsonify({"error": f"Missing required features: {missing_features}"}),
                400,
            )

        # Ensure correct feature order
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

        # Generate predictions
        pred_proba = model.predict_proba(X)
        predictions = model.classes_[np.argmax(pred_proba, axis=1)]

        # Format probabilities
        proba_dict = {}
        for i, class_name in enumerate(model.classes_):
            proba_dict[class_name] = pred_proba[:, i].tolist()

        return (
            jsonify({"predictions": predictions.tolist(), "probabilities": proba_dict}),
            200,
        )

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/fetch_and_predict", methods=["GET"])
def fetch_and_predict():
    """Fetch recent data for a ticker symbol and make predictions.

    Expects query parameters: ticker, start_date, end_date

    Example: /fetch_and_predict?ticker=AAPL&start_date=2023-01-01&end_date=2023-12-31

    Returns:
        flask.Response: JSON response with predictions and probabilities.
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Get parameters from request
        ticker_symbol = request.args.get("ticker", "AAPL")
        start_date = request.args.get("start_date", "2023-01-01")
        end_date = request.args.get("end_date", None)

        logger.info(
            f"Fetching data for {ticker_symbol} from {start_date} to {end_date}"
        )

        # Fetch data using yfinance
        ticker = yf.Ticker(ticker_symbol)
        stock_data = ticker.history(
            start=start_date, end=end_date, auto_adjust=False, actions=True
        )

        if stock_data.empty:
            return jsonify({"error": f"No data available for {ticker_symbol}"}), 400

        # Rename columns to match expected format
        stock_data.rename(
            columns={"Dividends": "Dividend", "Stock Splits": "Split Coefficient"},
            inplace=True,
        )

        # Add technical indicators
        df = add_technical_indicators(stock_data)

        # Make predictions
        X = df.drop(columns=["Signal"], errors='ignore')

        # Check if all required features are present
        missing_features = [
            feature for feature in model.feature_names_in_ if feature not in X.columns
        ]

        if missing_features:
            return (
                jsonify({"error": f"Missing required features: {missing_features}"}),
                400,
            )

        # Ensure correct feature order
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

        # Generate predictions
        pred_proba = model.predict_proba(X)
        predictions = model.classes_[np.argmax(pred_proba, axis=1)]

        # Get dates as strings for JSON output
        dates = X.index.strftime('%Y-%m-%d').tolist()

        # Format probabilities
        proba_dict = {}
        for i, class_name in enumerate(model.classes_):
            proba_dict[class_name] = pred_proba[:, i].tolist()

        return (
            jsonify(
                {
                    "ticker": ticker_symbol,
                    "dates": dates,
                    "predictions": predictions.tolist(),
                    "probabilities": proba_dict,
                    "close_prices": stock_data["Close"].tolist(),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error in fetch_and_predict: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Example of directly fetching data (for testing)
    ticker = yf.Ticker("AAPL")
    data = ticker.history(
        start="2023-01-01", end="2024-01-01", auto_adjust=False, actions=True
    )
    logger.info(f"Example data fetch completed with {len(data)} rows")

    # Run the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
