"""Main application module for Simba Signals ML.

This module serves as the entry point for the application and orchestrates
the data processing, model training, and visualization pipeline.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_fetcher import fetch_stock_data
from src.features import add_technical_indicators
from src.labeler import create_labels
from src.model import evaluate_model, load_model, train_model
from src.utils import get_logger, setup_logging

# Initialize logging configuration
setup_logging()
logger = get_logger('main')


def main():
    """Run the main ML pipeline for stock market signal generation.

    This function orchestrates the entire pipeline including data collection,
    feature engineering, label generation, model training, and evaluation.
    Results are saved as CSV files and visualizations.
    """
    logger.info("Starting Simba Signals ML pipeline")

    # Create necessary directories
    Path("data").mkdir(exist_ok=True)

    try:
        # Step 1: Data Collection
        logger.info("Step 1: Data Collection")
        symbols = [
            'AAPL',
            'MSFT',
            'GOOGL',
            'AMZN',
            'TSLA',
            'BRK.B',
            'JPM',
            'JNJ',
            'NVDA',
        ]

        for symbol in symbols:
            data_path = f"data/{symbol}.csv"
            if not os.path.exists(data_path):
                logger.info(f"Fetching data for {symbol}")
                df = fetch_stock_data(symbol, period="5y")
                df.to_csv(data_path)
                logger.info(f"Saved data for {symbol} to {data_path}")
            else:
                logger.info(f"Data for {symbol} already exists at {data_path}")

        # Step 2: Feature Engineering
        logger.info("Step 2: Feature Engineering")
        all_data = []

        for symbol in symbols:
            logger.info(f"Processing features for {symbol}")
            df = pd.read_csv(f"data/{symbol}.csv", index_col=0, parse_dates=True)
            df = add_technical_indicators(df)
            df['Symbol'] = symbol
            all_data.append(df)

        combined_data = pd.concat(all_data)
        logger.info(f"Combined data shape: {combined_data.shape}")

        # Step 3: Label Generation
        logger.info("Step 3: Label Generation")
        labeled_data = create_labels(combined_data)
        logger.info(f"Labeled data shape: {labeled_data.shape}")

        # Step 4: Model Training & Evaluation
        logger.info("Step 4: Model Training & Evaluation")
        X_train, X_test, y_train, y_test = train_model(labeled_data)
        model = load_model()

        metrics, threshold_results = evaluate_model(model, X_test, y_test)
        logger.info(f"Model performance: {metrics}")

        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv("model_performance.csv", index=False)
        logger.info("Saved model performance metrics to model_performance.csv")

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        logger.info("Saved confusion matrix visualization to confusion_matrix.png")

        # Plot threshold sweep
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='threshold', y='f1_score', data=threshold_results)
        sns.lineplot(x='threshold', y='precision', data=threshold_results)
        sns.lineplot(x='threshold', y='recall', data=threshold_results)
        plt.title('Performance Metrics vs. Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.legend(['F1 Score', 'Precision', 'Recall'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('threshold_sweep.png')
        logger.info("Saved threshold sweep visualization to threshold_sweep.png")

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
