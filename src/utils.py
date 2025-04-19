"""Utility functions for the Simba Signals ML application.

This module provides logging setup and configuration functions.
"""

import logging
import logging.config
from pathlib import Path


def setup_logging():
    """Set up logging configuration from the logging.ini file.

    Creates the logs directory if it doesn't exist.

    Returns:
        bool: True if configuration was successfully loaded, False otherwise
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)

    # Load configuration file
    config_path = Path(__file__).parent.parent / 'logging.ini'

    if config_path.exists():
        try:
            logging.config.fileConfig(config_path, disable_existing_loggers=False)
            logging.info(f"Logging configuration loaded from {config_path}")
            return True
        except Exception as e:
            print(f"Error loading logging configuration: {e}")
            return False
    else:
        print(f"Logging configuration file not found at {config_path}")
        return False


def get_logger(name):
    """Get a logger with the specified name.

    Args:
        name (str): Name of the logger

    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(name)
