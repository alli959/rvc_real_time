"""Logging configuration for Voice Engine."""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        name: Logger name (defaults to root logger)
    
    Returns:
        Configured logger instance
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Silence noisy third-party loggers
def silence_noisy_loggers():
    """Reduce log noise from third-party libraries."""
    noisy_loggers = [
        "urllib3",
        "httpx",
        "httpcore",
        "fairseq",
        "numba",
        "matplotlib",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
