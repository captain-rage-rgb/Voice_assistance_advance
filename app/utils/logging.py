"""
Central logging configuration for the application.
Provides structured, consistent console and optional file logging.
"""

import logging
import sys


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configures and returns the root application logger.

    Args:
        log_level: The logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        The configured root logger instance.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Standard format containing timestamp, level, logger name, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        root_logger.addHandler(console_handler)

    return logging.getLogger("voice_agent")
