"""
Tests for logging setup.
"""

import logging
from app.utils.logging import setup_logging


def test_setup_logging():
    """Verify that logging initializes with the requested log level."""
    logger = setup_logging("DEBUG")
    assert logger is not None
    assert logging.getLogger().level == logging.DEBUG
