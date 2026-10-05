"""
Application entry point for the Real-Time Voice Computer Agent.
Phase 1: Verifies configuration and logging initialization without invoking hardware or AI services.
"""

import sys
from app.config import settings
from app.utils import setup_logging


def main() -> int:
    """Entry point function."""
    logger = setup_logging(settings.log_level)
    logger.info("Initializing %s...", settings.app_name)
    logger.info("Environment: %s", settings.app_env)
    logger.info("Logging configured at level: %s", settings.log_level)
    logger.info("Phase 1 foundation initialized successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
