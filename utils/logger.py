# q-reply/utils/logger.py
"""Centralized logging configuration module.

This module provides a unified logging setup for all project components,
ensuring consistent formatting and output handling across the application.

Typical usage example:
    from utils.logger import get_logger, setup_logging

    # Setup logging once at application start
    setup_logging(log_file=Path("app.log"), level=logging.INFO)

    # Get logger for specific module
    logger = get_logger(__name__)
    logger.info("Application started")
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys


__all__ = [
    "DEFAULT_FORMAT",
    "DEFAULT_LEVEL",
    "close_logging",
    "get_logger",
    "setup_logging",
]

# Default configuration
DEFAULT_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
DEFAULT_LEVEL = logging.INFO


class LoggerSetup:
    """Singleton class for managing application-wide logging configuration.

    This class ensures that logging is configured only once and provides
    methods to update configuration when needed (e.g., for resume operations).

    Attributes:
        _instance: Singleton instance.
        _initialized: Flag indicating if logging has been configured.
        _file_handler: Current file handler for potential updates.
    """

    _instance: LoggerSetup | None = None
    _initialized: bool = False
    _file_handler: logging.FileHandler | None = None

    def __new__(cls) -> LoggerSetup:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def setup(
        self,
        log_file: Path | None = None,
        level: int = DEFAULT_LEVEL,
        format_string: str = DEFAULT_FORMAT,
        force_reconfigure: bool = False,
    ) -> None:
        """Configure unified logging for all modules.

        Sets up consistent logging configuration for the entire application.
        Can be called multiple times; subsequent calls update file handler only.

        Args:
            log_file: Optional path to log file. If provided, logs will be
                written to both console and file.
            level: Logging level (e.g., logging.INFO, logging.DEBUG).
            format_string: Format string for log messages.
            force_reconfigure: If True, forces complete reconfiguration.

        Raises:
            PermissionError: If log file cannot be created due to permissions.
            OSError: If log file path is invalid or inaccessible.
        """
        root_logger = logging.getLogger()

        # If already initialized and not forcing, only update file handler
        if self._initialized and not force_reconfigure:
            if log_file:
                self._update_file_handler(log_file, level, format_string)
            return

        # Create formatter
        formatter = logging.Formatter(format_string)

        # Configure root logger
        root_logger.setLevel(level)

        # Clear existing handlers to avoid duplicates
        root_logger.handlers.clear()
        self._file_handler = None

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler if requested
        if log_file:
            self._add_file_handler(log_file, level, formatter)

        self._initialized = True

        # Log successful initialization
        logger = logging.getLogger(__name__)
        if log_file:
            logger.info(f"Logging initialized with file: {log_file}")
        else:
            logger.info("Logging initialized (console only)")

    def _add_file_handler(self, log_file: Path, level: int, formatter: logging.Formatter) -> None:
        """Add or replace file handler.

        Args:
            log_file: Path to log file.
            level: Logging level.
            formatter: Log message formatter.

        Raises:
            PermissionError: If log file cannot be created.
            OSError: If path is invalid.
        """
        try:
            # Ensure directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Create new file handler
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)

            # Add to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)

            # Store reference for potential updates
            self._file_handler = file_handler

        except PermissionError as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Permission denied creating log file: {e}")
            raise
        except OSError as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Cannot create log file: {e}")
            raise

    def _update_file_handler(self, log_file: Path, level: int, format_string: str) -> None:
        """Update file handler for resumed sessions.

        Args:
            log_file: New log file path.
            level: Logging level.
            format_string: Format string for messages.
        """
        root_logger = logging.getLogger()

        # Remove old file handler if exists
        if self._file_handler:
            root_logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None

        # Add new file handler
        formatter = logging.Formatter(format_string)
        self._add_file_handler(log_file, level, formatter)

        # Log the change
        logger = logging.getLogger(__name__)
        logger.info(f"Log file updated to: {log_file}")

    def close(self) -> None:
        """Close file handler and clean up resources.

        Should be called at application shutdown to ensure
        all log messages are flushed and files are properly closed.
        """
        if self._file_handler:
            self._file_handler.close()
            self._file_handler = None


# Module-level convenience functions
_setup = LoggerSetup()


def setup_logging(
    log_file: Path | None = None,
    level: int = DEFAULT_LEVEL,
    format_string: str = DEFAULT_FORMAT,
    force_reconfigure: bool = False,
) -> None:
    """Configure unified logging for the application.

    This is a convenience wrapper around LoggerSetup.setup().

    Args:
        log_file: Optional path to log file.
        level: Logging level.
        format_string: Format string for log messages.
        force_reconfigure: If True, forces complete reconfiguration.

    Example:
        >>> from pathlib import Path
        >>> setup_logging(log_file=Path("app.log"), level=logging.DEBUG)
    """
    _setup.setup(log_file, level, format_string, force_reconfigure)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the specified module.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    # Ensure basic setup if not initialized
    if not _setup._initialized:
        setup_logging()

    return logging.getLogger(name)


def close_logging() -> None:
    """Close logging handlers and clean up resources.

    Should be called at application shutdown.
    """
    _setup.close()


# Configure basic logging on module import for early messages
# This ensures that import-time log messages are not lost
if not _setup._initialized:
    setup_logging()
