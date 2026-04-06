"""
Simple logging module for PipeEE simulator.
"""

import logging
import sys
from typing import Optional, Union

# Default logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Create a formatter
_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

# Create a stream handler (stdout)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_formatter)

# Configure the root logger
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_root_logger.handlers = []  # Remove any existing handlers
_root_logger.addHandler(_stream_handler)

# File handler
_file_handler = None


def configure_logging(
    log_level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    file_mode: str = "a",
    disable_console: bool = False,
):
    """
    Configure logging settings.

    Args:
        log_level: Log level (e.g., logging.INFO, 'DEBUG', 'ERROR')
        log_file: File path to write logs to (None for no file output)
        file_mode: File opening mode (default: 'a' for append)
        disable_console: Whether to disable console output
    """
    global _file_handler

    # Set root logger level
    if isinstance(log_level, str):
        log_level = log_level.upper()
        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        log_level = log_level_map.get(log_level, logging.INFO)

    _root_logger.setLevel(log_level)

    # Handle console output
    if disable_console:
        if _stream_handler in _root_logger.handlers:
            _root_logger.removeHandler(_stream_handler)
    else:
        if _stream_handler not in _root_logger.handlers:
            _root_logger.addHandler(_stream_handler)

    # Handle file output
    if log_file:
        # Remove existing file handler if it exists
        if _file_handler is not None:
            if _file_handler in _root_logger.handlers:
                _root_logger.removeHandler(_file_handler)
            _file_handler.close()

        try:
            _file_handler = logging.FileHandler(log_file, mode=file_mode)
            _file_handler.setFormatter(_formatter)
            _file_handler.setLevel(log_level)
            _root_logger.addHandler(_file_handler)
            _root_logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            _root_logger.error(f"Failed to open log file {log_file}: {e}")
    else:
        # Remove file handler if log_file is None
        if _file_handler is not None:
            if _file_handler in _root_logger.handlers:
                _root_logger.removeHandler(_file_handler)
            _file_handler.close()
            _file_handler = None

    # Also set levels for existing loggers
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and name.startswith("pipeee."):
            logger.setLevel(log_level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Name of the logger. If None, returns the root logger.

    Returns:
        Logger instance.
    """
    if name is None:
        return _root_logger

    logger = logging.getLogger(name)
    logger.setLevel(_root_logger.getEffectiveLevel())

    # Ensure the logger has handlers (only add once)
    if not logger.handlers:
        if _file_handler:
            logger.addHandler(_file_handler)
        if _stream_handler in _root_logger.handlers:
            logger.addHandler(_stream_handler)
        logger.propagate = False  # Prevent propagating to root logger twice

    return logger
