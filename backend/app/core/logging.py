# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""
Logging configuration for AutoLabeler.
Provides structured logging with rich console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from pythonjsonlogger import jsonlogger

from .config import settings


# Install rich traceback handler
install_rich_traceback(show_locals=True)

# Console for rich output
console = Console()


def _make_formatter():
    if str(settings.LOG_FORMAT).lower() == "json":
        return jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    return logging.Formatter(settings.LOG_FORMAT)


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup application logging with rich console output and file handler.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        
    Returns:
        Configured logger instance
    """
    level = log_level or settings.LOG_LEVEL
    
    # Create logger
    logger = logging.getLogger("autolabeler")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Rich console handler
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
        show_time=True,
        show_path=True
    )
    rich_handler.setLevel(getattr(logging, level.upper()))
    rich_handler.setFormatter(_make_formatter())
    logger.addHandler(rich_handler)
    
    # File handler
    if log_file is None:
        log_file = settings.LOGS_DIR / "autolabeler.log"
    
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(_make_formatter())
    logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logging()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance."""
    if name:
        return logging.getLogger(f"autolabeler.{name}")
    return logger
