"""
Centralized Logging Configuration for LLM Evaluation Pipeline

Provides consistent logging across all modules with:
- Console output with colored formatting
- File output with rotation
- Configurable log levels
"""
import logging
import logging.handlers
import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional


# ANSI color codes for console output
class LogColors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.GRAY,
        logging.INFO: LogColors.GREEN,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.MAGENTA,
    }
    
    def format(self, record):
        # Add color to level name
        color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)
        record.levelname = f"{color}{record.levelname:8}{LogColors.RESET}"
        
        # Add color to module name
        record.name = f"{LogColors.CYAN}{record.name}{LogColors.RESET}"
        
        return super().format(record)


def setup_logger(
    name: str = "llm_eval",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name (typically __name__ of the module)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
        console_output: Enable console logging
        file_output: Enable file logging
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_format = ColoredFormatter(
            fmt="%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output:
        requested_log_dir = Path(log_dir) if log_dir else Path(os.getenv("LLM_EVAL_LOG_DIR", "logs"))
        fallback_log_dir = Path(tempfile.gettempdir()) / "llm-eval-logs"

        candidate_dirs = [requested_log_dir]
        if fallback_log_dir != requested_log_dir:
            candidate_dirs.append(fallback_log_dir)

        file_handler = None
        for candidate_dir in candidate_dirs:
            try:
                candidate_dir.mkdir(parents=True, exist_ok=True)
                log_file = candidate_dir / f"eval_{datetime.now().strftime('%Y%m%d')}.log"
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=10_000_000,  # 10MB
                    backupCount=5,
                    encoding="utf-8"
                )
                break
            except (PermissionError, OSError):
                file_handler = None

        if file_handler:
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(funcName)s:%(lineno)d │ %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        elif console_output:
            logger.warning(
                "File logging disabled: could not create writable log directory (tried: %s)",
                ", ".join(str(path) for path in candidate_dirs)
            )
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a child logger with the given name.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance
    """
    # Ensure root logger is configured
    root_logger = logging.getLogger("llm_eval")
    if not root_logger.handlers:
        setup_logger()
    
    # Return child logger
    if name.startswith("llm_eval"):
        return logging.getLogger(name)
    return logging.getLogger(f"llm_eval.{name}")


# Convenience function for quick setup
def configure_logging(level: str = "INFO", log_dir: Optional[str] = None):
    """
    Quick logging configuration for the entire application.
    
    Args:
        level: Global log level
        log_dir: Log file directory
    """
    setup_logger(level=level, log_dir=log_dir)
