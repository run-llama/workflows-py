import logging
import sys
from typing import Any, Literal

from llama_agents.core._compat import get_logging_level_mapping
from pythonjsonlogger.json import JsonFormatter


class UvicornStyleFormatter(logging.Formatter):
    """Formatter that mimics uvicorn's beautiful colored style"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        # Get color for log level
        level_color = self.COLORS.get(record.levelname, "")

        # Format like uvicorn: "INFO:     message"
        formatted = (
            f"{level_color}{record.levelname:<8}{self.RESET} {record.getMessage()}"
        )

        return formatted


class CleanJsonFormatter(JsonFormatter):
    """JSON formatter that excludes redundant fields"""

    def add_fields(
        self,
        log_data: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_data, record, message_dict)
        # Remove redundant fields
        log_data.pop("color_message", None)


def setup_logging(
    log_level: str = "info",
    log_format: Literal["standard", "json"] = "standard",
) -> None:
    """Configure application logging"""
    level_mapping = get_logging_level_mapping()
    level = level_mapping[log_level.upper()]

    if log_format == "json":
        # Configure JSON for application logs
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            CleanJsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logging.basicConfig(level=level, handlers=[handler], force=True)
    else:
        # Use uvicorn-style formatting for standard format
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(UvicornStyleFormatter())
        logging.basicConfig(level=level, handlers=[handler], force=True)


def get_uvicorn_log_config(log_level: str = "info") -> dict:
    """Strip uvicorn's default handlers so all logs flow through the root logger
    configured by ``setup_logging``."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "loggers": {
            "uvicorn": {"handlers": [], "level": log_level.upper()},
            "uvicorn.error": {"handlers": [], "level": log_level.upper()},
            "uvicorn.access": {"handlers": [], "level": log_level.upper()},
        },
    }
