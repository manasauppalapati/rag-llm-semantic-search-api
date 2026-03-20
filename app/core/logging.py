"""
Structured logging configuration for the RAG LLM Semantic Search API.

What this module does:
    Owns the application's root logging configuration and ensures every log
    line is emitted in a single, consistent format from process startup onward.

Why this exists:
    Observability only works when every module shares one configuration path.
    Without that guarantee, incidents degrade into duplicate lines, missing
    context, mixed formats, and startup logs that cannot be correlated.

How to use it:
    Call setup_logging() once from app startup code before serving requests.
    After that, modules can use logging.getLogger(__name__) directly or call
    get_logger(__name__) as a thin convenience wrapper.

What breaks if this is wrong:
    Misconfigured handlers produce duplicate log lines, JSON consumers break if
    field names drift, and writing to stderr bypasses the stdout stream Docker
    log drivers are expected to capture.
"""

from datetime import datetime, timezone
import logging
import sys
from typing import TextIO

from pythonjsonlogger.jsonlogger import JsonFormatter  # type: ignore[attr-defined]


_NOISY_LOGGERS: tuple[str, ...] = ("httpx", "httpcore", "openai", "faiss")
_DEV_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEV_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
_JSON_FORMAT: str = "%(asctime)s %(levelname)s %(name)s %(message)s"
_PRODUCTION_ENV: str = "production"
_COLOR_RESET: str = "\x1b[0m"
_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: "\x1b[36m",
    logging.INFO: "\x1b[32m",
    logging.WARNING: "\x1b[33m",
    logging.ERROR: "\x1b[31m",
    logging.CRITICAL: "\x1b[35m",
}


class StructuredJsonFormatter(JsonFormatter):
    """JSON formatter with a stable field schema for machine-parsed logs."""

    rename_fields: dict[str, str] = {
        "asctime": "timestamp",
        "levelname": "level",
        "name": "logger",
    }

    def __init__(self) -> None:
        super().__init__(
            fmt=_JSON_FORMAT,
            rename_fields=self.rename_fields,
        )

    def formatTime(
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,
    ) -> str:
        """Return an ISO 8601 UTC timestamp for every JSON log record."""
        del datefmt
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class _DevelopmentFormatter(logging.Formatter):
    """Human-readable formatter with optional ANSI color for local terminals."""

    def __init__(self, use_color: bool) -> None:
        super().__init__(fmt=_DEV_FORMAT, datefmt=_DEV_DATE_FORMAT)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Return a readable log line, colorized when stdout is interactive."""
        rendered_message = super().format(record)
        if not self._use_color:
            return rendered_message

        color = _LEVEL_COLORS.get(record.levelno)
        if color is None:
            return rendered_message
        return f"{color}{rendered_message}{_COLOR_RESET}"


def _supports_color(stream: TextIO) -> bool:
    """Return True when stdout is interactive enough to benefit from ANSI."""
    return stream.isatty()


def _create_formatter(app_env: str, stream: TextIO) -> logging.Formatter:
    """Build the correct formatter for the current runtime environment."""
    if app_env == _PRODUCTION_ENV:
        return StructuredJsonFormatter()
    return _DevelopmentFormatter(use_color=_supports_color(stream))


def setup_logging() -> None:
    """
    Configure the root logger for the entire application.

    Params:
        None.

    Returns:
        None.

    Raises:
        ValidationError: Propagated if configuration required by get_settings()
            is missing or invalid.
        Exception: Propagates any handler or formatter setup failure so the
            process does not continue with broken logging.
    """
    from app.core.config import get_settings

    settings = get_settings()
    root_logger = logging.getLogger()

    for handler in tuple(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        _create_formatter(app_env=settings.APP_ENV, stream=sys.stdout)
    )

    root_logger.addHandler(stream_handler)
    root_logger.setLevel(settings.LOG_LEVEL)

    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "level": settings.LOG_LEVEL,
            "env": settings.APP_ENV,
            "version": settings.APP_VERSION,
        },
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for application code.

    Params:
        name: The logger name callers should usually pass as __name__.

    Returns:
        The stdlib logging.Logger instance for the provided name.

    Raises:
        None directly. This is a thin convenience wrapper, while the real
        logging configuration lives in setup_logging().
    """
    return logging.getLogger(name)
