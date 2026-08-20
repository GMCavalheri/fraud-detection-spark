"""Logging setup shared by the FastAPI app: console + rotating file handler
under a mounted logs/ volume, plus a request-logging middleware."""

import logging
import os
import time
from logging.handlers import RotatingFileHandler

from starlette.middleware.base import BaseHTTPMiddleware

LOG_DIR = os.environ.get("LOG_DIR", "/opt/logs")


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("api")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(LOG_DIR, "api.log"), maxBytes=5_000_000, backupCount=3
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        pass  # LOG_DIR not writable/mounted - console-only is fine

    logger.propagate = False
    return logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs method, path, status code, and duration for every request."""

    def __init__(self, app, logger: logging.Logger):
        super().__init__(app)
        self.logger = logger

    async def dispatch(self, request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000
        self.logger.info(
            "%s %s -> %d (%.1fms)",
            request.method, request.url.path, response.status_code, duration_ms,
        )
        return response
