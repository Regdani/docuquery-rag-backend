"""Logging setup for the application's own loggers (everything under `app.`).

Uvicorn configures only its own loggers, so without this the `app.*` INFO records (request
IDs, durations, sources) are dropped and warnings fall back to Python's unformatted
last-resort handler. Records still propagate to the root logger, so test capture works.
"""

import logging
import sys

APP_LOGGER = "app"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_HANDLER_NAME = "docuquery-app"


def configure_logging(level: str) -> None:
    """Send `app.*` logs to stdout at `level`. Safe to call more than once."""
    logger = logging.getLogger(APP_LOGGER)
    logger.setLevel(level)
    if any(h.get_name() == _HANDLER_NAME for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
