"""Structured (JSON) logging configuration.

Logs go to **stderr** by default so stdout stays clean for anything that
reserves it (e.g. stdio protocols).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Render log records as a single JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(
    level: str | None = None,
    *,
    json_output: bool | None = None,
) -> None:
    """Install a stderr handler on the root logger.

    Level/output come from ``KOJUTSU_LOG_LEVEL`` / ``KOJUTSU_LOG_JSON``
    unless overridden.
    """
    resolved_level = (
        level if level is not None else os.getenv("KOJUTSU_LOG_LEVEL", "INFO")
    ).upper()
    if json_output is None:
        json_output = os.getenv("KOJUTSU_LOG_JSON", "true").lower() not in {"0", "false", "no"}

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)
