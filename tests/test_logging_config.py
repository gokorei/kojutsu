"""Tests for structured logging configuration."""

from __future__ import annotations

import json
import logging

from kojutsu.logging_config import JsonFormatter, configure_logging


def test_json_formatter_includes_structured_fields() -> None:
    record = logging.LogRecord(
        "kojutsu.test", logging.INFO, __file__, 1, "hello %s", ("world",), None
    )
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "kojutsu.test"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_configure_logging_installs_json_handler() -> None:
    try:
        configure_logging("WARNING", json_output=True)
        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
    finally:
        # Restore a neutral configuration for other tests.
        configure_logging("INFO", json_output=False)
