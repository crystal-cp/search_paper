"""Incremental run logging for screening diagnostics."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lit_screening.utils import ensure_dir, to_plain_data


SECRET_MARKERS = ("api_key", "apikey", "token", "secret", "password", "key_input")


class ScreeningRunLogger:
    """Append JSONL events while a screening run is executing."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = ensure_dir(output_dir)
        self.path = self.output_dir / "run_events.jsonl"

    def log(
        self,
        stage: str,
        message: str,
        details: dict[str, Any] | None = None,
        level: str = "info",
    ) -> None:
        """Append one event record to run_events.jsonl."""

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "stage": stage,
            "message": message,
            "details": redact(details or {}),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_plain_data(record), ensure_ascii=False) + "\n")

    def log_exception(
        self,
        stage: str,
        exc: BaseException,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append an exception event with a compact traceback."""

        payload = dict(details or {})
        payload.update(
            {
                "exception_type": exc.__class__.__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc()[-4000:],
            }
        )
        self.log(stage, "Unhandled exception", payload, level="error")


def redact(value: Any) -> Any:
    """Remove likely secret values from nested diagnostics."""

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(marker in key_text for marker in SECRET_MARKERS):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = redact(item)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value
