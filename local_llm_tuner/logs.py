"""JSONL logging helpers for usage stats and prompt changes."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class UsageLogger:
    """Append-only JSONL writer for per-call LLM usage rows.

    Each call writes one line: timestamp + stats. Intended to be passed as
    DocumentHarness.logger_fn.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, row: dict) -> None:
        out = {"timestamp": datetime.now().isoformat(timespec="seconds"), **row}
        with self.path.open("a") as f:
            f.write(json.dumps(out) + "\n")


def log_prompt_change(
    path: str | Path,
    *,
    from_version: Optional[int] = None,
    to_version: Optional[int] = None,
    triggering_documents: Optional[list[str]] = None,
    observed_gaps: Optional[list[str]] = None,
    changes: Optional[list[str]] = None,
    expected_effect: Optional[str] = None,
    component: str = "prompts",
    extra: Optional[dict] = None,
) -> None:
    """Append a structured entry to prompt_changes.jsonl.

    Keep one entry per prompt bump. The entry should tell a future reader
    WHY the change was made and what effect was expected.
    """
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "component": component,
        "from_version": from_version,
        "to_version": to_version,
        "triggering_documents": triggering_documents or [],
        "observed_gaps": observed_gaps or [],
        "changes": changes or [],
        "expected_effect": expected_effect or "",
    }
    if extra:
        entry.update(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(entry) + "\n")
