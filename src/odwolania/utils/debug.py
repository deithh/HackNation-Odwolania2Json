"""Debug sink: persists the artifact of each pipeline step per document for auditability."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DebugSink:
    """No-op when `root` is None so callers can record artifacts unconditionally."""

    def __init__(self, root: Path | None, rel_path: Path):
        self.dir = (root / rel_path.parent / rel_path.stem) if root else None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def write(self, filename: str, content: Any) -> None:
        if self.dir is None:
            return
        target = self.dir / filename
        if isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            target.write_text(
                json.dumps(content, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
