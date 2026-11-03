from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FAMILIES = ("YACHT", "OSV", "BULKER", "TANKER", "CARGO", "PATROL")


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("global", "families"):
        if key not in cfg:
            raise ValueError(f"configuration missing `{key}`")
    missing = [f for f in FAMILIES if f not in cfg["families"]]
    if missing:
        raise ValueError(f"configuration missing families: {missing}")
    return cfg


def family_config(cfg: dict[str, Any], family: str) -> dict[str, Any]:
    merged = dict(cfg.get("global", {}))
    merged.update(cfg["families"][family])
    return merged
