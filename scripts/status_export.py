#!/usr/bin/env python3
"""Stub status exporter."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "STATUS.md"


def main() -> int:
    STATUS.write_text(
        "# Service status\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n"
        "\n- Mode: offline (fallback)\n"
        "- Active features: []\n"
        "- Trust summary: N/A\n"
        "- Timeline anchors: N/A\n"
    )
    print(f"STATUS exported to {STATUS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
