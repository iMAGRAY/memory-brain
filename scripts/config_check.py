#!/usr/bin/env python3
"""Minimal configuration sanity checks."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    ROOT / ".env.example",
    ROOT / "docker-compose.pro.yml",
    ROOT / "Cargo.toml",
    ROOT / "docs" / "manifest.schema.json",
]


def main() -> int:
    missing = [str(p) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        print("CONFIG_CHECK: missing files:\n" + "\n".join(missing), file=sys.stderr)
        return 1
    print("CONFIG_CHECK: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
