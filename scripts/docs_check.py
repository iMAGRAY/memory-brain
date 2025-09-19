#!/usr/bin/env python3
"""Check documentation consistency (stub)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

MANDATORY = [
    "README.md",
    "manifest.schema.json",
    "manifest.json",
    "contract.md",
    "memory_model.md",
    "time_model.md",
    "error_handling.md",
    "integration_agents.md",
    "cognitive_usage.md",
    "quality.md",
    "observability_pro.md",
    "release_checklist.md",
]


def main() -> int:
    missing = [name for name in MANDATORY if not (DOCS / name).exists()]
    if missing:
        print("DOCS_CHECK: missing files:\n" + "\n".join(missing), file=sys.stderr)
        return 1
    # Basic JSON validation
    try:
        json.loads((DOCS / "manifest.schema.json").read_text())
        json.loads((DOCS / "manifest.json").read_text())
    except json.JSONDecodeError as exc:
        print(f"DOCS_CHECK: invalid JSON — {exc}", file=sys.stderr)
        return 1
    print("DOCS_CHECK: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
