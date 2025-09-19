#!/usr/bin/env python3
"""Placeholder agent simulator harness.

Each test returns exit code 0 to indicate the workflow is wired.
Actual HTTP interactions must be implemented as the service evolves.
"""
import argparse
import sys

TESTS = {
    "baseline": "store/query determinism",
    "continuity": "timeline continuity",
    "session": "session persistence",
    "errors": "error policy",
    "similarity": "vector similarity",
    "human": "cognitive summary",
    "plan": "plan lifecycle",
    "conflict": "conflict reconciliation",
    "fallback": "LLM fallback"
}


def run_stub(name: str) -> int:
    print(f"[simulator] {name} — {TESTS[name]} (stub)")
    # TODO: replace stubs with real HTTP flows
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Agent simulator runner")
    parser.add_argument("test", choices=sorted(TESTS.keys()), help="Test to execute")
    args = parser.parse_args(argv)
    return run_stub(args.test)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
