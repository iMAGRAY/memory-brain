#!/usr/bin/env python3
"""
Seed a deterministic retrieval quality dataset into the AI Memory Service.

Reads datasets/quality/dataset.json and stores items under contexts quality/<cat>,
recording the server-assigned IDs to reports/dataset_seed.json for later use.

No external dependencies. Works on Windows and Unix (urllib only).
"""

import argparse
import json
import sys
import time
from urllib import request


def http_post(url: str, payload: dict, timeout: float = 8.0):
    data = json.dumps(payload).encode('utf-8')
    req = request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def http_get(url: str, timeout: float = 8.0):
    with request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def wait_for_health(base: str, timeout_sec: int = 30) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            h = http_get(f"{base}/health", timeout=3.0)
            if h and h.get("status") == "healthy":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def seed(base: str, dataset: dict) -> dict:
    out = {"name": dataset.get("name"), "version": dataset.get("version"), "seeded_at": int(time.time()), "categories": {}}
    cats = dataset.get("categories", {})
    for cat, spec in cats.items():
        items = spec.get("items", [])
        ctx = f"quality/{cat}"
        ids = []
        for text in items:
            body = {"content": text, "context_hint": ctx, "memory_type": "semantic"}
            # Try primary endpoint, fallback to alias for compatibility
            try:
                resp = http_post(f"{base}/memory", body)
            except Exception:
                resp = http_post(f"{base}/api/memory", body)
            mid = resp.get("id") or resp.get("memory_id")
            if mid:
                ids.append(mid)
            time.sleep(0.01)
        out["categories"][cat] = {"context": ctx, "ids": ids, "count": len(ids)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--dataset", default="datasets/quality/dataset.json")
    ap.add_argument("--out", default="reports/dataset_seed.json")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    if not wait_for_health(base, 30):
        print("ERROR: memory service health check failed", file=sys.stderr)
        sys.exit(2)

    with open(args.dataset, "r", encoding="utf-8") as f:
        ds = json.load(f)

    seeded = seed(base, ds)
    # minimal check
    total = sum(cat.get("count", 0) for cat in seeded.get("categories", {}).values())
    if total == 0:
        print("ERROR: dataset seeding produced zero items", file=sys.stderr)
        sys.exit(3)

    # ensure reports dir exists (portable)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(seeded, f, ensure_ascii=False, indent=2)
    print(json.dumps({"seeded_total": total, "out": args.out}))


if __name__ == "__main__":
    main()

