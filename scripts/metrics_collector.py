#!/usr/bin/env python3
"""
Lightweight live metrics collector for AI Memory Service.

Polls REST API /health and /metrics, plus the Python embedding server /health and /stats,
and writes timestamped JSON lines to a file for later analysis.

Dependencies: standard library only (urllib, json).
"""

import argparse
import json
import sys
import time
import re
from urllib import request, error
from datetime import datetime, timezone


def http_get_text(url: str, timeout: float = 5.0) -> str | None:
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def http_get_json(url: str, timeout: float = 5.0):
    try:
        with request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def parse_prom_value(text: str | None, metric: str, labels: dict[str, str] | None = None) -> float | None:
    """
    Very small Prometheus text parser for a single gauge/counter sample line.
    Example line: service_available{service="embedding"} 1
    """
    if not text:
        return None
    if labels:
        # Build label matcher like {a="b",c="d"}
        parts = [f"{k}=\"{v}\"" for k, v in sorted(labels.items())]
        lbl = "{" + ",".join(parts) + "}"
        pattern = rf"^{re.escape(metric)}\{re.escape(lbl)}\s+([-+]?[0-9]*\.?[0-9]+)\b"
    else:
        pattern = rf"^{re.escape(metric)}(?:\{{.*?\}})?\s+([-+]?[0-9]*\.?[0-9]+)\b"
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8080", help="Memory service base URL")
    ap.add_argument("--emb", default="http://127.0.0.1:8090", help="Embedding server base URL")
    ap.add_argument("--interval", type=float, default=2.0, help="Polling interval seconds")
    ap.add_argument("--out", default="/tmp/metrics_timeseries.jsonl", help="Output JSONL path")
    ap.add_argument("--max-samples", type=int, default=0, help="Stop after N samples (0=infinite)")
    args = ap.parse_args()

    api_health_url = f"{args.api}/health"
    api_metrics_url = f"{args.api}/metrics"
    emb_health_url = f"{args.emb}/health"
    emb_stats_url = f"{args.emb}/stats"

    samples = 0
    with open(args.out, "a", encoding="utf-8") as f:
        while True:
            ts = iso_now()
            api_health = http_get_json(api_health_url, timeout=3.0)
            api_metrics = http_get_text(api_metrics_url, timeout=3.0)
            emb_health = http_get_json(emb_health_url, timeout=3.0)
            emb_stats = http_get_json(emb_stats_url, timeout=3.0)

            # Extract a few useful numbers from Prometheus text
            service_available_embedding = parse_prom_value(
                api_metrics, "service_available", {"service": "embedding"}
            )
            emb_dim_autodetect = parse_prom_value(
                api_metrics, "embedding_dimension", {"source": "autodetect"}
            )
            emb_dim_config = parse_prom_value(
                api_metrics, "embedding_dimension", {"source": "config"}
            )

            rec = {
                "ts": ts,
                "api": {
                    "healthy": bool(api_health and api_health.get("status") == "healthy"),
                    "services": api_health.get("services") if isinstance(api_health, dict) else None,
                    "embedding_dimension": api_health.get("embedding_dimension") if isinstance(api_health, dict) else None,
                    "memory_stats": api_health.get("memory_stats") if isinstance(api_health, dict) else None,
                },
                "metrics": {
                    "service_available_embedding": service_available_embedding,
                    "embedding_dim_autodetect": emb_dim_autodetect,
                    "embedding_dim_config": emb_dim_config,
                },
                "embedding": {
                    "healthy": bool(emb_health and emb_health.get("status") == "healthy"),
                    "stats": emb_stats,
                },
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            # Console heartbeat (brief)
            dim = rec["api"].get("embedding_dimension")
            avail = rec["metrics"]["service_available_embedding"]
            hit_rate = None
            if isinstance(emb_stats, dict):
                # try common keys from server's /stats
                hit_rate = emb_stats.get("cache_hit_rate") or emb_stats.get("cache_hit_rate")
            sys.stdout.write(
                f"[{ts}] api_ok={rec['api']['healthy']} emb_ok={rec['embedding']['healthy']} dim={dim} avail={avail} hit_rate={hit_rate}\n"
            )
            sys.stdout.flush()

            samples += 1
            if args.max_samples > 0 and samples >= args.max_samples:
                break
            time.sleep(max(0.1, args.interval))


if __name__ == "__main__":
    main()

