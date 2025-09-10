#!/usr/bin/env python3
"""
Quality streaming for AI Memory Service.

Runs periodic evaluation of retrieval quality (P@k, MRR, nDCG) without re-seeding
on every iteration. Optionally seeds once if quality/* contexts are missing.

Outputs JSON lines with timestamped aggregate metrics.
Optionally reads a dataset file (datasets/quality/dataset.json) and uses it
for categories, items, and queries. If --seed-if-missing is passed, the dataset
will be seeded once if corresponding contexts are absent.
"""

import argparse
import json
import sys
import time
from urllib import request
import os
from datetime import datetime, timezone


def iso_now():
    return datetime.now(timezone.utc).isoformat()


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
            h = http_get(f"{base}/health", timeout=2.0)
            if h and h.get("status") == "healthy":
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def list_contexts(base: str) -> list[str]:
    try:
        d = http_get(f"{base}/contexts", timeout=4.0)
        return d.get("contexts", []) if isinstance(d, dict) else []
    except Exception:
        return []


def seed_memories(base: str, categories: dict) -> dict:
    cat_ids = {cat: [] for cat in categories}
    for cat, items in categories.items():
        for text in items:
            body = {
                'content': text,
                'context_hint': f'quality/{cat}',
                'memory_type': 'semantic'
            }
            try:
                resp = http_post(f'{base}/memory', body)
            except Exception:
                resp = http_post(f'{base}/api/memory', body)
            cat_ids[cat].append(resp.get('id') or resp.get('memory_id'))
            time.sleep(0.02)
    return cat_ids


def compute_metrics(relevant: set, retrieved: list, k: int) -> dict:
    topk = retrieved[:k]
    hits = [1 if r in relevant else 0 for r in topk]
    p_at_k = sum(hits) / max(1, k)
    r_at_k = sum(hits) / max(1, len(relevant))
    rr = 0.0
    for i, rid in enumerate(retrieved, 1):
        if rid in relevant:
            rr = 1.0 / i
            break
    # Simple approx nDCG using bit_length for log2(i+1)
    dcg = 0.0
    for i, h in enumerate(hits, 1):
        if h:
            dcg += 1.0 / (1.0 + (i).bit_length())
    ideal_hits = [1] * min(k, len(relevant))
    idcg = 0.0
    for i, h in enumerate(ideal_hits, 1):
        if h:
            idcg += 1.0 / (1.0 + (i).bit_length())
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {'p_at_k': p_at_k, 'r_at_k': r_at_k, 'mrr': rr, 'ndcg': ndcg}


def aggregate_metrics(results: list) -> dict:
    agg = {'p_at_k': 0.0, 'r_at_k': 0.0, 'mrr': 0.0, 'ndcg': 0.0}
    n = max(1, len(results))
    for r in results:
        m = r['metrics']
        for key in agg:
            agg[key] += m.get(key, 0.0)
    for key in agg:
        agg[key] /= n
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--out', default='/tmp/quality_stream.jsonl')
    ap.add_argument('--interval', type=float, default=15.0)
    ap.add_argument('--max-iterations', type=int, default=0, help='0 = infinite')
    ap.add_argument('--seed-if-missing', action='store_true', help='Seed dataset if quality/* contexts absent')
    ap.add_argument('--dataset', default='', help='Path to dataset JSON (defaults to builtin)')
    args = ap.parse_args()

    base = f'http://{args.host}:{args.port}'

    if not wait_for_health(base, timeout_sec=30):
        try:
            _ = http_get(f'{base}/health')
        except Exception as e:
            print(f'ERROR: memory service health check failed: {e}', file=sys.stderr)
        sys.exit(2)

    # Load dataset if provided; else fallback to builtin
    categories = {}
    queries = {}
    if args.dataset and os.path.isfile(args.dataset):
        try:
            with open(args.dataset, 'r', encoding='utf-8') as f:
                ds = json.load(f)
            for cat, spec in ds.get('categories', {}).items():
                categories[cat] = spec.get('items', [])
                queries[cat] = spec.get('queries', [])
        except Exception:
            categories = {}
            queries = {}
    if not categories:
        categories = {
            'rust': [
                'Rust is a systems programming language focused on safety',
                'Cargo is Rust package manager for crates',
                'Ownership and borrowing are core Rust concepts',
                'Tokio enables async in Rust applications',
                'Clippy helps lint Rust code',
                'Traits enable polymorphism in Rust'
            ],
            'python': [
                'Python is a high-level programming language',
                'Pip installs Python packages',
                'Virtualenv creates isolated Python environments',
                'Asyncio provides asynchronous IO in Python',
                'Type hints improve Python code readability',
                'Pandas is used for data analysis in Python'
            ],
            'cooking': [
                'Boil pasta in salted water until al dente',
                'Saute onions and garlic in olive oil',
                'Simmer tomato sauce with basil and oregano',
                'Bake bread at high temperature for crust',
                'Whisk eggs and sugar until fluffy',
                'Grill vegetables for a smoky flavor'
            ],
            'travel': [
                'Japan offers beautiful temples and gardens',
                'Use rail pass for efficient train travel',
                'Pack light for international flights',
                'Local cuisine is a key part of travel',
                'Hostels are budget-friendly accommodations',
                'Consider travel insurance for emergencies'
            ],
            'ml': [
                'Gradient descent optimizes neural networks',
                'Overfitting occurs when model memorizes data',
                'Cross-validation improves generalization',
                'Learning rate controls optimization step size',
                'Regularization reduces model complexity',
                'Batch normalization stabilizes training'
            ],
        }
    if not queries:
        queries = {
            'rust': [
                'What are Rust unique features for memory safety?',
                'Which tool manages Rust dependencies?'
            ],
            'python': [
                'How to manage Python packages?',
                'What library helps with dataframes in Python?'
            ],
            'cooking': [
                'How to cook spaghetti properly?',
                'What gives bread a crispy crust?'
            ],
            'travel': [
                'Advice for trains in Japan?',
                'How to travel light on flights?'
            ],
            'ml': [
                'How to prevent overfitting?',
                'What method adjusts step size in optimization?'
            ],
        }

    # Seed once if requested and missing
    cat_ids = None
    if args.seed_if_missing:
        ctxs = set(list_contexts(base))
        required = {f'quality/{c}' for c in categories.keys()}
        need_seed = any((r not in ctxs) for r in required)
        if need_seed:
            cat_ids = seed_memories(base, categories)

    # Build relevance map per category (either from seeding or by querying context)
    relevant_by_cat: dict[str, set] = {}
    if cat_ids:
        relevant_by_cat = {k: set(v) for k, v in cat_ids.items()}
    else:
        for cat in categories.keys():
            try:
                body = {'context': f'quality/{cat}', 'limit': 500}
                resp = http_post(f'{base}/search/context', body)
                items = resp.get('results', [])
                relevant_by_cat[cat] = set([it.get('id') for it in items if it.get('id')])
            except Exception:
                relevant_by_cat[cat] = set()

    iters = 0
    with open(args.out, 'a', encoding='utf-8') as f:
        while True:
            results = []
            # Build ground-truth by reading back ids per category if needed?
            # Simpler approach: treat top-k relevance by matching contexts.
            # We evaluate P@k etc by checking if returned memory IDs belong to context 'quality/<cat>'.
            for cat, qs in queries.items():
                for q in qs:
                    try:
                        body = {'query': q, 'limit': max(10, args.k)}
                        try:
                            resp = http_post(f'{base}/search', body)
                            items = resp.get('results', [])
                        except Exception:
                            resp = http_post(f'{base}/api/memory/search', body)
                            items = resp.get('memories', [])
                        retrieved = [it.get('id') for it in items if it.get('id')]
                        relevant_ids = relevant_by_cat.get(cat)
                        if not relevant_ids:
                            # Fallback: derive relevance by context membership in current results
                            relevant_ids = set([it.get('id') for it in items if str(it.get('context_path','')).startswith(f'quality/{cat}')])
                        metrics = compute_metrics(relevant_ids, retrieved, args.k)
                        results.append({'category': cat, 'query': q, 'metrics': metrics})
                    except Exception as e:
                        results.append({'category': cat, 'query': q, 'error': str(e), 'metrics': {'p_at_k': 0, 'r_at_k': 0, 'mrr': 0, 'ndcg': 0}})
                    time.sleep(0.01)

            agg = aggregate_metrics(results)
            rec = {"ts": iso_now(), "aggregate": agg, "detail_count": len(results)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps(rec))

            iters += 1
            if args.max_iterations > 0 and iters >= args.max_iterations:
                break
            time.sleep(max(0.5, args.interval))


if __name__ == '__main__':
    main()
