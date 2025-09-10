#!/usr/bin/env python3
"""
Quality evaluation script for AI Memory Service retrieval.

Seeds synthetic memories across several categories and queries with paraphrases,
then measures top-k retrieval metrics (P@k, R@k, MRR, nDCG).
Alternatively, read a fixed dataset JSON to avoid re-seeding on each run.

No external deps (uses urllib + json). Assumes service is running on host:port.
"""

import argparse
import json
import sys
import time
from urllib import request
import os


def http_post(url: str, payload: dict, timeout: float = 8.0):
    data = json.dumps(payload).encode('utf-8')
    req = request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def http_get(url: str, timeout: float = 8.0):
    with request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


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
                # Fallback alias
                resp = http_post(f'{base}/api/memory', body)
            cat_ids[cat].append(resp.get('id') or resp.get('memory_id'))
            time.sleep(0.05)
    return cat_ids


def eval_queries(base: str, relevant_by_cat: dict, queries: dict, k: int = 5) -> dict:
    results = []
    for cat, qs in queries.items():
        relevant = relevant_by_cat.get(cat, set())
        for q in qs:
            body = {'query': q, 'limit': max(10, k)}
            try:
                resp = http_post(f'{base}/search', body)
                items = resp.get('results', [])
            except Exception:
                resp = http_post(f'{base}/api/memory/search', body)
                items = resp.get('memories', [])
            # If relevance contains strings → content mode: compare normalized content strings
            if relevant and isinstance(next(iter(relevant)), str):
                def norm_txt(x):
                    import re
                    x = (x or '').lower()
                    x = re.sub(r"[^a-z0-9\s]+", "", x)
                    x = re.sub(r"\s+", " ", x).strip()
                    return x
                retrieved = [norm_txt(it.get('content') or '') for it in items]
            else:
                retrieved = [it.get('id') for it in items if it.get('id')]
            metrics = compute_metrics(set(relevant), retrieved, k)
            results.append({'category': cat, 'query': q, 'metrics': metrics})
            time.sleep(0.02)
    return aggregate_metrics(results)


def compute_metrics(relevant: set, retrieved: list, k: int) -> dict:
    topk = retrieved[:k]
    hits = [1 if r in relevant else 0 for r in topk]
    p_at_k = sum(hits) / max(1, k)
    # Recall@k uses |relevant| as denominator (bounded)
    r_at_k = sum(hits) / max(1, len(relevant))
    # MRR: reciprocal rank of first hit in full list
    rr = 0.0
    for i, rid in enumerate(retrieved, 1):
        if rid in relevant:
            rr = 1.0 / i
            break
    # nDCG@k
    dcg = 0.0
    for i, h in enumerate(hits, 1):
        if h:
            dcg += 1.0 / (1.0 + (i).bit_length())  # approx log2(i+1) via bit_length
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
    return {'results': results, 'aggregate': agg}


def wait_for_health(base: str, timeout_sec: int = 20) -> bool:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--out', default='quality_report.json')
    ap.add_argument('--min-p5', type=float, default=float(os.environ.get('MIN_P5', '0')))
    ap.add_argument('--min-mrr', type=float, default=float(os.environ.get('MIN_MRR', '0')))
    ap.add_argument('--min-ndcg', type=float, default=float(os.environ.get('MIN_NDCG', '0')))
    ap.add_argument('--dataset', default='', help='Path to dataset JSON (optional)')
    ap.add_argument('--use-existing', action='store_true', help='Do not seed; use existing quality/* contexts')
    ap.add_argument('--seed-file', default='reports/dataset_seed.json', help='Path to existing seed map (if present, used by default)')
    ap.add_argument('--force-seed', action='store_true', help='Seed even if seed-file exists')
    ap.add_argument('--relevance', choices=['ids','context','content'], default='context', help='Ground truth: ids (seeded), context membership, or content-based (normalized text)')
    args = ap.parse_args()

    base = f'http://{args.host}:{args.port}'

    # Wait health
    if not wait_for_health(base, timeout_sec=20):
        try:
            _ = http_get(f'{base}/health')
        except Exception as e:
            print(f'ERROR: health check failed: {e}', file=sys.stderr)
        sys.exit(2)

    # Load dataset JSON if provided; else use built-in default
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

    # Use existing seed-file if present (unless --force-seed), else apply --use-existing/seed logic
    cat_ids = None
    if not args.force_seed and os.path.isfile(args.seed_file):
        try:
            with open(args.seed_file,'r',encoding='utf-8') as f:
                data = json.load(f)
            cats = data.get('categories',{})
            cat_ids = {k: v.get('ids',[]) for k,v in cats.items() if isinstance(v,dict)}
        except Exception:
            cat_ids = None
    if cat_ids is None:
        if args.use_existing:
            cat_ids = None
        else:
            cat_ids = seed_memories(base, categories)

    # Build relevance map
    relevant_by_cat = {}
    if args.relevance == 'ids' and cat_ids:
        relevant_by_cat = {k: set(v) for k,v in cat_ids.items()}
    elif args.relevance == 'content':
        # Content-based: use dataset items as ground-truth (normalized)
        def norm_txt(x):
            import re
            x = (x or '').lower()
            x = re.sub(r"[^a-z0-9\s]+", "", x)
            x = re.sub(r"\s+", " ", x).strip()
            return x
        relevant_by_cat = {cat: set(norm_txt(s) for s in items) for cat, items in categories.items()}
    else:
        # Derive by context membership
        for cat in queries.keys():
            try:
                # Provide an empty query to satisfy tolerant parsing and avoid 422
                body = {'context': f'quality/{cat}', 'limit': 1000, 'query': ''}
                resp = http_post(f'{base}/search/context', body)
                items = resp.get('results', [])
                relevant_by_cat[cat] = set([it.get('id') for it in items if it.get('id')])
            except Exception:
                relevant_by_cat[cat] = set()

    report = eval_queries(base, relevant_by_cat, queries, args.k)

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    agg = report['aggregate']
    print(json.dumps({'p_at_k': agg['p_at_k'], 'r_at_k': agg['r_at_k'], 'mrr': agg['mrr'], 'ndcg': agg['ndcg']}, indent=2))
    # Thresholds gating
    if args.min_p5 or args.min_mrr or args.min_ndcg:
        ok = True
        if args.min_p5 and agg['p_at_k'] < args.min_p5:
            ok = False
        if args.min_mrr and agg['mrr'] < args.min_mrr:
            ok = False
        if args.min_ndcg and agg['ndcg'] < args.min_ndcg:
            ok = False
        if not ok:
            sys.exit(3)


if __name__ == '__main__':
    main()
