#!/usr/bin/env python3
"""
Extended quality metrics for AI Memory Service.

Computes diversity, contextual adaptation gain, consistency across paraphrases,
and graph connectivity/expansion (via /analytics/graph), appending results into
reports/quality_stream.jsonl under `extended` field. Optionally uses dataset
JSON (datasets/quality/dataset.json) for categories and queries.
"""

import argparse
import json
import time
from urllib import request
import os
from math import log2


def http_get(url: str, timeout: float = 8.0):
    with request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def http_post(url: str, payload: dict, timeout: float = 8.0):
    data = json.dumps(payload).encode('utf-8')
    req = request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def entropy(values):
    total = sum(values)
    if total <= 0:
        return 0.0
    e = 0.0
    for v in values:
        if v > 0:
            p = v / total
            e -= p * log2(p)
    return e


def ndcg_at_k(relevances, k):
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], 1):
        if rel:
            dcg += 1.0 / (1.0 + (i).bit_length())
    ideal = sum(1.0 / (1.0 + (i).bit_length()) for i in range(1, min(k, sum(relevances)) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def jaccard_at_k(list_a, list_b, k):
    a = set(list_a[:k])
    b = set(list_b[:k])
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--out', default='reports/quality_stream.jsonl')
    ap.add_argument('--dataset', default='', help='Path to dataset JSON (optional)')
    args = ap.parse_args()

    base = f'http://{args.host}:{args.port}'

    # Load dataset if provided, else use built-in query paraphrases
    categories = {}
    if args.dataset and os.path.isfile(args.dataset):
        try:
            with open(args.dataset, 'r', encoding='utf-8') as f:
                ds = json.load(f)
            for cat, spec in ds.get('categories', {}).items():
                categories[cat] = spec.get('queries', [])
        except Exception:
            categories = {}
    if not categories:
        categories = {
            'rust': [
                'What are Rust unique features for memory safety?',
                'Which tool manages Rust dependencies?',
                'Explain Rust ownership and borrowing briefly',
                'Which async runtime is popular in Rust?'
            ],
            'python': [
                'How to manage Python packages?',
                'What library helps with dataframes in Python?',
                'What is the purpose of virtual environments in Python?',
                'Which module provides async IO primitives in Python?'
            ],
            'cooking': [
                'How to cook spaghetti properly?',
                'What gives bread a crispy crust?',
                'What herbs are common for tomato sauce?',
                'How to saute onions correctly?'
            ],
            'travel': [
                'Advice for trains in Japan?',
                'How to travel light on flights?',
                'Why consider travel insurance?',
                'Budget accommodations for backpackers?'
            ],
            'ml': [
                'How to prevent overfitting?',
                'What method adjusts step size in optimization?',
                'What is cross validation used for?',
                'Why apply regularization?'
            ],
        }

    # Context mapping per category
    def ctx(cat):
        return f'quality/{cat}'

    # Retention (decay) evaluation — real decay via /maintenance/tick
    # Seeds isolated context, measures baseline P@K, applies tick(s), measures again, cleans up memories.
    # Enabled by env RETENTION_TICKS>0; default 1.
    retention_ticks = int(os.environ.get('RETENTION_TICKS', '10'))
    retention_items = int(os.environ.get('RETENTION_ITEMS', '8'))
    retention_k = int(os.environ.get('RETENTION_K', str(args.k)))
    retention_enabled = retention_ticks > 0 and retention_items > 0

    def store_memory(text, ctx_path):
        return http_post(f'{base}/memory', {'content': text, 'context_hint': ctx_path, 'memory_type': 'semantic'}).get('id')
    def delete_memory(mid):
        try:
            req = request.Request(f'{base}/memory/{mid}', method='DELETE')
            with request.urlopen(req, timeout=6) as _:
                return True
        except Exception:
            return False
    def p_at_k_for_seed(ctx_path, q_texts, k):
        hits = 0; total = 0
        for qt in q_texts:
            items = http_post(f'{base}/search', {'query': qt, 'limit': max(10,k)}).get('results', [])
            topk = items[:k]
            h = sum(1 for m in topk if (m.get('context_path') or '').startswith(ctx_path))
            hits += h; total += k
        return hits/total if total>0 else 0.0

    retention_result = None
    seed_ids = []
    seed_ctx = f'quality/retention/{int(time.time())}'
    if retention_enabled:
        # Seed N items
        seed_texts = [f'Retention Test Item {i} unique {time.time()}' for i in range(retention_items)]
        for t in seed_texts:
            try:
                mid = store_memory(t, seed_ctx)
                if mid: seed_ids.append(mid)
            except Exception:
                pass
        # Baseline P@K
        try:
            baseline = p_at_k_for_seed(seed_ctx, seed_texts, retention_k)
        except Exception:
            baseline = 0.0
        # Apply ticks sequentially and record curve
        curve = [{'tick': 0, 'p_at_k': baseline}]
        post = baseline
        for t in range(1, retention_ticks+1):
            try:
                http_post(f'{base}/maintenance/tick', {'ticks': 1})
            except Exception:
                pass
            try:
                post = p_at_k_for_seed(seed_ctx, seed_texts, retention_k)
            except Exception:
                post = 0.0
            curve.append({'tick': t, 'p_at_k': post})
        # Retention AUC (average of points)
        auc = sum(pt['p_at_k'] for pt in curve) / len(curve)
        # Cleanup
        for mid in seed_ids:
            delete_memory(mid)
        retention_result = {'baseline': baseline, 'post': post, 'ratio': (post/(baseline or 1.0)), 'curve': curve, 'auc': auc}

    # Advanced recall gains (proxy for multi-hop reasoning)
    depth2_success = 0; depth3_success = 0; total_q = 0
    try:
        for cat, queries in categories.items():
            for q in queries:
                nctx = http_post(f'{base}/search', {'query': q, 'limit': max(10,args.k)}).get('results', [])
                adv  = http_post(f'{base}/search/advanced', {'query': q, 'limit': max(10,args.k), 'context': ctx(cat), 'include_related': True}).get('results', [])
                rel_n = [(1 if (m.get('context_path') or '').startswith(ctx(cat)) else 0) for m in nctx]
                rel_a = [(1 if (m.get('context_path') or '').startswith(ctx(cat)) else 0) for m in adv]
                ndcg_n = ndcg_at_k(rel_n, args.k)
                ndcg_a = ndcg_at_k(rel_a, args.k)
                gain = max(0.0, ndcg_a - ndcg_n)
                if gain >= 0.1: depth2_success += 1
                if gain >= 0.2: depth3_success += 1
                total_q += 1
    except Exception:
        pass

    # Graph-based connection quality composite (requires /analytics/graph)
    conn_quality = None
    try:
        g = http_get(f'{base}/analytics/graph').get('graph')
        if g:
            import math
            a = min(1.0, float(g.get('two_hop_expansion', 0.0))/5.0)
            b = max(0.0, min(1.0, float(g.get('avg_closure', 0.0))))
            c = 1.0 - min(1.0, float(g.get('avg_shortest_path', 0.0))/5.0)
            conn_quality = 0.4*a + 0.4*b + 0.2*c
    except Exception:
        pass

    # Helper: fetch context stats for stratified weights (path->count)
    def fetch_context_stats():
        try:
            d = http_get(f'{base}/analytics/contexts')
            ctxs = d.get('contexts', []) if isinstance(d, dict) else []
            return {c.get('path'): int(c.get('count', 0)) for c in ctxs if isinstance(c, dict)}
        except Exception:
            # Fallback: enumerate /contexts then /context/:path
            stats = {}
            try:
                allc = http_get(f'{base}/contexts').get('contexts', [])
            except Exception:
                allc = []
            for p in allc:
                try:
                    info = http_get(f'{base}/context/{p}')
                    stats[p] = int(info.get('memory_count', 0))
                except Exception:
                    stats[p] = 0
            return stats

    ctx_stats = fetch_context_stats()

    # Inverse propensity weight for context c: 1/log(2+count)
    def ctx_weight(path: str) -> float:
        c = ctx_stats.get(path, 0)
        import math
        return 1.0 / math.log(2.0 + float(c)) if c >= 0 else 1.0

    # Collect results for /search and /search/context
    contexts_counts = {}
    jc_list = []
    ndcg_gain_list = []
    stratified_p_list = []
    stratified_mrr_list = []
    p_at5_list = []
    p_at10_list = []
    for cat, queries in categories.items():
        # Consistency via jaccard@k between two paraphrases (if 2 queries)
        retrieved_nocontext = []
        retrieved_context = []
        for q in queries:
            body = {'query': q, 'limit': max(10, args.k)}
            nctx = http_post(f'{base}/search', body).get('results', [])
            cctx = http_post(f'{base}/search/context', {'context': ctx(cat), 'limit': max(10,args.k), 'query': q}).get('results', [])
            retrieved_nocontext.append([m.get('id') for m in nctx])
            retrieved_context.append([m.get('id') for m in cctx])
            # Diversity: count contexts in non-contextual results
            for m in nctx:
                cp = m.get('context_path') or ''
                contexts_counts[cp] = contexts_counts.get(cp, 0) + 1
            # Contextual adaptation gain (per query): ndcg_ctx - ndcg_nctx with proxy relevance
            # Proxy relevance: a result is relevant if its context matches category
            rel_ctx = [(1 if (m.get('context_path') or '').startswith(ctx(cat)) else 0) for m in cctx]
            rel_nctx = [(1 if (m.get('context_path') or '').startswith(ctx(cat)) else 0) for m in nctx]
            ndcg_ctx = ndcg_at_k(rel_ctx, args.k)
            ndcg_nctx = ndcg_at_k(rel_nctx, args.k)
            ndcg_gain_list.append(max(0.0, ndcg_ctx - ndcg_nctx))

            # Stratified P@K and MRR on non-contextual results
            topk = nctx[:args.k]
            if topk:
                weights = [ctx_weight(m.get('context_path') or '') for m in topk]
                hits = [1 if (m.get('context_path') or '').startswith(ctx(cat)) else 0 for m in topk]
                sw = sum(weights) or 1.0
                stratified_p_list.append(sum(w*h for w,h in zip(weights,hits)) / sw)
                # Stratified normalized MRR: (w_first/rank)/max_weight
                first_rank = None; first_w = 0.0
                for i,m in enumerate(topk, start=1):
                    if (m.get('context_path') or '').startswith(ctx(cat)):
                        first_rank = i; first_w = ctx_weight(m.get('context_path') or ''); break
                if first_rank:
                    w_max = max(weights) or 1.0
                    stratified_mrr_list.append((first_w / first_rank) / w_max)
                else:
                    stratified_mrr_list.append(0.0)

            # Perturbation robustness proxy: P@5 vs P@10
            k5 = 5; k10 = 10
            top5 = nctx[:k5]; top10 = nctx[:k10]
            p5 = sum(1 for m in top5 if (m.get('context_path') or '').startswith(ctx(cat))) / max(1,k5)
            p10 = sum(1 for m in top10 if (m.get('context_path') or '').startswith(ctx(cat))) / max(1,k10)
            p_at5_list.append(p5); p_at10_list.append(p10)
        if len(retrieved_nocontext) >= 2:
            jc = jaccard_at_k(retrieved_nocontext[0], retrieved_nocontext[1], args.k)
            jc_list.append(jc)

    diversity_entropy = entropy(list(contexts_counts.values()))
    try:
        total_contexts = len(http_get(f'{base}/contexts').get('contexts', []))
    except Exception:
        total_contexts = 0
    covered_contexts = len([k for k,v in contexts_counts.items() if v>0])
    coverage_ratio = (covered_contexts/total_contexts) if total_contexts>0 else 0.0
    contextual_gain_ndcg = sum(ndcg_gain_list)/len(ndcg_gain_list) if ndcg_gain_list else 0.0
    consistency_jaccard_at_k = sum(jc_list)/len(jc_list) if jc_list else 0.0

    # Graph connectivity/expansion via server analytics
    try:
        graph = http_get(f'{base}/analytics/graph').get('graph')
    except Exception:
        graph = None

    # Aggregate extended metrics
    stratified_p_at_k = sum(stratified_p_list)/len(stratified_p_list) if stratified_p_list else 0.0
    stratified_mrr = sum(stratified_mrr_list)/len(stratified_mrr_list) if stratified_mrr_list else 0.0
    if p_at5_list and p_at10_list:
        diffs = [abs(a-b) for a,b in zip(p_at5_list,p_at10_list)]
        perturbation_robustness = 1.0 - (sum(diffs)/len(diffs))
    else:
        perturbation_robustness = 0.0
    reasoning_depth = 0.0
    if graph and isinstance(graph, dict):
        try:
            reasoning_depth = min(1.0, float(graph.get('two_hop_expansion', 0.0))/5.0)
        except Exception:
            reasoning_depth = 0.0

    record = {
        'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'aggregate': {},
        'extended': {
            'diversity_entropy': diversity_entropy,
            'coverage_ratio': coverage_ratio,
            'contextual_gain_ndcg': contextual_gain_ndcg,
            'consistency_jaccard_at_k': consistency_jaccard_at_k,
            'stratified_p_at_k': stratified_p_at_k,
            'stratified_mrr': stratified_mrr,
            'perturbation_robustness': perturbation_robustness,
            'reasoning_depth': reasoning_depth,
            'graph': graph,
        }
    }
    if retention_result:
        record['extended']['memory_retention'] = retention_result
    if total_q > 0:
        record['extended']['depth2_success'] = depth2_success/total_q
        record['extended']['depth3_success'] = depth3_success/total_q
    if conn_quality is not None:
        record['extended']['connection_quality'] = conn_quality

    # Append JSONL
    with open(args.out, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
