#!/usr/bin/env python3
"""
Deterministic parameter tuner for AI Memory Service retrieval quality.

- Uses coordinate descent over fixed grids for key env parameters.
- For each step: starts memory-server with given env, waits health, runs content-based
  quality_eval (P@5/MRR/nDCG), records metrics, picks best value per-parameter.
- No external deps; uses subprocess + urllib.

Usage (example):
  python3 scripts/param_tuner.py --host 127.0.0.1 --port 8080 \
    --dataset datasets/quality/dataset.json --k 5 --passes 1

Will emit reports/param_tuning_report.json and print recommended env line.
"""

import argparse
import json
import os
import signal
import sys
import time
from urllib import request
import subprocess


SERVER_BIN = os.path.abspath('./target/release/memory-server')
REPORT_PATH = 'reports/param_tuning_report.json'


def http_get_json(url: str, timeout: float = 8.0):
    with request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def run_cmd(cmd, env=None, cwd=None):
    return subprocess.Popen(cmd, env=env, cwd=cwd)


def kill_process_tree(proc: subprocess.Popen, timeout=5.0):
    if proc is None:
        return
    try:
        proc.terminate()
        t0 = time.time()
        while time.time() - t0 < timeout:
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        proc.kill()
    except Exception:
        pass


def wait_health(base: str, timeout_sec: int = 30) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            d = http_get_json(f'{base}/health', timeout=3.0)
            if d and d.get('status') == 'healthy':
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def quality_eval(base: str, dataset: str, k: int) -> dict:
    # Run scripts/quality_eval.py in-process via subprocess to avoid import deps
    cmd = [sys.executable, 'scripts/quality_eval.py',
           '--host', base.split('://',1)[1].split(':')[0],
           '--port', base.rsplit(':',1)[1],
           '--k', str(k),
           '--dataset', dataset,
           '--relevance', 'content',
           '--use-existing',
           '--out', '/tmp/quality_tune.json']
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=180)
    except subprocess.CalledProcessError as e:
        return {'error': f'quality_eval failed: {e.output.decode("utf-8", errors="ignore")[:200]}'}
    except subprocess.TimeoutExpired:
        return {'error': 'quality_eval timeout'}
    try:
        with open('/tmp/quality_tune.json','r',encoding='utf-8') as f:
            data = json.load(f)
        return data.get('aggregate', {})
    except Exception as e:
        return {'error': f'parse error: {e}'}


def score_tuple(agg: dict):
    # Lexicographic by (P@5, nDCG, MRR)
    return (float(agg.get('p_at_k', 0.0)), float(agg.get('ndcg', 0.0)), float(agg.get('mrr', 0.0)))


def ensure_embedding():
    try:
        subprocess.check_call(['make','embedding-up'])
    except Exception:
        pass


def ensure_dataset(host: str, port: int):
    base = f'http://{host}:{port}'
    try:
        ctxs = http_get_json(f'{base}/contexts').get('contexts', [])
        if any(str(c).startswith('quality/') for c in ctxs):
            return
    except Exception:
        pass
    try:
        subprocess.check_call(['make','dataset-seed'])
    except Exception:
        pass


def start_server(env_overrides: dict) -> subprocess.Popen:
    base_env = os.environ.copy()
    base_env.update({
        'RUST_LOG': 'info',
        'EMBEDDING_SERVER_URL': 'http://127.0.0.1:8091',
        'NEO4J_URI': 'bolt://localhost:7688',
        'NEO4J_USER': 'neo4j',
        'NEO4J_PASSWORD': 'testpass',
        'ORCHESTRATOR_FORCE_DISABLE': 'true',
        'DISABLE_SCHEDULERS': 'true',
        'SERVICE_HOST': '0.0.0.0',
        'SERVICE_PORT': '8080',
    })
    base_env.update(env_overrides)
    # Kill residual server
    try:
        subprocess.check_call(['pkill','-f','target/release/memory-server'])
        time.sleep(0.5)
    except Exception:
        pass
    # Start server
    return run_cmd([SERVER_BIN], env=base_env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--dataset', default='datasets/quality/dataset.json')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--passes', type=int, default=1)
    args = ap.parse_args()

    base = f'http://{args.host}:{args.port}'

    # Ensure build and embedding
    try:
        subprocess.check_call(['cargo','build','--release'])
    except Exception as e:
        print(f'ERROR: cargo build failed: {e}', file=sys.stderr)
        sys.exit(2)
    ensure_embedding()
    ensure_dataset(args.host, args.port)

    # Parameter grids (deterministic, moderate size)
    grids = {
        'CAND_SEM_MULT': [8, 10, 12],
        'CAND_SEM_THRESH': [0.05, 0.08, 0.10],
        'CAND_LEX_MULT': [3, 5, 8],
        'HYBRID_ALPHA': [0.4, 0.5, 0.6],
        'MMR_LAMBDA': [0.2, 0.3, 0.4],
        'CONTEXT_BOOST': [0.25, 0.35, 0.45],
        'CAND_SEM_CAP': [300, 500],
        'CAND_LEX_CAP': [100, 150],
        'MMR_TOP': [100, 200],
    }

    order = ['CAND_SEM_MULT','CAND_SEM_THRESH','CAND_LEX_MULT','HYBRID_ALPHA','MMR_LAMBDA','CONTEXT_BOOST','CAND_SEM_CAP','CAND_LEX_CAP','MMR_TOP']

    current = {
        'CAND_SEM_MULT': 10,
        'CAND_SEM_THRESH': 0.08,
        'CAND_LEX_MULT': 5,
        'HYBRID_ALPHA': 0.5,
        'MMR_LAMBDA': 0.3,
        'CONTEXT_BOOST': 0.35,
        'CAND_SEM_CAP': 300,
        'CAND_LEX_CAP': 100,
        'MMR_TOP': 150,
        'ENABLE_MMR': 1,
        'ALPHA_SHORT': 0.4,
        'ALPHA_LONG': 0.6,
    }

    history = []

    for _pass in range(max(1, args.passes)):
        for key in order:
            best_val = current[key]
            best_score = None
            for val in grids[key]:
                trial = current.copy()
                trial[key] = val
                # Prepare env overrides as strings
                env = { k: (str(v) if not isinstance(v,bool) else ('1' if v else '0')) for k,v in trial.items() }
                proc = start_server(env)
                ok = wait_health(base, 30)
                if not ok:
                    kill_process_tree(proc)
                    continue
                agg = quality_eval(base, args.dataset, args.k)
                kill_process_tree(proc)
                if 'error' in agg:
                    history.append({'params': trial, 'error': agg['error']})
                    continue
                sc = score_tuple(agg)
                history.append({'params': trial, 'aggregate': agg, 'score': sc})
                if (best_score is None) or (sc > best_score):
                    best_score = sc
                    best_val = val
            current[key] = best_val

    # Final evaluation with best params
    env = { k: (str(v) if not isinstance(v,bool) else ('1' if v else '0')) for k,v in current.items() }
    proc = start_server(env)
    wait_health(base, 30)
    final_agg = quality_eval(base, args.dataset, args.k)
    kill_process_tree(proc)

    result = {
        'best_params': current,
        'final_aggregate': final_agg,
        'history_count': len(history),
    }
    os.makedirs('reports', exist_ok=True)
    with open(REPORT_PATH,'w',encoding='utf-8') as f:
        json.dump({'history': history, 'result': result}, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # Print recommended env line
    env_line = ' '.join([f"{k}={v}" for k,v in current.items()])
    print('\nRecommended env:\n' + env_line)


if __name__ == '__main__':
    main()

