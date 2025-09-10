#!/usr/bin/env python3
import argparse, json, sys, time
from urllib import request

def http_get(url, timeout=8.0):
    with request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))

def http_post(url, payload, timeout=8.0):
    data = json.dumps(payload).encode('utf-8')
    req = request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))

def http_delete(url, timeout=8.0):
    req = request.Request(url, method='DELETE')
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--prefix', default='quality/', help='Context path prefix to purge')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    base = f'http://{args.host}:{args.port}'

    try:
        ctxs = http_get(f'{base}/contexts').get('contexts', [])
    except Exception as e:
        print(f'ERROR: failed to list contexts: {e}', file=sys.stderr)
        sys.exit(2)

    targets = [c for c in ctxs if c.startswith(args.prefix)]
    total = 0
    for c in targets:
        try:
            data = http_post(f'{base}/search/context', {'context': c, 'limit': 2000, 'query': ''})
            items = data.get('results', [])
        except Exception:
            items = []
        if args.dry_run:
            print(f'[dry-run] {c}: {len(items)} memories')
            total += len(items)
            continue
        for it in items:
            mid = it.get('id')
            if not mid:
                continue
            try:
                http_delete(f'{base}/memory/{mid}')
                total += 1
            except Exception:
                pass
            time.sleep(0.005)
        print(f'[purge] {c}: removed {len(items)} memories')
    print(json.dumps({'contexts': len(targets), 'removed': total}))

if __name__ == '__main__':
    main()

