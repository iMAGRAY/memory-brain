#!/usr/bin/env python3
"""
Lightweight mock Embedding Server for deterministic testing.
Implements the same API surface as embedding_server.py without ML deps.

Endpoints:
- GET  /health -> {status, service, model}
- GET  /stats  -> summary counters
- POST /embed  -> {embedding: [..], dimension}
- POST /embed_batch -> {embeddings: [[..],..], count, dimension}

Deterministic 512-dim embeddings are produced from blake2b(text) seed.
"""

import asyncio
import json
import os
import hashlib
from aiohttp import web
import numpy as np

DIM = int(os.getenv("MOCK_EMBEDDING_DIM", "512"))

def text_to_vec(text: str, dim: int = DIM) -> np.ndarray:
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()
    # expand seed into deterministic pseudo-random vector
    rng = np.random.default_rng(int.from_bytes(h, "little"))
    vec = rng.standard_normal(dim).astype(np.float32)
    # L2 normalize to mimic real embedding
    n = np.linalg.norm(vec) or 1.0
    return (vec / n).astype(np.float32)

class MockService:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.stats = {
            "total_requests": 0,
            "single_requests": 0,
            "batch_requests": 0,
        }

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({
            "status": "healthy",
            "service": "mock-embedding-server",
            "model": f"mock-dim-{self.dim}",
        })

    async def handle_stats(self, request: web.Request) -> web.Response:
        return web.json_response({
            **self.stats,
            "dimension": self.dim,
        })

    async def handle_embed(self, request: web.Request) -> web.Response:
        self.stats["total_requests"] += 1
        self.stats["single_requests"] += 1
        data = await request.json()
        text = data.get("text") or data.get("query") or ""
        vec = text_to_vec(text, self.dim)
        return web.json_response({
            "embedding": vec.tolist(),
            "dimension": int(self.dim),
        })

    async def handle_embed_batch(self, request: web.Request) -> web.Response:
        self.stats["total_requests"] += 1
        self.stats["batch_requests"] += 1
        data = await request.json()
        texts = data.get("texts") or []
        embs = [text_to_vec(t, self.dim).tolist() for t in texts]
        return web.json_response({
            "embeddings": embs,
            "count": len(embs),
            "dimension": int(self.dim),
        })

def main():
    svc = MockService(DIM)
    app = web.Application()
    app.router.add_get("/health", svc.handle_health)
    app.router.add_get("/stats", svc.handle_stats)
    app.router.add_post("/embed", svc.handle_embed)
    app.router.add_post("/embed_batch", svc.handle_embed_batch)
    port = int(os.getenv("EMBEDDING_SERVER_PORT", "8090"))
    web.run_app(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()

