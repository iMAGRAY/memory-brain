#!/usr/bin/env python3
"""
Comprehensive test suite for AI Memory Service
Tests all components and their integration
"""

import json
import time
import asyncio
import aiohttp
import sys
import os
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Test results collector
class TestResults:
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
    def add(self, component: str, test: str, status: str, details: str = ""):
        self.results.append({
            "component": component,
            "test": test,
            "status": status,
            "details": details,
            "timestamp": time.time() - self.start_time
        })
        
    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / total * 100) if total > 0 else 0
        }

# Test functions
async def test_embedding_server(results: TestResults):
    """Test embedding server functionality"""
    print("\n=== Testing Embedding Server ===")
    
    async with aiohttp.ClientSession() as session:
        # 1. Health check
        try:
            async with session.get("http://localhost:8090/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    results.add("embedding", "health_check", "PASS", f"Model: {data.get('model', 'unknown')}")
                    print("✓ Health check passed")
                else:
                    results.add("embedding", "health_check", "FAIL", f"Status: {resp.status}")
                    print(f"✗ Health check failed: {resp.status}")
        except Exception as e:
            results.add("embedding", "health_check", "FAIL", str(e))
            print(f"✗ Health check error: {e}")
            
        # 2. Single embedding
        try:
            payload = {
                "texts": ["Hello world"],
                "task_type": "document"
            }
            async with session.post("http://localhost:8090/embed", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings and len(embeddings[0]) == 512:
                        results.add("embedding", "single_embed", "PASS", "512-dim vector")
                        print("✓ Single embedding passed")
                    else:
                        results.add("embedding", "single_embed", "FAIL", f"Invalid shape: {len(embeddings[0]) if embeddings else 0}")
                        print(f"✗ Invalid embedding shape")
                else:
                    results.add("embedding", "single_embed", "FAIL", f"Status: {resp.status}")
                    print(f"✗ Single embedding failed: {resp.status}")
        except Exception as e:
            results.add("embedding", "single_embed", "FAIL", str(e))
            print(f"✗ Single embedding error: {e}")
            
        # 3. Batch embedding
        try:
            payload = {
                "texts": ["First text", "Second text", "Third text"],
                "task_type": "query"
            }
            async with session.post("http://localhost:8090/embed", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    embeddings = data.get("embeddings", [])
                    if len(embeddings) == 3:
                        results.add("embedding", "batch_embed", "PASS", "3 embeddings")
                        print("✓ Batch embedding passed")
                    else:
                        results.add("embedding", "batch_embed", "FAIL", f"Got {len(embeddings)} embeddings")
                        print(f"✗ Batch embedding count mismatch")
                else:
                    results.add("embedding", "batch_embed", "FAIL", f"Status: {resp.status}")
                    print(f"✗ Batch embedding failed: {resp.status}")
        except Exception as e:
            results.add("embedding", "batch_embed", "FAIL", str(e))
            print(f"✗ Batch embedding error: {e}")
            
        # 4. Similarity test
        try:
            payload = {
                "texts": ["cat", "dog", "airplane"],
                "task_type": "document"
            }
            async with session.post("http://localhost:8090/embed", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    embeddings = data.get("embeddings", [])
                    if len(embeddings) == 3:
                        # Calculate cosine similarities
                        cat = np.array(embeddings[0])
                        dog = np.array(embeddings[1])
                        plane = np.array(embeddings[2])
                        
                        sim_cat_dog = np.dot(cat, dog) / (np.linalg.norm(cat) * np.linalg.norm(dog))
                        sim_cat_plane = np.dot(cat, plane) / (np.linalg.norm(cat) * np.linalg.norm(plane))
                        
                        # Animals should be more similar than animal-vehicle
                        if sim_cat_dog > sim_cat_plane:
                            results.add("embedding", "similarity", "PASS", f"cat-dog: {sim_cat_dog:.3f}, cat-plane: {sim_cat_plane:.3f}")
                            print(f"✓ Similarity test passed (cat-dog: {sim_cat_dog:.3f} > cat-plane: {sim_cat_plane:.3f})")
                        else:
                            results.add("embedding", "similarity", "FAIL", "Unexpected similarities")
                            print("✗ Similarity test failed")
                    else:
                        results.add("embedding", "similarity", "FAIL", "Invalid embeddings")
                        print("✗ Similarity test: invalid embeddings")
                else:
                    results.add("embedding", "similarity", "FAIL", f"Status: {resp.status}")
                    print(f"✗ Similarity test failed: {resp.status}")
        except Exception as e:
            results.add("embedding", "similarity", "FAIL", str(e))
            print(f"✗ Similarity test error: {e}")

async def test_memory_service(results: TestResults):
    """Test memory service functionality"""
    print("\n=== Testing Memory Service ===")
    
    async with aiohttp.ClientSession() as session:
        # 1. Health check
        try:
            async with session.get("http://localhost:3030/health") as resp:
                if resp.status == 200:
                    results.add("memory", "health_check", "PASS")
                    print("✓ Memory service health check passed")
                else:
                    results.add("memory", "health_check", "FAIL", f"Status: {resp.status}")
                    print(f"✗ Memory service not available: {resp.status}")
                    return
        except Exception as e:
            results.add("memory", "health_check", "FAIL", str(e))
            print(f"✗ Memory service not running: {e}")
            return
            
        # 2. Store memory
        test_memory = {
            "content": f"Test memory created at {datetime.now().isoformat()}",
            "metadata": {
                "type": "test",
                "importance": 0.8
            }
        }
        
        try:
            async with session.post("http://localhost:3030/api/memory", json=test_memory) as resp:
                if resp.status in [200, 201]:
                    data = await resp.json()
                    memory_id = data.get("id")
                    results.add("memory", "store", "PASS", f"ID: {memory_id}")
                    print(f"✓ Memory stored: {memory_id}")
                else:
                    results.add("memory", "store", "FAIL", f"Status: {resp.status}")
                    print(f"✗ Failed to store memory: {resp.status}")
                    return
        except Exception as e:
            results.add("memory", "store", "FAIL", str(e))
            print(f"✗ Store memory error: {e}")
            return
            
        # 3. Search memories
        try:
            search_query = {
                "query": "test memory",
                "limit": 10
            }
            async with session.post("http://localhost:3030/api/memory/search", json=search_query) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    memories = data.get("memories", [])
                    results.add("memory", "search", "PASS", f"Found {len(memories)} memories")
                    print(f"✓ Search passed: found {len(memories)} memories")
                else:
                    results.add("memory", "search", "FAIL", f"Status: {resp.status}")
                    print(f"✗ Search failed: {resp.status}")
        except Exception as e:
            results.add("memory", "search", "FAIL", str(e))
            print(f"✗ Search error: {e}")

async def test_rust_services(results: TestResults):
    """Test Rust-based services"""
    print("\n=== Testing Rust Services ===")
    
    # Check if Rust binaries exist
    binaries = [
        ("memory-server", "target/release/memory-server.exe"),
        ("orchestrator", "target/release/orchestrator.exe"),
        ("memory_organizer", "target/release/memory_organizer.exe")
    ]
    
    for name, path in binaries:
        if os.path.exists(path):
            results.add("rust", f"{name}_binary", "PASS", "Binary exists")
            print(f"✓ {name} binary found")
        else:
            results.add("rust", f"{name}_binary", "FAIL", "Binary not found")
            print(f"✗ {name} binary not found at {path}")

async def test_integrations(results: TestResults):
    """Test component integrations"""
    print("\n=== Testing Integrations ===")
    
    async with aiohttp.ClientSession() as session:
        # Test embedding + memory integration
        try:
            # First embed a text
            embed_payload = {
                "texts": ["Integration test document"],
                "task_type": "document"
            }
            async with session.post("http://localhost:8090/embed", json=embed_payload) as resp:
                if resp.status == 200:
                    embed_data = await resp.json()
                    embedding = embed_data.get("embeddings", [[]])[0]
                    
                    # Store with embedding
                    memory_payload = {
                        "content": "Integration test memory",
                        "embedding": embedding,
                        "metadata": {"test": "integration"}
                    }
                    
                    async with session.post("http://localhost:3030/api/memory", json=memory_payload) as mem_resp:
                        if mem_resp.status in [200, 201]:
                            results.add("integration", "embed_store", "PASS")
                            print("✓ Embedding + storage integration passed")
                        else:
                            results.add("integration", "embed_store", "FAIL", f"Store failed: {mem_resp.status}")
                            print(f"✗ Failed to store embedded memory: {mem_resp.status}")
                else:
                    results.add("integration", "embed_store", "FAIL", f"Embed failed: {resp.status}")
                    print(f"✗ Embedding failed: {resp.status}")
        except Exception as e:
            results.add("integration", "embed_store", "FAIL", str(e))
            print(f"✗ Integration test error: {e}")

async def main():
    """Run all tests"""
    print("=" * 60)
    print("AI MEMORY SERVICE - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    results = TestResults()
    
    # Run all test suites
    await test_embedding_server(results)
    await test_memory_service(results)
    await test_rust_services(results)
    await test_integrations(results)
    
    # Generate report
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    summary = results.summary()
    print(f"Total tests: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    
    # Save detailed report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "details": results.results,
        "duration": time.time() - results.start_time
    }
    
    with open("comprehensive_test_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to comprehensive_test_results.json")
    
    # Component status
    print("\n" + "=" * 60)
    print("COMPONENT STATUS")
    print("=" * 60)
    
    components = {}
    for result in results.results:
        comp = result["component"]
        if comp not in components:
            components[comp] = {"passed": 0, "failed": 0}
        if result["status"] == "PASS":
            components[comp]["passed"] += 1
        else:
            components[comp]["failed"] += 1
    
    for comp, stats in components.items():
        total = stats["passed"] + stats["failed"]
        status = "✓ WORKING" if stats["failed"] == 0 else "✗ ISSUES" if stats["passed"] > 0 else "✗ FAILED"
        print(f"{comp.upper()}: {status} ({stats['passed']}/{total} tests passed)")
    
    return summary["success_rate"] >= 70  # Consider 70% as minimum acceptable

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)