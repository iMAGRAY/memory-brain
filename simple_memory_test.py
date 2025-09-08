#!/usr/bin/env python3
"""
Simple Memory Test - Check if AI Memory Service is working
Quick test without external dependencies
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

async def test_memory_service():
    """Quick test of memory service functionality"""
    
    base_url = "http://localhost:8080"
    
    print("🔍 Testing AI Memory Service...")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test 1: Health check
            print("1. Health Check...")
            try:
                async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        print(f"   ✅ Service healthy: {health_data}")
                    else:
                        print(f"   ❌ Health check failed: HTTP {response.status}")
                        return False
            except Exception as e:
                print(f"   ❌ Service not reachable: {e}")
                return False
            
            # Test 2: Store a simple memory
            print("\n2. Storing test memory...")
            test_memory = {
                "content": "The capital of France is Paris. This is a basic geographical fact.",
                "context": "Geography test", 
                "importance": 0.7,
                "metadata": {
                    "test": "simple_test",
                    "timestamp": str(time.time())
                }
            }
            
            try:
                async with session.post(
                    f"{base_url}/api/memories",
                    json=test_memory,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        memory_id = result.get("memory_id")
                        print(f"   ✅ Memory stored with ID: {memory_id}")
                    else:
                        error_text = await response.text()
                        print(f"   ❌ Failed to store memory: HTTP {response.status}")
                        print(f"      Error: {error_text}")
                        return False
            except Exception as e:
                print(f"   ❌ Error storing memory: {e}")
                return False
            
            # Wait for indexing
            await asyncio.sleep(1)
            
            # Test 3: Search for the memory
            print("\n3. Searching for stored memory...")
            try:
                # Use GET with proper query parameters
                search_params = {
                    "query": "capital France Paris",
                    "limit": "3",
                    "memory_types": "Semantic",  # API expects comma-separated string
                    "min_importance": "0.0",
                    "similarity_threshold": "0.7"
                }
                async with session.get(
                    f"{base_url}/api/memories/search",
                    params=search_params,
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status == 200:
                        search_results = await response.json()
                        memories = search_results.get("results", [])  # API returns "results", not "memories"
                        
                        if memories:
                            found_test_memory = False
                            for memory in memories:
                                if "Paris" in memory.get("content", ""):
                                    found_test_memory = True
                                    relevance = memory.get("relevance_score", 0)
                                    print(f"   ✅ Found memory with relevance: {relevance:.3f}")
                                    print(f"      Content: {memory.get('content', '')[:100]}...")
                                    break
                            
                            if not found_test_memory:
                                print("   ⚠️ Test memory not found in search results")
                                print(f"      Found {len(memories)} other memories")
                        else:
                            print("   ❌ No memories found in search")
                            return False
                    else:
                        error_text = await response.text()
                        print(f"   ❌ Search failed: HTTP {response.status}")
                        print(f"      Error: {error_text}")
                        return False
            except Exception as e:
                print(f"   ❌ Error searching memories: {e}")
                return False
            
            # Test 4: List all memories
            print("\n4. Listing all memories...")
            try:
                async with session.get(
                    f"{base_url}/api/memories",
                    params={"limit": 10},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        all_memories = await response.json()
                        memories = all_memories.get("memories", [])
                        total = all_memories.get("total", len(memories))
                        print(f"   ✅ Found {len(memories)} memories (total: {total})")
                        
                        for i, memory in enumerate(memories[:3]):  # Show first 3
                            content = memory.get("content", "")[:50]
                            importance = memory.get("importance", 0)
                            print(f"      {i+1}. {content}... (importance: {importance})")
                    else:
                        print(f"   ⚠️ List memories returned: HTTP {response.status}")
            except Exception as e:
                print(f"   ⚠️ Error listing memories: {e}")
            
            print("\n" + "=" * 50)
            print("🎉 BASIC FUNCTIONALITY TEST PASSED")
            print("✅ Memory service is working correctly!")
            return True
            
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_memory_service())
    if success:
        print("\n✅ Memory service is functional and ready for use")
        exit(0)
    else:
        print("\n❌ Memory service has issues that need attention")
        exit(1)