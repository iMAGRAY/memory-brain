#!/usr/bin/env python3
"""
Quick test to check if memory server now finds results after embedding fix
"""

import asyncio
import aiohttp
import json

async def test_memory_search():
    """Test that the memory server can now find results"""
    print("🔍 Testing Memory Server Search after Embedding Fix...")
    print("=" * 60)
    
    test_queries = [
        "Claude Code hooks for development",
        "GPT-5 API integration",  
        "memory optimization techniques",
        "vector database strategies",
        "embedding models best practices"
    ]
    
    async with aiohttp.ClientSession() as session:
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing query: '{query}'")
            print("-" * 40)
            
            try:
                async with session.post(
                    "http://localhost:8080/search",
                    json={"query": query, "limit": 5},
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 200:
                        print(f"❌ Search failed: HTTP {response.status}")
                        continue
                        
                    result = await response.json()
                    memories = result.get("memories", [])
                    confidence = result.get("confidence", 0)
                    
                    print(f"📊 Results: {len(memories)} memories found")
                    print(f"📊 Confidence: {confidence:.3f}")
                    
                    if len(memories) > 0:
                        print("✅ SUCCESS: Found results!")
                        # Show first result
                        top_memory = memories[0]
                        content = top_memory.get("content", "")
                        score = top_memory.get("score", 0)
                        context_path = top_memory.get("context_path", "unknown")
                        
                        print(f"   🥇 Top result (score: {score:.3f}):")
                        print(f"      Source: {context_path}")
                        print(f"      Content: {content[:100]}...")
                        
                        # Show other results briefly
                        if len(memories) > 1:
                            print(f"   📝 Other results:")
                            for j, mem in enumerate(memories[1:], 2):
                                score = mem.get("score", 0)
                                content = mem.get("content", "")
                                print(f"      {j}. Score: {score:.3f} - {content[:50]}...")
                    else:
                        print("❌ No results found")
                        
            except Exception as e:
                print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Memory Search Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_memory_search())