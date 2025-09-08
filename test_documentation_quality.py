#!/usr/bin/env python3
"""
Test search quality on imported documentation
"""

import asyncio
import aiohttp
import time
import statistics
import os

# Configuration
API_HOST = os.getenv("MEMORY_API_HOST", "localhost")
API_PORT = os.getenv("MEMORY_API_PORT", "8080")
API_BASE = f"http://{API_HOST}:{API_PORT}"

# Test queries specific to imported documentation
test_queries = [
    "how to use Claude Code hooks for development",
    "EmbeddingGemma model configuration and optimization",
    "GPT-5 API best practices and integration",
    "Python dependency management and package selection",
    "TypeScript development mistakes to avoid",
    "Rust programming best practices and common errors",
    "Next.js framework optimization techniques",
    "slash commands implementation in Claude Code",
    "memory optimization in AI systems",
    "vector database integration strategies"
]

async def test_search_quality(session, query):
    """Test search with a specific query"""
    try:
        start_time = time.time()
        
        # Test search endpoint
        async with session.post(
            f"{API_BASE}/search",
            json={
                "query": query,
                "limit": 5
            },
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                result = await response.json()
                end_time = time.time()
                
                memories = result.get("memories", [])
                confidence = result.get("confidence", 0.0)
                
                # Calculate quality based on result count, confidence and relevance
                quality_score = 0
                if memories:
                    quality_score = min(100, len(memories) * 15 + confidence * 40)
                    
                    # Bonus for high confidence with multiple results
                    if confidence > 0.8 and len(memories) >= 3:
                        quality_score += 10
                    
                    # Check content relevance (basic keyword matching)
                    query_keywords = set(query.lower().split())
                    for memory in memories:
                        content = memory.get("content", "").lower()
                        matches = sum(1 for keyword in query_keywords if keyword in content)
                        if matches > 0:
                            quality_score += matches * 2
                
                return {
                    "query": query[:50] + "..." if len(query) > 50 else query,
                    "results": len(memories),
                    "confidence": confidence,
                    "quality": min(quality_score, 100),
                    "latency": (end_time - start_time) * 1000,
                    "memories": memories[:3]  # Show top 3 for analysis
                }
            else:
                return {"error": f"HTTP {response.status}", "query": query}
    except Exception as e:
        return {"error": str(e), "query": query}

async def main():
    print("=" * 60)
    print("📚 Documentation Search Quality Test")
    print("=" * 60)
    print()
    
    print("🔍 Testing search quality on imported documentation...")
    print("-" * 40)
    
    results = []
    latencies = []
    
    async with aiohttp.ClientSession() as session:
        # Test API availability
        try:
            async with session.get(f"{API_BASE}/health") as response:
                if response.status != 200:
                    print(f"⚠️  API health check failed: {response.status}")
                    return
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return
        
        # Test each query
        for query in test_queries:
            result = await test_search_quality(session, query)
            
            if "error" in result:
                print(f"❌ Query: '{result['query']}'")
                print(f"   Error: {result['error']}")
            else:
                results.append(result)
                latencies.append(result["latency"])
                
                print(f"✅ Query: '{result['query']}'")
                print(f"   Results: {result['results']}, Confidence: {result['confidence']:.2f}")
                print(f"   Quality Score: {result['quality']:.1f}%")
                
                # Show top result content snippet
                if result["memories"]:
                    top_memory = result["memories"][0]
                    content_snippet = top_memory.get("content", "")[:100]
                    context_path = top_memory.get("context_path", "unknown")
                    print(f"   Top result from: {context_path}")
                    print(f"   Snippet: {content_snippet}...")
                print()
    
    if results:
        # Calculate overall metrics
        qualities = [r["quality"] for r in results]
        confidences = [r["confidence"] for r in results]
        
        avg_quality = statistics.mean(qualities)
        avg_confidence = statistics.mean(confidences)
        avg_latency = statistics.mean(latencies)
        success_rate = len([r for r in results if r["quality"] > 30]) / len(results)
        
        print("-" * 40)
        print("📈 Overall Quality Metrics:")
        print(f"   Average Quality Score: {avg_quality:.1f}%")
        print(f"   Average Confidence: {avg_confidence:.2f}")
        print(f"   Success Rate: {len(results)}/{len(test_queries)}")
        print(f"   High Quality Results: {len([r for r in results if r['quality'] > 60])}/{len(results)}")
        print()
        
        print("🎯 Documentation Coverage Analysis:")
        if avg_quality > 60:
            print("   🟢 EXCELLENT: Great documentation coverage and relevance")
        elif avg_quality > 40:
            print("   🟡 GOOD: Solid documentation coverage, some gaps")
        elif avg_quality > 25:
            print("   🟠 MODERATE: Basic coverage, needs improvement")
        else:
            print("   🔴 POOR: Limited coverage, major improvements needed")
        print()
        
        print("-" * 40)
        print("⚡ Performance Metrics:")
        print(f"   Average Latency: {avg_latency:.1f}ms")
        print(f"   Min/Max: {min(latencies):.1f}ms / {max(latencies):.1f}ms")
        if avg_latency < 50:
            print("   ✅ Performance: Excellent")
        elif avg_latency < 200:
            print("   🟡 Performance: Good")
        else:
            print("   🟠 Performance: Needs optimization")
    
    print()
    print("=" * 60)
    print("✅ Documentation Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())