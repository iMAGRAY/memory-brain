#!/usr/bin/env python3
"""
Test script to measure search quality improvement with GPT-5-nano integration
"""

import asyncio
import aiohttp
import json
import time
import os
from typing import List, Dict

# Configuration from environment or defaults
API_HOST = os.getenv("MEMORY_API_HOST", "localhost")
API_PORT = os.getenv("MEMORY_API_PORT", "8080")
API_BASE = f"http://{API_HOST}:{API_PORT}"

# Test queries for measuring quality
test_queries = [
    {
        "query": "how to optimize memory usage in ai systems",
        "expected_topics": ["memory", "optimization", "ai", "embedding", "cache"]
    },
    {
        "query": "find information about user preferences and learning",
        "expected_topics": ["user", "preference", "learning", "distillation"]
    },
    {
        "query": "what are the best practices for vector search",
        "expected_topics": ["vector", "search", "cosine", "similarity", "embedding"]
    },
    {
        "query": "error handling and debugging techniques",
        "expected_topics": ["error", "debug", "log", "trace", "handle"]
    },
    {
        "query": "performance optimization strategies",
        "expected_topics": ["performance", "optimize", "cache", "speed", "efficiency"]
    }
]

async def test_search_quality(session: aiohttp.ClientSession, query_info: Dict) -> Dict:
    """Test search quality for a single query"""
    query = query_info["query"]
    expected = query_info["expected_topics"]
    
    # Perform search
    async with session.post(
        f"{API_BASE}/search",
        json={"query": query, "limit": 10},
        headers={"X-Client-Type": "api", "X-Session-Id": "test-session"}
    ) as response:
        if response.status != 200:
            print(f"❌ Search failed for '{query}': {response.status}")
            return {"query": query, "success": False, "score": 0}
        
        result = await response.json()
        
    # Analyze results
    memories = result.get("results", [])
    confidence = result.get("confidence", 0)
    
    # Calculate relevance score
    relevance_score = 0
    topic_matches = 0
    
    for memory in memories:
        content = memory.get("content", "").lower()
        importance = memory.get("importance", 0)
        
        # Check how many expected topics are present
        matches = sum(1 for topic in expected if topic in content)
        if matches > 0:
            topic_matches += matches
            relevance_score += importance * (matches / len(expected))
    
    # Normalize score
    if memories:
        avg_relevance = relevance_score / len(memories)
        topic_coverage = min(topic_matches / (len(expected) * 2), 1.0)  # Expect each topic to appear at least twice
    else:
        avg_relevance = 0
        topic_coverage = 0
    
    # Combined quality score
    quality_score = (avg_relevance * 0.3 + topic_coverage * 0.4 + confidence * 0.3) * 100
    
    return {
        "query": query,
        "success": True,
        "num_results": len(memories),
        "confidence": confidence,
        "relevance": avg_relevance,
        "topic_coverage": topic_coverage,
        "quality_score": quality_score
    }

async def measure_performance(session: aiohttp.ClientSession) -> Dict:
    """Measure search performance metrics"""
    query = "test performance measurement"
    times = []
    
    for _ in range(5):
        start = time.time()
        async with session.post(
            f"{API_BASE}/search",
            json={"query": query, "limit": 5},
            headers={"X-Client-Type": "api"}
        ) as response:
            await response.json()
        elapsed = (time.time() - start) * 1000  # Convert to ms
        times.append(elapsed)
    
    return {
        "avg_latency_ms": sum(times) / len(times),
        "min_latency_ms": min(times),
        "max_latency_ms": max(times)
    }

async def main():
    print("=" * 60)
    print("🧪 GPT-5-nano Integration Quality Test")
    print("=" * 60)
    print()
    
    async with aiohttp.ClientSession() as session:
        # Test search quality
        print("📊 Testing Search Quality with GPT-5 Enhancement...")
        print("-" * 40)
        
        quality_results = []
        for query_info in test_queries:
            result = await test_search_quality(session, query_info)
            quality_results.append(result)
            
            if result["success"]:
                print(f"✅ Query: '{result['query'][:40]}...'")
                print(f"   Results: {result['num_results']}, Confidence: {result['confidence']:.2f}")
                print(f"   Quality Score: {result['quality_score']:.1f}%")
            else:
                print(f"❌ Query failed: '{result['query'][:40]}...'")
            print()
        
        # Calculate overall quality
        successful_tests = [r for r in quality_results if r["success"]]
        if successful_tests:
            avg_quality = sum(r["quality_score"] for r in successful_tests) / len(successful_tests)
            avg_confidence = sum(r["confidence"] for r in successful_tests) / len(successful_tests)
            
            print("-" * 40)
            print("📈 Overall Quality Metrics:")
            print(f"   Average Quality Score: {avg_quality:.1f}%")
            print(f"   Average Confidence: {avg_confidence:.2f}")
            print(f"   Success Rate: {len(successful_tests)}/{len(quality_results)}")
            
            # Compare with baseline (11.7% from previous test)
            baseline = 11.7
            improvement = avg_quality - baseline
            improvement_pct = (improvement / baseline) * 100
            
            print()
            print("🎯 Improvement Analysis:")
            print(f"   Baseline Quality: {baseline:.1f}%")
            print(f"   Current Quality: {avg_quality:.1f}%")
            print(f"   Improvement: +{improvement:.1f}% ({improvement_pct:.0f}% better)")
            
            if avg_quality >= 70:
                print("   ✅ EXCELLENT: Target quality (70-85%) achieved!")
            elif avg_quality >= 50:
                print("   🟡 GOOD: Significant improvement, approaching target")
            elif avg_quality >= 30:
                print("   🟠 MODERATE: Some improvement, more tuning needed")
            else:
                print("   🔴 LIMITED: Minimal improvement, check integration")
        
        # Test performance
        print()
        print("-" * 40)
        print("⚡ Testing Performance...")
        perf_metrics = await measure_performance(session)
        print(f"   Average Latency: {perf_metrics['avg_latency_ms']:.1f}ms")
        print(f"   Min/Max: {perf_metrics['min_latency_ms']:.1f}ms / {perf_metrics['max_latency_ms']:.1f}ms")
        
        if perf_metrics['avg_latency_ms'] < 100:
            print("   ✅ Performance: Excellent")
        elif perf_metrics['avg_latency_ms'] < 500:
            print("   🟡 Performance: Good")
        else:
            print("   🔴 Performance: Needs optimization")
    
    print()
    print("=" * 60)
    print("✅ Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())