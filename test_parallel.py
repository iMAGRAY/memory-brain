#!/usr/bin/env python3
"""
Comprehensive parallel testing for AI Memory Service
Tests performance, reliability, and functionality under concurrent load
"""

import asyncio
import aiohttp
import time
import json
from typing import List, Dict, Any
import statistics

class MemoryServiceTester:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.results = []
    
    async def test_store_memory(self, session: aiohttp.ClientSession, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test storing a memory with timing"""
        start_time = time.time()
        try:
            async with session.post(f"{self.base_url}/memory", json=memory_data) as resp:
                end_time = time.time()
                result = await resp.json()
                return {
                    "success": resp.status == 200,
                    "status_code": resp.status,
                    "response_time": end_time - start_time,
                    "memory_id": result.get("memory_id") if resp.status == 200 else None,
                    "error": result.get("error") if resp.status != 200 else None
                }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "status_code": 0,
                "response_time": end_time - start_time,
                "memory_id": None,
                "error": str(e)
            }
    
    async def test_search_memory(self, session: aiohttp.ClientSession, query: str) -> Dict[str, Any]:
        """Test searching memories with timing"""
        start_time = time.time()
        try:
            params = {"query": query, "limit": 5}
            async with session.get(f"{self.base_url}/search", params=params) as resp:
                end_time = time.time()
                result = await resp.json()
                return {
                    "success": resp.status == 200,
                    "status_code": resp.status,
                    "response_time": end_time - start_time,
                    "results_count": len(result.get("memories", [])) if resp.status == 200 else 0,
                    "error": result.get("error") if resp.status != 200 else None
                }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "status_code": 0,
                "response_time": end_time - start_time,
                "results_count": 0,
                "error": str(e)
            }
    
    async def test_batch_parallel_stores(self, num_parallel: int = 10) -> Dict[str, Any]:
        """Test parallel memory storage operations"""
        print(f"🔄 Testing {num_parallel} parallel memory stores...")
        
        test_memories = []
        for i in range(num_parallel):
            test_memories.append({
                "content": f"Test memory {i}: This is a test memory for parallel processing validation. It contains important information about system performance and reliability testing.",
                "context": f"parallel_test_context_{i}",
                "importance": 0.7 + (i % 3) * 0.1,
                "memory_type": "Semantic",
                "tags": [f"test_{i}", "performance", "parallel"]
            })
        
        connector = aiohttp.TCPConnector(limit=20)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.test_store_memory(session, memory) for memory in test_memories]
            results = await asyncio.gather(*tasks)
        
        # Analyze results
        success_count = sum(1 for r in results if r["success"])
        response_times = [r["response_time"] for r in results if r["success"]]
        
        return {
            "total_requests": num_parallel,
            "successful_requests": success_count,
            "failed_requests": num_parallel - success_count,
            "success_rate": success_count / num_parallel * 100,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "errors": [r["error"] for r in results if r["error"]]
        }
    
    async def test_batch_parallel_searches(self, num_parallel: int = 10) -> Dict[str, Any]:
        """Test parallel memory search operations"""
        print(f"🔍 Testing {num_parallel} parallel memory searches...")
        
        search_queries = [
            "machine learning algorithms",
            "database performance optimization", 
            "parallel processing techniques",
            "test memory validation",
            "system reliability testing",
            "artificial intelligence concepts",
            "software engineering practices",
            "memory management strategies",
            "concurrent programming patterns",
            "data processing efficiency"
        ]
        
        queries = [search_queries[i % len(search_queries)] for i in range(num_parallel)]
        
        connector = aiohttp.TCPConnector(limit=20)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self.test_search_memory(session, query) for query in queries]
            results = await asyncio.gather(*tasks)
        
        # Analyze results
        success_count = sum(1 for r in results if r["success"])
        response_times = [r["response_time"] for r in results if r["success"]]
        total_results = sum(r["results_count"] for r in results if r["success"])
        
        return {
            "total_requests": num_parallel,
            "successful_requests": success_count,
            "failed_requests": num_parallel - success_count,
            "success_rate": success_count / num_parallel * 100,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "total_search_results": total_results,
            "avg_results_per_query": total_results / success_count if success_count > 0 else 0,
            "errors": [r["error"] for r in results if r["error"]]
        }
    
    async def test_mixed_workload(self, store_count: int = 5, search_count: int = 5) -> Dict[str, Any]:
        """Test mixed workload of stores and searches"""
        print(f"🔀 Testing mixed workload: {store_count} stores + {search_count} searches...")
        
        # Prepare store operations
        store_memories = []
        for i in range(store_count):
            store_memories.append({
                "content": f"Mixed workload test memory {i}: Advanced testing of concurrent operations with both storage and retrieval patterns.",
                "context": f"mixed_workload_context_{i}",
                "importance": 0.6 + (i % 4) * 0.1,
                "memory_type": "Semantic",
                "tags": ["mixed_test", "workload", f"batch_{i}"]
            })
        
        # Prepare search queries
        search_queries = [
            "advanced testing patterns",
            "concurrent operations",
            "mixed workload scenarios",
            "storage retrieval efficiency",
            "system performance analysis"
        ]
        queries = [search_queries[i % len(search_queries)] for i in range(search_count)]
        
        connector = aiohttp.TCPConnector(limit=20)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Mix store and search operations
            all_tasks = []
            
            # Add store tasks
            for memory in store_memories:
                all_tasks.append(("store", self.test_store_memory(session, memory)))
            
            # Add search tasks  
            for query in queries:
                all_tasks.append(("search", self.test_search_memory(session, query)))
            
            # Execute all tasks concurrently
            task_types, tasks = zip(*all_tasks)
            results = await asyncio.gather(*tasks)
            
            # Separate results by type
            store_results = [r for i, r in enumerate(results) if task_types[i] == "store"]
            search_results = [r for i, r in enumerate(results) if task_types[i] == "search"]
        
        return {
            "store_operations": {
                "total": len(store_results),
                "successful": sum(1 for r in store_results if r["success"]),
                "avg_response_time": statistics.mean([r["response_time"] for r in store_results if r["success"]]) if any(r["success"] for r in store_results) else 0
            },
            "search_operations": {
                "total": len(search_results),
                "successful": sum(1 for r in search_results if r["success"]),
                "avg_response_time": statistics.mean([r["response_time"] for r in search_results if r["success"]]) if any(r["success"] for r in search_results) else 0,
                "total_results": sum(r["results_count"] for r in search_results if r["success"])
            },
            "overall": {
                "total_operations": len(results),
                "successful_operations": sum(1 for r in results if r["success"]),
                "success_rate": sum(1 for r in results if r["success"]) / len(results) * 100,
                "errors": [r["error"] for r in results if r["error"]]
            }
        }

    async def check_system_health(self) -> Dict[str, Any]:
        """Check system health and status"""
        print("🏥 Checking system health...")
        
        try:
            connector = aiohttp.TCPConnector(limit=5)
            async with aiohttp.ClientSession(connector=connector) as session:
                # Check memory service health
                async with session.get(f"{self.base_url}/health") as resp:
                    health_data = await resp.json() if resp.status == 200 else {}
                
                # Check embedding server health  
                async with session.get("http://localhost:8090/health") as resp:
                    embed_health = await resp.json() if resp.status == 200 else {}
                
                return {
                    "memory_service": {
                        "status": "healthy" if resp.status == 200 else "unhealthy",
                        "total_memories": health_data.get("memory_stats", {}).get("total_memories", 0),
                        "orchestrator_available": health_data.get("orchestrator", {}).get("available", False)
                    },
                    "embedding_service": {
                        "status": "healthy" if embed_health.get("status") == "healthy" else "unhealthy",
                        "model": embed_health.get("model", "unknown")
                    }
                }
        except Exception as e:
            return {"error": str(e)}

    async def run_comprehensive_test(self):
        """Run complete test suite"""
        print("🚀 Starting comprehensive AI Memory Service testing...")
        print("=" * 60)
        
        # 1. System health check
        health = await self.check_system_health()
        print(f"📊 System Health:")
        print(f"   Memory Service: {health.get('memory_service', {}).get('status', 'unknown')}")
        print(f"   Embedding Service: {health.get('embedding_service', {}).get('status', 'unknown')}")
        print(f"   Total Memories: {health.get('memory_service', {}).get('total_memories', 0)}")
        print(f"   Orchestrator: {'✅' if health.get('memory_service', {}).get('orchestrator_available') else '❌'}")
        print()
        
        # 2. Parallel store testing
        store_results = await self.test_batch_parallel_stores(10)
        print(f"📝 Parallel Memory Storage (10 concurrent):")
        print(f"   Success Rate: {store_results['success_rate']:.1f}%")
        print(f"   Avg Response Time: {store_results['avg_response_time']:.3f}s")
        print(f"   Median Response Time: {store_results['median_response_time']:.3f}s")
        print(f"   Min/Max: {store_results['min_response_time']:.3f}s / {store_results['max_response_time']:.3f}s")
        if store_results['errors']:
            print(f"   Errors: {len(store_results['errors'])}")
        print()
        
        # 3. Parallel search testing
        search_results = await self.test_batch_parallel_searches(10)
        print(f"🔍 Parallel Memory Search (10 concurrent):")
        print(f"   Success Rate: {search_results['success_rate']:.1f}%")
        print(f"   Avg Response Time: {search_results['avg_response_time']:.3f}s")
        print(f"   Total Search Results: {search_results['total_search_results']}")
        print(f"   Avg Results per Query: {search_results['avg_results_per_query']:.1f}")
        if search_results['errors']:
            print(f"   Errors: {len(search_results['errors'])}")
        print()
        
        # 4. Mixed workload testing
        mixed_results = await self.test_mixed_workload(7, 8)
        print(f"🔀 Mixed Workload Testing (7 stores + 8 searches):")
        print(f"   Overall Success Rate: {mixed_results['overall']['success_rate']:.1f}%")
        print(f"   Store Operations: {mixed_results['store_operations']['successful']}/{mixed_results['store_operations']['total']} successful")
        print(f"   Search Operations: {mixed_results['search_operations']['successful']}/{mixed_results['search_operations']['total']} successful")
        print(f"   Store Avg Time: {mixed_results['store_operations']['avg_response_time']:.3f}s")
        print(f"   Search Avg Time: {mixed_results['search_operations']['avg_response_time']:.3f}s")
        print()
        
        # 5. Final system check
        final_health = await self.check_system_health()
        print(f"📊 Final System State:")
        print(f"   Total Memories: {final_health.get('memory_service', {}).get('total_memories', 0)}")
        print()
        
        # Summary
        overall_success = (
            store_results['success_rate'] >= 95 and
            search_results['success_rate'] >= 95 and
            mixed_results['overall']['success_rate'] >= 95
        )
        
        print("=" * 60)
        print(f"🎯 TEST SUMMARY:")
        print(f"   Overall System Status: {'✅ EXCELLENT' if overall_success else '⚠️ NEEDS ATTENTION'}")
        print(f"   Parallel Performance: {'✅ GOOD' if store_results['avg_response_time'] < 2.0 else '⚠️ SLOW'}")
        print(f"   Search Effectiveness: {'✅ GOOD' if search_results['total_search_results'] > 0 else '❌ POOR'}")
        print(f"   System Reliability: {'✅ STABLE' if not any([store_results['errors'], search_results['errors']]) else '⚠️ UNSTABLE'}")
        
        return {
            "health": health,
            "parallel_stores": store_results,
            "parallel_searches": search_results,
            "mixed_workload": mixed_results,
            "final_health": final_health,
            "overall_success": overall_success
        }

async def main():
    """Main test execution"""
    tester = MemoryServiceTester()
    results = await tester.run_comprehensive_test()
    
    # Save results to file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to test_results.json")

if __name__ == "__main__":
    asyncio.run(main())