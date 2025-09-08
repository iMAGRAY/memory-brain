#!/usr/bin/env python3
"""
Test script to validate the TaskType consistency fix
Tests the impact of changing storage TaskType from Document to Query
"""

import asyncio
import json
import time
import requests
import uuid
import logging
from typing import Dict, Any, List, Optional


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TaskTypeTestSuite:
    """Test suite for validating TaskType consistency improvements"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url
        self.session_id = str(uuid.uuid4())
        self.timeout = 10  # Request timeout in seconds
        
    def health_check(self) -> bool:
        """Check if the service is running with proper error handling"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("Health check successful")
                return True
            else:
                logger.warning(f"Health check failed with status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed due to request error: {e}")
            return False
        except Exception as e:
            logger.error(f"Health check failed due to unexpected error: {e}")
            return False
    
    def store_memory(self, content: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Store a memory and return sanitized response"""
        data = {
            "content": content,
            "context_hint": context,
            "metadata": {
                "test_type": "tasktype_validation",
                "timestamp": str(int(time.time())),
                "expected_tags": f"test,validation,{content[:20].replace(' ', '_')}"
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/memory",
                json=data,
                timeout=self.timeout
            )
            
            # Sanitize response to avoid data leakage
            success = response.status_code == 200
            result = {
                "status_code": response.status_code,
                "success": success
            }
            
            if success:
                try:
                    response_data = response.json()
                    # Only include safe fields, not full response
                    result["memory_id"] = response_data.get("memory_id", "unknown")
                    result["message"] = "Storage successful"
                except json.JSONDecodeError:
                    logger.warning("Failed to decode JSON response for successful storage")
                    result["message"] = "Storage successful but response not JSON"
            else:
                # Log error details but don't include in results
                logger.error(f"Storage failed with status {response.status_code}: {response.text[:200]}")
                result["message"] = f"Storage failed with status {response.status_code}"
                
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Storage request failed: {e}")
            return {
                "status_code": 500,
                "success": False,
                "message": f"Request error: {type(e).__name__}"
            }
        except Exception as e:
            logger.error(f"Unexpected error during storage: {e}")
            return {
                "status_code": 500,
                "success": False,
                "message": f"Unexpected error: {type(e).__name__}"
            }
    
    def search_memory(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for memories and return sanitized response"""
        try:
            response = requests.get(
                f"{self.base_url}/search",
                params={"query": query, "limit": limit},
                timeout=self.timeout
            )
            
            success = response.status_code == 200
            result = {
                "status_code": response.status_code,
                "success": success,
                "results_count": 0
            }
            
            if success:
                try:
                    response_data = response.json()
                    if isinstance(response_data, list):
                        result["results_count"] = len(response_data)
                        result["message"] = f"Found {len(response_data)} results"
                    else:
                        logger.warning("Search response is not a list as expected")
                        result["message"] = "Search successful but unexpected response format"
                except json.JSONDecodeError:
                    logger.warning("Failed to decode JSON response for successful search")
                    result["message"] = "Search successful but response not JSON"
            else:
                # Log error details but don't include in results
                logger.error(f"Search failed with status {response.status_code}: {response.text[:200]}")
                result["message"] = f"Search failed with status {response.status_code}"
                
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Search request failed: {e}")
            return {
                "status_code": 500,
                "success": False,
                "results_count": 0,
                "message": f"Request error: {type(e).__name__}"
            }
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return {
                "status_code": 500,
                "success": False,
                "results_count": 0,
                "message": f"Unexpected error: {type(e).__name__}"
            }
    
    async def run_consistency_test(self) -> Dict[str, Any]:
        """Run comprehensive TaskType consistency test"""
        
        logger.info("Starting TaskType Consistency Test")
        
        # Test data with high semantic similarity
        test_memories = [
            ("Python programming language syntax and features", "programming"),
            ("Python development best practices and conventions", "programming"), 
            ("JavaScript framework React component lifecycle", "web_development"),
            ("React hooks useState and useEffect patterns", "web_development"),
            ("Machine learning neural network architecture", "ai_ml"),
            ("Deep learning convolutional neural networks", "ai_ml"),
            ("Database SQL query optimization techniques", "database"),
            ("PostgreSQL performance tuning and indexing", "database"),
        ]
        
        # Search queries that should find similar memories
        test_queries = [
            "Python coding standards",
            "React component patterns", 
            "neural network design",
            "SQL optimization strategies"
        ]
        
        results = {
            "test_metadata": {
                "test_id": str(uuid.uuid4()),
                "timestamp": int(time.time()),
                "test_type": "tasktype_consistency"
            },
            "storage_results": [],
            "search_results": [],
            "consistency_metrics": {},
            "total_time": 0,
            "errors": []
        }
        
        start_time = time.time()
        
        # Phase 1: Store memories
        logger.info("Phase 1: Storing test memories")
        stored_count = 0
        
        for i, (content, context) in enumerate(test_memories, 1):
            logger.info(f"Storing memory {i}/8: {content[:40]}...")
            
            result = self.store_memory(content, context)
            
            # Store only essential information
            results["storage_results"].append({
                "memory_index": i,
                "content_preview": content[:40] + "..." if len(content) > 40 else content,
                "context": context,
                "success": result["success"],
                "status_code": result["status_code"],
                "message": result["message"]
            })
            
            if result["success"]:
                stored_count += 1
            else:
                results["errors"].append(f"Storage failed for memory {i}: {result['message']}")
            
            # Rate limiting
            await asyncio.sleep(0.5)
        
        # Phase 2: Search and validate consistency
        logger.info("Phase 2: Testing search consistency")
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"Searching {i}/4: {query}")
            
            result = self.search_memory(query, limit=5)
            
            results["search_results"].append({
                "query_index": i,
                "query": query,
                "success": result["success"],
                "status_code": result["status_code"],
                "results_count": result["results_count"],
                "message": result["message"]
            })
            
            if not result["success"]:
                results["errors"].append(f"Search failed for query {i}: {result['message']}")
            
            await asyncio.sleep(0.5)
        
        # Phase 3: Calculate consistency metrics
        logger.info("Phase 3: Calculating consistency metrics")
        
        successful_searches = len([r for r in results["search_results"] if r["success"]])
        total_results_found = sum(r["results_count"] for r in results["search_results"])
        
        results["consistency_metrics"] = {
            "storage_success_rate": stored_count / len(test_memories),
            "search_success_rate": successful_searches / len(test_queries),
            "avg_results_per_query": total_results_found / len(test_queries) if test_queries else 0,
            "total_memories_stored": stored_count,
            "total_searches_performed": len(test_queries),
            "total_results_found": total_results_found,
            "error_count": len(results["errors"])
        }
        
        results["total_time"] = time.time() - start_time
        
        logger.info("TaskType consistency test completed")
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted test results"""
        
        print("\n" + "=" * 60)
        print("🎯 TASKTYPE CONSISTENCY TEST RESULTS")
        print("=" * 60)
        
        metrics = results["consistency_metrics"]
        
        print(f"\n📈 Performance Metrics:")
        print(f"  Storage Success Rate:    {metrics['storage_success_rate']:.1%}")
        print(f"  Search Success Rate:     {metrics['search_success_rate']:.1%}")
        print(f"  Avg Results per Query:   {metrics['avg_results_per_query']:.1f}")
        print(f"  Total Test Time:         {results['total_time']:.2f}s")
        
        print(f"\n📊 Operational Stats:")
        print(f"  Memories Stored:         {metrics['total_memories_stored']}")
        print(f"  Searches Performed:      {metrics['total_searches_performed']}")
        print(f"  Total Results Found:     {metrics['total_results_found']}")
        print(f"  Errors Encountered:      {metrics['error_count']}")
        
        if results["errors"]:
            print(f"\n❌ Errors:")
            for i, error in enumerate(results["errors"], 1):
                print(f"  {i}. {error}")
        
        # Quality assessment
        storage_rate = metrics['storage_success_rate']
        search_rate = metrics['search_success_rate']
        avg_results = metrics['avg_results_per_query']
        
        print(f"\n🎯 Quality Assessment:")
        if storage_rate >= 0.8 and search_rate >= 0.8 and avg_results >= 1.5:
            print("  ✅ EXCELLENT - TaskType consistency fix is working well")
        elif storage_rate >= 0.6 and search_rate >= 0.6 and avg_results >= 1.0:
            print("  ⚠️  GOOD - Improvement detected, but room for optimization")
        else:
            print("  ❌ NEEDS_WORK - Limited improvement from TaskType fix")


async def main():
    """Main test execution"""
    
    logger.info("Initializing AI Memory Service TaskType Consistency Validation")
    
    test_suite = TaskTypeTestSuite()
    
    print("🚀 AI Memory Service - TaskType Consistency Validation")
    print("Testing the fix for TaskType::Document -> TaskType::Query change")
    
    # Health check
    if not test_suite.health_check():
        print("❌ Service not responding at http://127.0.0.1:8080")
        print("Please start the memory service first.")
        return
    
    print("✅ Service is running and responsive")
    
    # Run the test suite
    try:
        results = await test_suite.run_consistency_test()
        
        # Display results
        test_suite.print_results(results)
        
        # Save results to file
        timestamp = int(time.time())
        filename = f"tasktype_consistency_test_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {filename}")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"❌ Test execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())