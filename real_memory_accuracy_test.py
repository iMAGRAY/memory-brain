#!/usr/bin/env python3
"""
Real Memory Accuracy Test - AI Memory Service
Tests actual memory storage, retrieval, and accuracy with realistic scenarios
"""

import asyncio
import json
import time
import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import aiohttp
import numpy as np
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MemoryTestCase:
    """Test case for memory validation"""
    id: str
    content: str
    context: str
    importance: float
    expected_tags: List[str]
    related_memories: List[str] = None

@dataclass 
class AccuracyResult:
    """Result of accuracy testing"""
    test_id: str
    stored_successfully: bool
    retrieved_successfully: bool
    content_accuracy: float
    semantic_similarity: float
    tag_accuracy: float
    retrieval_time_ms: float
    errors: List[str]

class RealMemoryAccuracyTester:
    """Tests real accuracy of memory service with comprehensive scenarios"""
    
    def __init__(self, service_url: str = "http://localhost:8080"):
        self.service_url = service_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.test_results: List[AccuracyResult] = []
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def generate_realistic_test_cases(self) -> List[MemoryTestCase]:
        """Generate realistic test scenarios for memory validation"""
        return [
            # Technical knowledge
            MemoryTestCase(
                id="tech_001",
                content="Python asyncio provides coroutines for concurrent programming. Use async/await syntax with event loops.",
                context="Programming discussion about concurrency patterns",
                importance=0.8,
                expected_tags=["python", "asyncio", "concurrency", "programming"]
            ),
            
            # Personal conversation
            MemoryTestCase(
                id="personal_001", 
                content="User mentioned they prefer working from home on Tuesdays and Thursdays, but likes office collaboration on Mondays.",
                context="Work preferences discussion",
                importance=0.6,
                expected_tags=["work", "schedule", "preferences", "remote"]
            ),
            
            # Project information
            MemoryTestCase(
                id="project_001",
                content="The AI Memory Service uses EmbeddingGemma-300M model with 768-dimensional vectors for semantic search and Neo4j for graph storage.",
                context="Architecture overview",
                importance=0.9,
                expected_tags=["architecture", "embedding", "neo4j", "ai-memory-service"]
            ),
            
            # Complex technical concept
            MemoryTestCase(
                id="complex_001",
                content="Transformer attention mechanism computes attention weights using Q, K, V matrices where Attention(Q,K,V) = softmax(QK^T/√d_k)V. This allows the model to focus on relevant parts of the input sequence.",
                context="Deep learning architecture explanation",
                importance=0.85,
                expected_tags=["transformer", "attention", "deep-learning", "neural-networks"]
            ),
            
            # Temporal information  
            MemoryTestCase(
                id="temporal_001",
                content="Meeting scheduled for next Monday at 2 PM to discuss quarterly results and budget planning for Q4 2024.",
                context="Calendar and planning",
                importance=0.7,
                expected_tags=["meeting", "schedule", "budget", "q4-2024"]
            ),
            
            # Factual information
            MemoryTestCase(
                id="fact_001",
                content="The speed of light in vacuum is exactly 299,792,458 meters per second. This is a fundamental physical constant.",
                context="Physics discussion",
                importance=0.5,
                expected_tags=["physics", "constants", "light-speed", "science"]
            ),
            
            # Procedural knowledge
            MemoryTestCase(
                id="procedure_001",
                content="To deploy the service: 1) Build with cargo build --release, 2) Start Neo4j database, 3) Set environment variables, 4) Run ./target/release/memory-server",
                context="Deployment instructions",
                importance=0.75,
                expected_tags=["deployment", "instructions", "cargo", "neo4j"]
            ),
            
            # Emotional/subjective content
            MemoryTestCase(
                id="subjective_001",
                content="User expressed frustration with slow API responses and mentioned it impacts their workflow significantly. They emphasized this is a high priority issue.",
                context="User feedback",
                importance=0.8,
                expected_tags=["feedback", "performance", "priority", "frustration"]
            ),
            
            # Multi-lingual content
            MemoryTestCase(
                id="multilingual_001",
                content="The French phrase 'l'art de vivre' means the art of living well. It encompasses enjoying life's pleasures and maintaining balance.",
                context="Cultural discussion",
                importance=0.4,
                expected_tags=["french", "culture", "philosophy", "lifestyle"]
            ),
            
            # Code snippet
            MemoryTestCase(
                id="code_001",
                content="Rust ownership prevents memory leaks: let data = vec![1, 2, 3]; let reference = &data[0]; // data can't be moved while reference exists",
                context="Programming concepts",
                importance=0.7,
                expected_tags=["rust", "ownership", "memory-safety", "programming"]
            )
        ]
    
    async def store_memory(self, test_case: MemoryTestCase) -> Tuple[bool, List[str]]:
        """Store a memory and return success status with errors"""
        try:
            payload = {
                "content": test_case.content,
                "context": test_case.context,
                "importance": test_case.importance,
                "metadata": {
                    "test_id": test_case.id,
                    "expected_tags": test_case.expected_tags,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            async with self.session.post(
                f"{self.service_url}/api/memories",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    logger.info(f"✅ Stored memory {test_case.id}: {result.get('memory_id', 'unknown')}")
                    return True, []
                else:
                    error = f"HTTP {response.status}: {await response.text()}"
                    logger.error(f"❌ Failed to store {test_case.id}: {error}")
                    return False, [error]
                    
        except Exception as e:
            error = f"Exception storing memory: {str(e)}"
            logger.error(f"❌ {test_case.id}: {error}")
            return False, [error]
    
    async def retrieve_memory(self, query: str, test_id: str) -> Tuple[bool, Dict, float, List[str]]:
        """Retrieve memories and measure accuracy"""
        try:
            start_time = time.time()
            
            async with self.session.get(
                f"{self.service_url}/api/memories/search",
                params={"query": query, "limit": 5},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                retrieval_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    result = await response.json()
                    memories = result.get("memories", [])
                    
                    # Find our test memory
                    target_memory = None
                    for memory in memories:
                        metadata = memory.get("metadata", {})
                        if metadata.get("test_id") == test_id:
                            target_memory = memory
                            break
                    
                    if target_memory:
                        logger.info(f"✅ Retrieved memory {test_id} in {retrieval_time:.1f}ms")
                        return True, target_memory, retrieval_time, []
                    else:
                        error = f"Memory {test_id} not found in search results"
                        logger.warning(f"⚠️ {error}")
                        return False, {}, retrieval_time, [error]
                else:
                    error = f"HTTP {response.status}: {await response.text()}"
                    return False, {}, 0, [error]
                    
        except Exception as e:
            error = f"Exception retrieving memory: {str(e)}"
            logger.error(f"❌ {test_id}: {error}")
            return False, {}, 0, [error]
    
    def calculate_content_accuracy(self, original: str, retrieved: str) -> float:
        """Calculate content accuracy using simple similarity"""
        if not retrieved:
            return 0.0
        
        # Simple word-based similarity
        orig_words = set(original.lower().split())
        retr_words = set(retrieved.lower().split())
        
        if not orig_words:
            return 1.0 if not retr_words else 0.0
        
        intersection = orig_words.intersection(retr_words)
        union = orig_words.union(retr_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_tag_accuracy(self, expected: List[str], retrieved_memory: Dict) -> float:
        """Calculate tag accuracy based on content analysis"""
        if not expected:
            return 1.0
        
        # Extract potential tags from retrieved memory
        content = retrieved_memory.get("content", "").lower()
        context = retrieved_memory.get("context", "").lower()
        combined_text = f"{content} {context}"
        
        matches = 0
        for tag in expected:
            if tag.lower() in combined_text or any(part in combined_text for part in tag.split("-")):
                matches += 1
        
        return matches / len(expected) if expected else 1.0
    
    async def run_accuracy_test(self, test_case: MemoryTestCase) -> AccuracyResult:
        """Run complete accuracy test for a single memory"""
        logger.info(f"🧪 Testing memory accuracy for {test_case.id}")
        
        # Store the memory
        stored_ok, store_errors = await self.store_memory(test_case)
        
        if not stored_ok:
            return AccuracyResult(
                test_id=test_case.id,
                stored_successfully=False,
                retrieved_successfully=False,
                content_accuracy=0.0,
                semantic_similarity=0.0,
                tag_accuracy=0.0,
                retrieval_time_ms=0.0,
                errors=store_errors
            )
        
        # Wait a moment for indexing
        await asyncio.sleep(0.5)
        
        # Test retrieval with various queries
        queries = [
            test_case.content[:50] + "...",  # Partial content
            " ".join(test_case.expected_tags[:2]),  # Tag-based query
            test_case.context,  # Context-based query
        ]
        
        best_result = None
        best_accuracy = 0.0
        
        for query in queries:
            retrieved_ok, memory_data, retrieval_time, retrieve_errors = await self.retrieve_memory(query, test_case.id)
            
            if retrieved_ok:
                content_accuracy = self.calculate_content_accuracy(
                    test_case.content, 
                    memory_data.get("content", "")
                )
                
                if content_accuracy > best_accuracy:
                    best_accuracy = content_accuracy
                    best_result = (retrieved_ok, memory_data, retrieval_time, retrieve_errors, content_accuracy)
        
        if best_result:
            retrieved_ok, memory_data, retrieval_time, retrieve_errors, content_accuracy = best_result
            
            tag_accuracy = self.calculate_tag_accuracy(test_case.expected_tags, memory_data)
            
            # Simple semantic similarity (placeholder)
            semantic_similarity = content_accuracy * 0.9  # Approximation
            
            result = AccuracyResult(
                test_id=test_case.id,
                stored_successfully=stored_ok,
                retrieved_successfully=retrieved_ok,
                content_accuracy=content_accuracy,
                semantic_similarity=semantic_similarity,
                tag_accuracy=tag_accuracy,
                retrieval_time_ms=retrieval_time,
                errors=retrieve_errors
            )
        else:
            result = AccuracyResult(
                test_id=test_case.id,
                stored_successfully=stored_ok,
                retrieved_successfully=False,
                content_accuracy=0.0,
                semantic_similarity=0.0,
                tag_accuracy=0.0,
                retrieval_time_ms=0.0,
                errors=["Memory not retrieved with any query"]
            )
        
        logger.info(f"📊 {test_case.id}: Content={result.content_accuracy:.2f}, Tags={result.tag_accuracy:.2f}, Time={result.retrieval_time_ms:.1f}ms")
        return result
    
    async def check_service_health(self) -> Tuple[bool, str]:
        """Check if memory service is running and healthy"""
        try:
            async with self.session.get(
                f"{self.service_url}/health",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return True, f"Service healthy: {data}"
                else:
                    return False, f"Service unhealthy: HTTP {response.status}"
        except Exception as e:
            return False, f"Service unreachable: {str(e)}"
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive memory accuracy testing"""
        logger.info("🚀 STARTING COMPREHENSIVE MEMORY ACCURACY TEST")
        logger.info("=" * 80)
        
        # Check service health
        healthy, health_msg = await self.check_service_health()
        logger.info(f"🏥 Service Health: {health_msg}")
        
        if not healthy:
            return {
                "status": "failed",
                "error": "Service not available",
                "health_check": health_msg
            }
        
        # Generate test cases
        test_cases = self.generate_realistic_test_cases()
        logger.info(f"📋 Generated {len(test_cases)} test cases")
        
        # Run tests
        start_time = time.time()
        results = []
        
        for test_case in test_cases:
            result = await self.run_accuracy_test(test_case)
            results.append(result)
            self.test_results.append(result)
        
        total_time = time.time() - start_time
        
        # Calculate overall metrics
        successful_stores = sum(1 for r in results if r.stored_successfully)
        successful_retrievals = sum(1 for r in results if r.retrieved_successfully)
        avg_content_accuracy = np.mean([r.content_accuracy for r in results])
        avg_semantic_similarity = np.mean([r.semantic_similarity for r in results])  
        avg_tag_accuracy = np.mean([r.tag_accuracy for r in results])
        avg_retrieval_time = np.mean([r.retrieval_time_ms for r in results if r.retrieval_time_ms > 0])
        
        # Performance grades
        def grade_performance(score: float) -> str:
            if score >= 0.9: return "A+ (Excellent)"
            elif score >= 0.8: return "A (Very Good)"
            elif score >= 0.7: return "B (Good)"
            elif score >= 0.6: return "C (Fair)"
            elif score >= 0.5: return "D (Poor)"
            else: return "F (Failing)"
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPREHENSIVE TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"🏪 Storage Success Rate: {successful_stores}/{len(results)} ({successful_stores/len(results)*100:.1f}%)")
        logger.info(f"🔍 Retrieval Success Rate: {successful_retrievals}/{len(results)} ({successful_retrievals/len(results)*100:.1f}%)")
        logger.info(f"📝 Content Accuracy: {avg_content_accuracy:.3f} - {grade_performance(avg_content_accuracy)}")
        logger.info(f"🧠 Semantic Similarity: {avg_semantic_similarity:.3f} - {grade_performance(avg_semantic_similarity)}")
        logger.info(f"🏷️ Tag Accuracy: {avg_tag_accuracy:.3f} - {grade_performance(avg_tag_accuracy)}")
        logger.info(f"⚡ Avg Retrieval Time: {avg_retrieval_time:.1f}ms")
        logger.info(f"⏱️ Total Test Time: {total_time:.1f}s")
        
        return {
            "status": "completed",
            "summary": {
                "total_tests": len(results),
                "storage_success_rate": successful_stores / len(results),
                "retrieval_success_rate": successful_retrievals / len(results),
                "content_accuracy": avg_content_accuracy,
                "semantic_similarity": avg_semantic_similarity,
                "tag_accuracy": avg_tag_accuracy,
                "avg_retrieval_time_ms": avg_retrieval_time,
                "total_time_seconds": total_time
            },
            "detailed_results": [
                {
                    "test_id": r.test_id,
                    "stored": r.stored_successfully,
                    "retrieved": r.retrieved_successfully,
                    "content_accuracy": r.content_accuracy,
                    "semantic_similarity": r.semantic_similarity,
                    "tag_accuracy": r.tag_accuracy,
                    "retrieval_time_ms": r.retrieval_time_ms,
                    "errors": r.errors
                }
                for r in results
            ],
            "health_check": health_msg
        }

async def main():
    """Main execution function"""
    tester = RealMemoryAccuracyTester()
    
    try:
        async with tester:
            results = await tester.run_comprehensive_test()
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"memory_accuracy_results_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 Results saved to {filename}")
            
            return results
            
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())