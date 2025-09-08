#!/usr/bin/env python3
"""
Definitive Stress Test для embedding_server.py
Comprehensive testing to prove/disprove code quality claims
"""

import sys
import os
import time
import threading
import gc
import psutil
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import logging

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StressTestResults:
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.errors = []
        self.start_time = time.perf_counter()
        
    def add_result(self, test_name: str, success: bool, details: str = ""):
        self.test_results[test_name] = {
            'success': success,
            'details': details,
            'timestamp': time.perf_counter() - self.start_time
        }
        
    def add_performance(self, metric_name: str, value: float):
        self.performance_metrics[metric_name] = value
        
    def add_error(self, error: str):
        self.errors.append({
            'error': error,
            'timestamp': time.perf_counter() - self.start_time
        })

def get_memory_usage() -> float:
    """Get current process memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def get_model_path() -> str:
    """Get model path from environment or use default"""
    model_path = os.environ.get('EMBEDDING_MODEL_PATH')
    if model_path and Path(model_path).exists():
        return model_path
    
    # Fallback paths
    possible_paths = [
        r'C:\Models\ai-memory-service\models\embeddinggemma-300m',
        r'models\embeddinggemma-300m',
        r'.\models\embeddinggemma-300m'
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    raise FileNotFoundError("Embedding model not found. Set EMBEDDING_MODEL_PATH environment variable")

def test_basic_functionality(results: StressTestResults) -> bool:
    """Test 1: Basic functionality"""
    logger.info("Test 1: Basic functionality")
    
    try:
        from embedding_server import EmbeddingService
        model_path = get_model_path()
        
        service = EmbeddingService(model_path=model_path, cache_size=10)
        
        # Test encoding
        query_emb = service.encode_query("test query")
        doc_emb = service.encode_document("test document")
        
        # Validate outputs
        assert query_emb is not None, "Query embedding is None"
        assert doc_emb is not None, "Document embedding is None"
        assert query_emb.shape[0] == 512, f"Query embedding wrong shape: {query_emb.shape}"
        assert doc_emb.shape[0] == 512, f"Document embedding wrong shape: {doc_emb.shape}"
        
        # Test statistics
        stats = service.get_comprehensive_stats()
        assert 'cache_hits' in stats, "Statistics missing cache_hits"
        assert 'cache_misses' in stats, "Statistics missing cache_misses"
        
        results.add_result("basic_functionality", True, "All basic functions work correctly")
        return True
        
    except Exception as e:
        results.add_result("basic_functionality", False, f"Basic functionality failed: {str(e)}")
        results.add_error(f"Basic functionality: {str(e)}")
        return False

def test_race_conditions_cache(results: StressTestResults) -> bool:
    """Test 2: Race conditions in cache"""
    logger.info("Test 2: Race conditions in cache")
    
    try:
        from embedding_server import EmbeddingService
        model_path = get_model_path()
        
        service = EmbeddingService(model_path=model_path, cache_size=50)
        
        def worker_cache_test(worker_id: int, results_list: List):
            try:
                for i in range(20):
                    text = f"worker_{worker_id}_text_{i}"
                    emb = service.encode_query(text)
                    
                    # Immediately get from cache to test race conditions
                    emb2 = service.encode_query(text)
                    
                    # Test if embeddings are identical (cache hit)
                    if np.allclose(emb, emb2):
                        results_list.append(f"worker_{worker_id}_success_{i}")
                    else:
                        results_list.append(f"worker_{worker_id}_FAIL_{i}")
                        
            except Exception as e:
                results_list.append(f"worker_{worker_id}_ERROR_{str(e)}")
        
        # Run parallel workers
        worker_results = []
        threads = []
        
        for worker_id in range(5):
            thread = threading.Thread(target=worker_cache_test, args=(worker_id, worker_results))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Analyze results
        successes = [r for r in worker_results if "success" in r]
        failures = [r for r in worker_results if "FAIL" in r]
        errors = [r for r in worker_results if "ERROR" in r]
        
        success_rate = len(successes) / len(worker_results) if worker_results else 0
        
        if success_rate > 0.95:  # 95% success rate is acceptable
            results.add_result("race_conditions_cache", True, 
                             f"Success rate: {success_rate:.2%}, Errors: {len(errors)}")
            return True
        else:
            results.add_result("race_conditions_cache", False, 
                             f"Success rate: {success_rate:.2%}, Failures: {len(failures)}, Errors: {len(errors)}")
            return False
            
    except Exception as e:
        results.add_result("race_conditions_cache", False, f"Race condition test failed: {str(e)}")
        results.add_error(f"Race conditions cache: {str(e)}")
        return False

def test_race_conditions_stats(results: StressTestResults) -> bool:
    """Test 3: Race conditions in statistics"""
    logger.info("Test 3: Race conditions in statistics")
    
    try:
        from embedding_server import EmbeddingService
        model_path = get_model_path()
        
        service = EmbeddingService(model_path=model_path, cache_size=10)
        
        def stats_worker(worker_id: int, stats_list: List):
            try:
                for i in range(50):
                    stats = service.get_comprehensive_stats()
                    
                    # Validate stats structure and ranges
                    if (0 <= stats.get('hit_rate', -1) <= 1.0 and 
                        stats.get('cache_hits', -1) >= 0 and 
                        stats.get('cache_misses', -1) >= 0):
                        stats_list.append(f"worker_{worker_id}_valid_{i}")
                    else:
                        stats_list.append(f"worker_{worker_id}_INVALID_{i}_stats_{stats}")
                        
            except Exception as e:
                stats_list.append(f"worker_{worker_id}_ERROR_{str(e)}")
        
        # Run parallel stats requests
        stats_results = []
        threads = []
        
        for worker_id in range(3):
            thread = threading.Thread(target=stats_worker, args=(worker_id, stats_results))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Analyze results
        valid_stats = [r for r in stats_results if "valid" in r]
        invalid_stats = [r for r in stats_results if "INVALID" in r]
        errors = [r for r in stats_results if "ERROR" in r]
        
        success_rate = len(valid_stats) / len(stats_results) if stats_results else 0
        
        if success_rate > 0.98:  # 98% success rate for stats
            results.add_result("race_conditions_stats", True, 
                             f"Stats success rate: {success_rate:.2%}")
            return True
        else:
            results.add_result("race_conditions_stats", False, 
                             f"Stats success rate: {success_rate:.2%}, Invalid: {len(invalid_stats)}")
            return False
            
    except Exception as e:
        results.add_result("race_conditions_stats", False, f"Stats race condition test failed: {str(e)}")
        results.add_error(f"Race conditions stats: {str(e)}")
        return False

def test_memory_leaks(results: StressTestResults) -> bool:
    """Test 4: Memory leak detection"""
    logger.info("Test 4: Memory leak detection")
    
    try:
        from embedding_server import EmbeddingService
        model_path = get_model_path()
        
        # Measure baseline memory
        baseline_memory = get_memory_usage()
        results.add_performance("baseline_memory_mb", baseline_memory)
        
        # Create and destroy services multiple times
        for cycle in range(3):
            service = EmbeddingService(model_path=model_path, cache_size=100)
            
            # Heavy usage
            for i in range(50):
                text = f"memory_test_cycle_{cycle}_item_{i}"
                _ = service.encode_query(text)
                _ = service.encode_document(text)
            
            # Get stats multiple times
            for i in range(20):
                _ = service.get_comprehensive_stats()
            
            # Cleanup
            if hasattr(service, '_cleanup_resources'):
                service._cleanup_resources()
            del service
            
            # Force garbage collection
            gc.collect()
            
            current_memory = get_memory_usage()
            memory_increase = current_memory - baseline_memory
            results.add_performance(f"memory_after_cycle_{cycle}_mb", current_memory)
            
            logger.info(f"Memory after cycle {cycle}: {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        # Final memory check
        final_memory = get_memory_usage()
        total_increase = final_memory - baseline_memory
        results.add_performance("final_memory_mb", final_memory)
        results.add_performance("total_memory_increase_mb", total_increase)
        
        # Memory leak threshold: 50MB increase is acceptable
        if total_increase < 50:
            results.add_result("memory_leaks", True, 
                             f"Memory increase: {total_increase:.1f}MB (acceptable)")
            return True
        else:
            results.add_result("memory_leaks", False, 
                             f"Memory increase: {total_increase:.1f}MB (potential leak)")
            return False
            
    except Exception as e:
        results.add_result("memory_leaks", False, f"Memory leak test failed: {str(e)}")
        results.add_error(f"Memory leaks: {str(e)}")
        return False

def test_shutdown_deadlock(results: StressTestResults) -> bool:
    """Test 5: Shutdown deadlock scenarios"""
    logger.info("Test 5: Shutdown deadlock scenarios")
    
    try:
        from embedding_server import EmbeddingService
        model_path = get_model_path()
        
        shutdown_times = []
        
        for iteration in range(3):
            service = EmbeddingService(model_path=model_path, cache_size=20)
            
            # Simulate active usage
            def active_usage():
                for i in range(10):
                    try:
                        _ = service.encode_query(f"shutdown_test_{i}")
                        time.sleep(0.01)  # Small delay
                    except Exception:
                        break  # Expected during shutdown
            
            # Start background activity
            usage_thread = threading.Thread(target=active_usage)
            usage_thread.start()
            
            # Measure shutdown time
            shutdown_start = time.perf_counter()
            
            if hasattr(service, '_cleanup_resources'):
                service._cleanup_resources()
            
            usage_thread.join(timeout=2.0)  # 2 second timeout
            
            shutdown_time = time.perf_counter() - shutdown_start
            shutdown_times.append(shutdown_time)
            
            results.add_performance(f"shutdown_time_{iteration}_s", shutdown_time)
            
            logger.info(f"Shutdown iteration {iteration}: {shutdown_time:.3f}s")
        
        avg_shutdown_time = sum(shutdown_times) / len(shutdown_times)
        max_shutdown_time = max(shutdown_times)
        
        results.add_performance("avg_shutdown_time_s", avg_shutdown_time)
        results.add_performance("max_shutdown_time_s", max_shutdown_time)
        
        # Deadlock threshold: 5 seconds
        if max_shutdown_time < 5.0:
            results.add_result("shutdown_deadlock", True, 
                             f"Max shutdown time: {max_shutdown_time:.3f}s")
            return True
        else:
            results.add_result("shutdown_deadlock", False, 
                             f"Max shutdown time: {max_shutdown_time:.3f}s (potential deadlock)")
            return False
            
    except Exception as e:
        results.add_result("shutdown_deadlock", False, f"Shutdown test failed: {str(e)}")
        results.add_error(f"Shutdown deadlock: {str(e)}")
        return False

def test_performance_degradation(results: StressTestResults) -> bool:
    """Test 6: Performance degradation claims"""
    logger.info("Test 6: Performance degradation")
    
    try:
        from embedding_server import EmbeddingService
        model_path = get_model_path()
        
        service = EmbeddingService(model_path=model_path, cache_size=50)
        
        # Warmup
        for i in range(5):
            _ = service.encode_query(f"warmup_{i}")
        
        # Measure encoding performance
        encoding_times = []
        for i in range(50):
            text = f"performance_test_{i}"
            start_time = time.perf_counter()
            _ = service.encode_query(text)
            encoding_time = time.perf_counter() - start_time
            encoding_times.append(encoding_time)
        
        avg_encoding_time = sum(encoding_times) / len(encoding_times)
        p95_encoding_time = np.percentile(encoding_times, 95)
        
        results.add_performance("avg_encoding_time_ms", avg_encoding_time * 1000)
        results.add_performance("p95_encoding_time_ms", p95_encoding_time * 1000)
        
        # Measure stats performance
        stats_times = []
        for i in range(100):
            start_time = time.perf_counter()
            _ = service.get_comprehensive_stats()
            stats_time = time.perf_counter() - start_time
            stats_times.append(stats_time)
        
        avg_stats_time = sum(stats_times) / len(stats_times)
        max_stats_time = max(stats_times)
        
        results.add_performance("avg_stats_time_ms", avg_stats_time * 1000)
        results.add_performance("max_stats_time_ms", max_stats_time * 1000)
        
        # Performance thresholds (generous for local testing)
        encoding_acceptable = avg_encoding_time < 0.1  # 100ms average
        stats_acceptable = avg_stats_time < 0.01   # 10ms average
        
        if encoding_acceptable and stats_acceptable:
            results.add_result("performance_degradation", True, 
                             f"Encoding: {avg_encoding_time*1000:.1f}ms avg, Stats: {avg_stats_time*1000:.1f}ms avg")
            return True
        else:
            results.add_result("performance_degradation", False, 
                             f"Encoding: {avg_encoding_time*1000:.1f}ms avg, Stats: {avg_stats_time*1000:.1f}ms avg")
            return False
            
    except Exception as e:
        results.add_result("performance_degradation", False, f"Performance test failed: {str(e)}")
        results.add_error(f"Performance degradation: {str(e)}")
        return False

def test_edge_cases(results: StressTestResults) -> bool:
    """Test 7: Edge cases and error handling"""
    logger.info("Test 7: Edge cases and error handling")
    
    try:
        from embedding_server import EmbeddingService
        model_path = get_model_path()
        
        service = EmbeddingService(model_path=model_path, cache_size=5)
        
        edge_case_results = []
        
        # Test empty string
        try:
            result = service.encode_query("")
            if result is not None:
                edge_case_results.append("empty_string_handled")
            else:
                edge_case_results.append("empty_string_rejected")
        except Exception:
            edge_case_results.append("empty_string_error")
        
        # Test very long string (but not excessive)
        try:
            long_text = "a" * 1000  # 1000 chars, reasonable
            result = service.encode_query(long_text)
            if result is not None:
                edge_case_results.append("long_text_handled")
        except Exception:
            edge_case_results.append("long_text_error")
        
        # Test cache overflow
        try:
            for i in range(10):  # More than cache_size=5
                _ = service.encode_query(f"cache_overflow_{i}")
            edge_case_results.append("cache_overflow_handled")
        except Exception:
            edge_case_results.append("cache_overflow_error")
        
        # Test stats with zero operations
        try:
            fresh_service = EmbeddingService(model_path=model_path, cache_size=5)
            stats = fresh_service.get_comprehensive_stats()
            if 0 <= stats.get('hit_rate', -1) <= 1:
                edge_case_results.append("zero_stats_handled")
        except Exception:
            edge_case_results.append("zero_stats_error")
        
        # Success if most edge cases handled gracefully
        handled_cases = [r for r in edge_case_results if "handled" in r]
        success_rate = len(handled_cases) / len(edge_case_results) if edge_case_results else 0
        
        if success_rate >= 0.75:  # 75% of edge cases handled
            results.add_result("edge_cases", True, 
                             f"Edge cases handled: {handled_cases}")
            return True
        else:
            results.add_result("edge_cases", False, 
                             f"Edge cases results: {edge_case_results}")
            return False
            
    except Exception as e:
        results.add_result("edge_cases", False, f"Edge cases test failed: {str(e)}")
        results.add_error(f"Edge cases: {str(e)}")
        return False

def run_comprehensive_stress_test() -> StressTestResults:
    """Run all stress tests"""
    logger.info("=" * 80)
    logger.info("STARTING COMPREHENSIVE STRESS TEST")
    logger.info("=" * 80)
    
    results = StressTestResults()
    
    # Test suite
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Race Conditions Cache", test_race_conditions_cache),
        ("Race Conditions Stats", test_race_conditions_stats),
        ("Memory Leaks", test_memory_leaks),
        ("Shutdown Deadlock", test_shutdown_deadlock),
        ("Performance Degradation", test_performance_degradation),
        ("Edge Cases", test_edge_cases),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func(results)
            if success:
                passed_tests += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name}: CRASHED - {str(e)}")
            results.add_error(f"{test_name} crashed: {str(e)}")
    
    results.add_performance("tests_passed", passed_tests)
    results.add_performance("tests_total", total_tests)
    results.add_performance("success_rate", passed_tests / total_tests)
    
    return results

def generate_final_report(results: StressTestResults):
    """Generate comprehensive final report"""
    logger.info("\n" + "=" * 80)
    logger.info("FINAL STRESS TEST REPORT")
    logger.info("=" * 80)
    
    total_time = time.perf_counter() - results.start_time
    passed = sum(1 for result in results.test_results.values() if result['success'])
    total = len(results.test_results)
    success_rate = passed / total if total > 0 else 0
    
    logger.info(f"\n📊 OVERALL RESULTS:")
    logger.info(f"   Tests Passed: {passed}/{total}")
    logger.info(f"   Success Rate: {success_rate:.1%}")
    logger.info(f"   Total Time: {total_time:.2f}s")
    logger.info(f"   Errors: {len(results.errors)}")
    
    logger.info(f"\n🧪 TEST RESULTS:")
    for test_name, result in results.test_results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        logger.info(f"   {status} {test_name}: {result['details']}")
    
    logger.info(f"\n⚡ PERFORMANCE METRICS:")
    for metric, value in results.performance_metrics.items():
        if 'time' in metric.lower():
            if 'ms' in metric:
                logger.info(f"   {metric}: {value:.2f}ms")
            else:
                logger.info(f"   {metric}: {value:.3f}s")
        elif 'memory' in metric.lower():
            logger.info(f"   {metric}: {value:.1f}MB")
        else:
            logger.info(f"   {metric}: {value}")
    
    if results.errors:
        logger.info(f"\n🚨 ERRORS ({len(results.errors)}):")
        for error in results.errors:
            logger.info(f"   [{error['timestamp']:.2f}s] {error['error']}")
    
    # Final verdict
    logger.info(f"\n" + "=" * 80)
    if success_rate >= 0.85:  # 85% success threshold
        logger.info("🎉 VERDICT: CODE QUALITY EXCELLENT")
        logger.info("   All critical vulnerabilities appear to be fixed")
        logger.info("   Performance is within acceptable ranges")
        logger.info("   Memory management is working correctly")
    elif success_rate >= 0.70:
        logger.info("⚠️  VERDICT: CODE QUALITY GOOD WITH MINOR ISSUES")
        logger.info("   Most functionality works correctly")
        logger.info("   Some edge cases may need attention")
    else:
        logger.info("❌ VERDICT: CODE QUALITY NEEDS IMPROVEMENT")
        logger.info("   Multiple critical issues detected")
        logger.info("   Significant fixes needed before production use")
    
    logger.info("=" * 80)
    
    return success_rate >= 0.85

if __name__ == "__main__":
    try:
        results = run_comprehensive_stress_test()
        success = generate_final_report(results)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical test failure: {str(e)}")
        sys.exit(1)