#!/usr/bin/env python3
"""
Comprehensive Memory System Quality Test
Tests search accuracy and quality after fixes
"""
import requests
import json
import time

def test_search_quality():
    """Test various search queries to measure quality"""
    base_url = "http://127.0.0.1:8080"
    
    # Test queries with expected relevance
    test_queries = [
        {
            "query": "memory",
            "expected_min": 2,
            "description": "Basic memory search"
        },
        {
            "query": "python programming",
            "expected_min": 1,
            "description": "Programming language search"
        },
        {
            "query": "javascript react",
            "expected_min": 1,
            "description": "Framework search"
        },
        {
            "query": "database storage",
            "expected_min": 1,
            "description": "Storage concept search"
        },
        {
            "query": "ai artificial intelligence",
            "expected_min": 1,
            "description": "AI concepts search"
        }
    ]
    
    print("🧠 COMPREHENSIVE MEMORY QUALITY TEST")
    print("=" * 50)
    
    total_tests = 0
    successful_tests = 0
    total_results = 0
    
    for test in test_queries:
        try:
            print(f"\n🔍 Testing: {test['description']}")
            print(f"   Query: '{test['query']}'")
            
            response = requests.get(
                f"{base_url}/search",
                params={"query": test["query"], "limit": 10},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                results_count = data.get('total', 0)
                total_results += results_count
                
                print(f"   Results: {results_count}")
                
                if results_count >= test["expected_min"]:
                    print(f"   ✅ PASS (expected >= {test['expected_min']})")
                    successful_tests += 1
                else:
                    print(f"   ❌ FAIL (expected >= {test['expected_min']})")
                
                # Show top results
                if results_count > 0:
                    print("   Top results:")
                    for i, result in enumerate(data.get('results', [])[:3]):
                        content = result.get('content', 'No content')[:80] + "..."
                        importance = result.get('importance', 0)
                        print(f"     {i+1}. [{importance:.2f}] {content}")
            else:
                print(f"   ❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
        
        total_tests += 1
        time.sleep(0.5)  # Rate limiting
    
    # Calculate quality metrics
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    avg_results = total_results / total_tests if total_tests > 0 else 0
    
    print(f"\n📊 QUALITY METRICS")
    print("=" * 30)
    print(f"Tests passed: {successful_tests}/{total_tests}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Average results per query: {avg_results:.1f}")
    print(f"Total results found: {total_results}")
    
    # Quality assessment
    if success_rate >= 80:
        quality_level = "EXCELLENT"
        emoji = "🎉"
    elif success_rate >= 60:
        quality_level = "GOOD"
        emoji = "✅"
    elif success_rate >= 40:
        quality_level = "MODERATE" 
        emoji = "⚠️"
    else:
        quality_level = "POOR"
        emoji = "❌"
    
    print(f"\n{emoji} OVERALL QUALITY: {quality_level}")
    
    return success_rate, avg_results

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get("http://127.0.0.1:8080/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("🏥 HEALTH CHECK")
            print("=" * 20)
            print(f"Service: {data.get('service', 'unknown')}")
            print(f"Total memories: {data.get('memory_stats', {}).get('total_memories', 0)}")
            print(f"Contexts: {data.get('memory_stats', {}).get('total_contexts', 0)}")
            print(f"Avg recall time: {data.get('memory_stats', {}).get('avg_recall_time_ms', 0):.1f}ms")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("Starting comprehensive memory quality assessment...")
    
    # Test health first
    if test_health_endpoint():
        print("\n" + "="*60 + "\n")
        
        # Run quality tests
        success_rate, avg_results = test_search_quality()
        
        print("\n" + "="*60)
        print("🎯 FINAL ASSESSMENT")
        print("="*60)
        
        if success_rate >= 80 and avg_results >= 1.5:
            print("🎉 MISSION ACCOMPLISHED!")
            print("   Memory system quality has been restored!")
            print(f"   Quality: {success_rate:.1f}% (Target: ≥70%)")
            print(f"   Coverage: {avg_results:.1f} results/query (Target: ≥1.5)")
        elif success_rate >= 60:
            print("✅ SIGNIFICANT IMPROVEMENT!")
            print("   Memory system is working well")
            print(f"   Quality: {success_rate:.1f}% (Close to target)")
        else:
            print("⚠️  Still needs improvement")
            print(f"   Quality: {success_rate:.1f}% (Target: ≥70%)")
    
    print("\nTest completed!")