#!/usr/bin/env python3
"""
Test script to verify search functionality after embedding fix
"""

import requests
import json

API_BASE = "http://localhost:8080"

def test_search(query, expected_results=None):
    """Test search functionality"""
    print(f"\n🔍 Testing search: '{query}'")
    print("-" * 50)
    
    try:
        response = requests.post(
            f"{API_BASE}/api/search",
            json={"query": query, "limit": 5},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"❌ Search failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        results = response.json()
        
        if "memories" not in results:
            print("❌ No memories field in response")
            return False
        
        memories = results["memories"]
        total_results = len(memories)
        
        print(f"✅ Found {total_results} results")
        
        if total_results == 0:
            print("⚠️  No results found - this may indicate embedding issues")
            return False
        
        # Show top results
        for i, memory in enumerate(memories[:3], 1):
            score = memory.get("similarity_score", 0)
            content_preview = memory.get("content", "")[:100]
            context_path = memory.get("context_path", "unknown")
            
            print(f"{i}. Score: {score:.4f}")
            print(f"   Path: {context_path}")
            print(f"   Content: {content_preview}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return False

def test_memory_recall():
    """Test memory recall endpoint"""
    print("\n🧠 Testing memory recall")
    print("-" * 50)
    
    try:
        response = requests.post(
            f"{API_BASE}/api/recall",
            json={"query": "Claude Code hooks", "limit": 3},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"❌ Memory recall failed: {response.status_code}")
            return False
        
        results = response.json()
        
        if "memories" not in results:
            print("❌ No memories field in response")
            return False
        
        memories = results["memories"]
        print(f"✅ Memory recall returned {len(memories)} results")
        return True
        
    except Exception as e:
        print(f"❌ Error during memory recall: {e}")
        return False

def main():
    print("=" * 60)
    print("🧪 Testing Search Functionality After Embedding Fix")
    print("=" * 60)
    
    # Test various search queries
    test_queries = [
        "Claude Code hooks",
        "TypeScript mistakes",
        "Python dependencies",
        "React components",
        "embeddings documentation",
        "GPT-5 API guide",
        "Rust crates 2025"
    ]
    
    successful_tests = 0
    
    for query in test_queries:
        if test_search(query):
            successful_tests += 1
    
    # Test memory recall
    if test_memory_recall():
        successful_tests += 1
    
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"Successful tests: {successful_tests}/{len(test_queries) + 1}")
    
    if successful_tests == len(test_queries) + 1:
        print("✅ All tests passed! Embedding fix is working correctly.")
    elif successful_tests > len(test_queries) // 2:
        print("⚠️  Most tests passed, but some issues remain.")
    else:
        print("❌ Many tests failed, embedding issues may persist.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()