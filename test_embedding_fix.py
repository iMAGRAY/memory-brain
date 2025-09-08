#!/usr/bin/env python3
"""
Test script to verify that the embedding server now properly handles task_type
and generates different embeddings for 'query' vs 'document' task types
"""

import asyncio
import aiohttp
import numpy as np
import json

async def test_task_type_differentiation():
    """Test that query and document embeddings are different"""
    print("🧪 Testing embedding task_type differentiation...")
    print("-" * 50)
    
    test_text = "how to optimize memory usage in ai systems"
    
    async with aiohttp.ClientSession() as session:
        # Test query embedding
        async with session.post(
            "http://localhost:8090/embed",
            json={"text": test_text, "task_type": "query"},
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                print(f"❌ Query embedding failed: {response.status}")
                return False
            
            query_result = await response.json()
            query_embedding = query_result["embedding"]
            
        print(f"✅ Query embedding: {len(query_embedding)} dimensions")
        print(f"   First 5 values: {query_embedding[:5]}")
        
        # Test document embedding
        async with session.post(
            "http://localhost:8090/embed", 
            json={"text": test_text, "task_type": "document"},
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                print(f"❌ Document embedding failed: {response.status}")
                return False
                
            doc_result = await response.json()
            doc_embedding = doc_result["embedding"]
            
        print(f"✅ Document embedding: {len(doc_embedding)} dimensions")
        print(f"   First 5 values: {doc_embedding[:5]}")
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            a = np.array(a)
            b = np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        similarity = cosine_similarity(query_embedding, doc_embedding)
        print(f"\n📊 Cosine similarity between query and document: {similarity:.6f}")
        
        # Test general embedding (should be different from both)
        async with session.post(
            "http://localhost:8090/embed",
            json={"text": test_text, "task_type": "general"},
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                print(f"❌ General embedding failed: {response.status}")
                return False
                
            general_result = await response.json()
            general_embedding = general_result["embedding"]
            
        print(f"✅ General embedding: {len(general_embedding)} dimensions")
        print(f"   First 5 values: {general_embedding[:5]}")
        
        general_query_sim = cosine_similarity(general_embedding, query_embedding)
        general_doc_sim = cosine_similarity(general_embedding, doc_embedding)
        
        print(f"\n📊 General vs Query similarity: {general_query_sim:.6f}")
        print(f"📊 General vs Document similarity: {general_doc_sim:.6f}")
        
        # Evaluate results
        print("\n" + "=" * 50)
        print("🎯 RESULTS ANALYSIS:")
        
        if similarity < 0.99:  # Embeddings should be different
            print("✅ SUCCESS: Query and document embeddings are DIFFERENT!")
            print(f"   Similarity: {similarity:.6f} (< 0.99)")
        else:
            print("❌ FAILURE: Query and document embeddings are too similar!")
            print(f"   Similarity: {similarity:.6f} (>= 0.99)")
            return False
            
        if abs(general_query_sim - general_doc_sim) > 0.01:
            print("✅ SUCCESS: General embedding behaves differently from task-specific ones")
        else:
            print("⚠️  WARNING: General embedding might not be working as expected")
        
        return True

async def test_memory_server_integration():
    """Test that the memory server can now find results with fixed embeddings"""
    print("\n" + "=" * 50)
    print("🔍 Testing Memory Server Integration...")
    print("-" * 50)
    
    # Test a query that should return results from the documentation we imported
    test_query = "Claude Code hooks for development"
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8080/search",
            json={"query": test_query, "limit": 3},
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                print(f"❌ Memory server search failed: {response.status}")
                return False
                
            result = await response.json()
            memories = result.get("memories", [])
            confidence = result.get("confidence", 0)
            
            print(f"📊 Search Results for: '{test_query}'")
            print(f"   Found: {len(memories)} memories")
            print(f"   Confidence: {confidence:.3f}")
            
            if len(memories) > 0:
                print("\n📖 Top Results:")
                for i, memory in enumerate(memories[:3], 1):
                    content = memory.get("content", "")
                    score = memory.get("score", 0)
                    print(f"   {i}. Score: {score:.3f}")
                    print(f"      Content: {content[:100]}...")
                    
                print("✅ SUCCESS: Memory server now returns results!")
                return True
            else:
                print("❌ FAILURE: Memory server still returns no results")
                return False

async def main():
    print("🚀 Testing Fixed Embedding Service")
    print("=" * 60)
    
    # Test 1: Task type differentiation
    embedding_success = await test_task_type_differentiation()
    
    # Test 2: Memory server integration
    search_success = await test_memory_server_integration()
    
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY:")
    print(f"✅ Embedding Differentiation: {'PASS' if embedding_success else 'FAIL'}")
    print(f"🔍 Memory Search: {'PASS' if search_success else 'FAIL'}")
    
    if embedding_success and search_success:
        print("\n🎉 ALL TESTS PASSED! The fix is working correctly!")
        print("   - Embeddings now differentiate between query and document types")
        print("   - Memory server can find relevant results")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())