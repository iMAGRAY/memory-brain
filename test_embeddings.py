#!/usr/bin/env python3
"""
Comprehensive embedding quality test for AI Memory Service
Tests semantic similarity, consistency, and performance
"""
import json
import requests
import numpy as np
from typing import List, Dict
import time

API_URL = "http://localhost:8001"

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def test_semantic_similarity():
    """Test that semantically similar texts have high similarity scores"""
    print("\n=== SEMANTIC SIMILARITY TEST ===")
    
    test_pairs = [
        # Very similar sentences
        ("The cat sits on the mat", "A cat is sitting on a mat", 0.85),
        ("I love machine learning", "I enjoy studying machine learning", 0.80),
        
        # Related but different
        ("The weather is sunny today", "It's a beautiful bright day", 0.60),
        ("Python is a programming language", "JavaScript is used for web development", 0.40),
        
        # Completely different
        ("The cat sits on the mat", "Quantum physics is complex", 0.20),
        ("I love pizza", "The stock market crashed yesterday", 0.15),
    ]
    
    results = []
    for text1, text2, expected_min in test_pairs:
        # Get embeddings
        resp1 = requests.post(f"{API_URL}/embed", json={"text": text1})
        resp2 = requests.post(f"{API_URL}/embed", json={"text": text2})
        
        emb1 = resp1.json()["embedding"]
        emb2 = resp2.json()["embedding"]
        
        similarity = cosine_similarity(emb1, emb2)
        passed = similarity >= expected_min
        
        results.append({
            "text1": text1[:30] + "...",
            "text2": text2[:30] + "...",
            "similarity": round(similarity, 3),
            "expected_min": expected_min,
            "passed": passed
        })
        
        status = "✅" if passed else "❌"
        print(f"{status} Similarity: {similarity:.3f} (expected >= {expected_min})")
        print(f"   '{text1[:40]}...' vs '{text2[:40]}...'")
    
    return all(r["passed"] for r in results)

def test_consistency():
    """Test that same text always produces same embedding"""
    print("\n=== CONSISTENCY TEST ===")
    
    test_text = "The quick brown fox jumps over the lazy dog"
    embeddings = []
    
    for i in range(5):
        resp = requests.post(f"{API_URL}/embed", json={"text": test_text})
        embeddings.append(np.array(resp.json()["embedding"]))
        time.sleep(0.1)  # Small delay between requests
    
    # Check all embeddings are identical
    consistent = True
    for i in range(1, len(embeddings)):
        if not np.allclose(embeddings[0], embeddings[i], rtol=1e-7):
            consistent = False
            print(f"❌ Embedding {i+1} differs from first")
        else:
            print(f"✅ Embedding {i+1} is consistent")
    
    return consistent

def test_batch_api():
    """Test batch embedding API"""
    print("\n=== BATCH API TEST ===")
    
    texts = [
        "First test sentence",
        "Second test sentence", 
        "Third test sentence",
        "Fourth test sentence",
        "Fifth test sentence"
    ]
    
    # Batch request
    batch_resp = requests.post(f"{API_URL}/embed/batch", json={"texts": texts})
    batch_data = batch_resp.json()
    
    # Verify response
    if batch_data["count"] != len(texts):
        print(f"❌ Count mismatch: got {batch_data['count']}, expected {len(texts)}")
        return False
    
    if batch_data["dimension"] != 768:
        print(f"❌ Wrong dimension: {batch_data['dimension']}")
        return False
    
    # Compare with individual requests
    for i, text in enumerate(texts):
        single_resp = requests.post(f"{API_URL}/embed", json={"text": text})
        single_emb = single_resp.json()["embedding"]
        batch_emb = batch_data["embeddings"][i]
        
        if not np.allclose(single_emb, batch_emb, rtol=1e-7):
            print(f"❌ Batch embedding {i} differs from single")
            return False
    
    print(f"✅ Batch API working correctly - {len(texts)} embeddings match")
    return True

def test_performance():
    """Test embedding generation performance"""
    print("\n=== PERFORMANCE TEST ===")
    
    # Single embedding
    text = "Performance test sentence"
    start = time.time()
    resp = requests.post(f"{API_URL}/embed", json={"text": text})
    single_time = (time.time() - start) * 1000
    
    print(f"Single embedding: {single_time:.1f}ms")
    
    # Batch of 10
    texts = [f"Test sentence number {i}" for i in range(10)]
    start = time.time()
    resp = requests.post(f"{API_URL}/embed/batch", json={"texts": texts})
    batch_time = (time.time() - start) * 1000
    avg_time = batch_time / 10
    
    print(f"Batch of 10: {batch_time:.1f}ms total, {avg_time:.1f}ms per embedding")
    
    # Performance criteria
    if single_time > 500:
        print(f"⚠️  Single embedding slow: {single_time:.1f}ms")
        return False
    
    if avg_time > 100:
        print(f"⚠️  Batch processing slow: {avg_time:.1f}ms per item")
        return False
    
    print("✅ Performance acceptable")
    return True

def test_vector_properties():
    """Test mathematical properties of embeddings"""
    print("\n=== VECTOR PROPERTIES TEST ===")
    
    texts = [
        "Apple is a fruit",
        "Car is a vehicle",
        "Programming is fun"
    ]
    
    resp = requests.post(f"{API_URL}/embed/batch", json={"texts": texts})
    embeddings = [np.array(emb) for emb in resp.json()["embeddings"]]
    
    all_passed = True
    
    # Check dimension
    for i, emb in enumerate(embeddings):
        if len(emb) != 768:
            print(f"❌ Wrong dimension for text {i}: {len(emb)}")
            all_passed = False
        else:
            print(f"✅ Dimension correct: 768")
            break
    
    # Check normalization (should be unit vectors or close to it)
    for i, emb in enumerate(embeddings):
        norm = np.linalg.norm(emb)
        if abs(norm - 1.0) > 0.1:  # Allow some deviation
            print(f"⚠️  Vector {i} norm: {norm:.3f} (not unit normalized)")
        else:
            print(f"✅ Vector {i} norm: {norm:.3f}")
    
    # Check value ranges
    for i, emb in enumerate(embeddings):
        min_val = np.min(emb)
        max_val = np.max(emb)
        mean_val = np.mean(emb)
        std_val = np.std(emb)
        
        print(f"Vector {i} stats: min={min_val:.3f}, max={max_val:.3f}, mean={mean_val:.3f}, std={std_val:.3f}")
        
        if min_val < -10 or max_val > 10:
            print(f"⚠️  Unusual value range")
    
    return all_passed

def test_multilingual():
    """Test multilingual support"""
    print("\n=== MULTILINGUAL TEST ===")
    
    texts = [
        "Hello world",  # English
        "Привет мир",   # Russian
        "你好世界",      # Chinese
        "Hola mundo",   # Spanish
        "Bonjour le monde",  # French
        "こんにちは世界"  # Japanese
    ]
    
    try:
        resp = requests.post(f"{API_URL}/embed/batch", json={"texts": texts})
        data = resp.json()
        
        if len(data["embeddings"]) == len(texts):
            print(f"✅ All {len(texts)} languages processed successfully")
            
            # Check similarity between "Hello world" in different languages
            embeddings = [np.array(emb) for emb in data["embeddings"]]
            english_emb = embeddings[0]
            
            for i, (text, emb) in enumerate(zip(texts[1:], embeddings[1:]), 1):
                sim = cosine_similarity(english_emb.tolist(), emb.tolist())
                print(f"  '{texts[0]}' vs '{text}': similarity = {sim:.3f}")
            
            return True
        else:
            print(f"❌ Failed to process all languages")
            return False
    except Exception as e:
        print(f"❌ Multilingual test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 AI MEMORY SERVICE EMBEDDING QUALITY TEST")
    print("=" * 50)
    
    # Check service is running
    try:
        resp = requests.get(f"{API_URL}/health")
        if resp.json()["status"] != "healthy":
            print("❌ Service not healthy")
            return
    except:
        print("❌ Service not responding at", API_URL)
        return
    
    results = {
        "semantic_similarity": test_semantic_similarity(),
        "consistency": test_consistency(),
        "batch_api": test_batch_api(),
        "performance": test_performance(),
        "vector_properties": test_vector_properties(),
        "multilingual": test_multilingual()
    }
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} : {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED - EMBEDDINGS ARE HIGH QUALITY")
    else:
        print("⚠️  SOME TESTS FAILED - CHECK DETAILS ABOVE")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)