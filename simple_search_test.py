#!/usr/bin/env python3
"""
Простой тест для проверки исправленных search endpoints
"""
import requests
import json

def test_health():
    """Тест health endpoint"""
    try:
        response = requests.get("http://127.0.0.1:8080/health", timeout=5)
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_search():
    """Тест GET search endpoint"""  
    try:
        response = requests.get(
            "http://127.0.0.1:8080/search", 
            params={"query": "test", "limit": 3},
            timeout=10
        )
        print(f"Search test: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ Search successful: found {data.get('total', 0)} results")
                if data.get('results'):
                    print(f"First result preview: {str(data['results'][0])[:100]}...")
                return data.get('total', 0)
            except json.JSONDecodeError:
                print("❌ Invalid JSON response")
                print(f"Response: {response.text[:200]}...")
                return -1
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            return -1
            
    except Exception as e:
        print(f"❌ Search request failed: {e}")
        return -1

def main():
    print("🚀 Testing AI Memory Service Search Endpoints")
    print("=" * 50)
    
    if not test_health():
        print("Server is not accessible. Exiting.")
        return
    
    results = test_search()
    
    print("=" * 50)
    if results >= 0:
        print(f"🎯 Search endpoint works! Found {results} results")
        if results == 0:
            print("⚠️  No results found - this indicates a quality issue")
        else:
            print("✅ Search endpoint is functional")
    else:
        print("❌ Search endpoint has issues")

if __name__ == "__main__":
    main()