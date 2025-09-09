#!/usr/bin/env python3
"""
Быстрая диагностика подключения к AI Memory Service
"""
import socket
import requests
import json

def check_port():
    """Проверяем доступность порта 8080"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(('127.0.0.1', 8080))
        sock.close()
        
        if result == 0:
            print("✅ Порт 8080 доступен")
            return True
        else:
            print(f"❌ Порт 8080 недоступен (error {result})")
            return False
    except Exception as e:
        print(f"❌ Ошибка проверки порта: {e}")
        return False

def quick_test():
    """Быстрый тест API"""
    if not check_port():
        return
        
    # Тест 1: Health endpoint с расширенным timeout 
    try:
        response = requests.get("http://127.0.0.1:8080/health", timeout=15)
        print(f"Health: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Сервис работает: {data.get('service', 'unknown')}")
            print(f"📊 Memory stats: {data.get('memory_stats', {})}")
        else:
            print(f"❌ Health check failed: {response.text[:100]}")
            return
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return
        
    # Тест 2: Search endpoint
    try:
        response = requests.get(
            "http://127.0.0.1:8080/search",
            params={"query": "memory", "limit": 2},
            timeout=15
        )
        print(f"Search: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            total = data.get('total', 0)
            print(f"🔍 Найдено результатов: {total}")
            
            if total > 0:
                print(f"✅ Поиск работает! Качество системы восстановлено")
                for i, result in enumerate(data.get('results', [])[:2]):
                    content = result.get('content', 'No content')[:60] + "..."
                    print(f"  {i+1}: {content}")
            else:
                print(f"⚠️  Поиск работает, но не находит результаты - проблема с качеством")
                
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(f"Error: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Search endpoint error: {e}")

if __name__ == "__main__":
    print("🔍 Quick AI Memory Service Diagnostic")
    print("=" * 40)
    quick_test()