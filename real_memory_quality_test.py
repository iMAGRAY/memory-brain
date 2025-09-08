#!/usr/bin/env python3
"""
Real Memory Quality Test - проверка качества воспоминаний AI Memory Service
Тестирует реальную работоспособность системы памяти и качество результатов
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import os

# Конфигурация
API_BASE = "http://localhost:8080"

# Тестовые данные для реального теста качества памяти
REAL_TEST_MEMORIES = [
    {
        "content": "Пользователь предпочитает использовать Rust для системного программирования из-за безопасности памяти и высокой производительности",
        "context": "programming_preferences",
        "importance": 0.9,
        "tags": ["rust", "programming", "performance", "safety"]
    },
    {
        "content": "В проекте AI Memory Service используется Neo4j для графовой базы данных и EmbeddingGemma для векторных представлений",
        "context": "project_architecture",
        "importance": 0.95,
        "tags": ["neo4j", "embeddinggemma", "architecture", "database"]
    },
    {
        "content": "GPT-5-nano настроен с параметром max_completion_tokens вместо max_tokens для совместимости с API",
        "context": "api_configuration",
        "importance": 0.8,
        "tags": ["gpt5", "api", "configuration", "tokens"]
    },
    {
        "content": "Система показала качество embeddings 83.6% после применения специализированных промптов",
        "context": "performance_metrics",
        "importance": 0.85,
        "tags": ["embeddings", "quality", "metrics", "optimization"]
    },
    {
        "content": "Пользователь часто работает с многопоточными приложениями и требует thread-safe решений",
        "context": "user_requirements",
        "importance": 0.75,
        "tags": ["multithreading", "safety", "requirements", "concurrent"]
    }
]

# Тестовые запросы для проверки качества поиска
TEST_QUERIES = [
    {
        "query": "Какие технологии используются в нашем проекте для работы с данными?",
        "expected_context": "project_architecture",
        "expected_keywords": ["neo4j", "embeddinggemma", "database", "граф"]
    },
    {
        "query": "Как настроен GPT-5 в системе?",
        "expected_context": "api_configuration",
        "expected_keywords": ["gpt5", "max_completion_tokens", "api"]
    },
    {
        "query": "Какие результаты показала оптимизация embeddings?",
        "expected_context": "performance_metrics",
        "expected_keywords": ["83.6%", "качество", "embeddings"]
    },
    {
        "query": "Какие предпочтения пользователя в программировании?",
        "expected_context": "programming_preferences",
        "expected_keywords": ["rust", "безопасность", "производительность"]
    }
]

def store_memory(memory: Dict[str, Any]) -> bool:
    """Сохранить воспоминание в систему"""
    try:
        response = requests.post(
            f"{API_BASE}/memory",
            json=memory,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")
        return False

def search_memory(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Поиск воспоминаний по запросу"""
    try:
        response = requests.post(
            f"{API_BASE}/search",
            json={"query": query, "limit": limit},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("results", [])
        return []
    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")
        return []

def evaluate_memory_quality(query_data: Dict, search_results: List[Dict]) -> float:
    """Оценить качество возвращенных воспоминаний"""
    if not search_results:
        return 0.0
    
    quality_score = 0.0
    total_weight = 0.0
    
    for i, result in enumerate(search_results[:3]):  # Топ-3 результата
        weight = 1.0 / (i + 1)  # Весовые коэффициенты для ранжирования
        total_weight += weight
        
        # Проверка контекста
        context_match = result.get("context") == query_data["expected_context"]
        
        # Проверка ключевых слов в контенте
        content = result.get("content", "").lower()
        keyword_matches = sum(1 for keyword in query_data["expected_keywords"] 
                            if keyword.lower() in content)
        keyword_score = keyword_matches / len(query_data["expected_keywords"])
        
        # Общая оценка результата
        result_score = (0.4 * (1.0 if context_match else 0.0)) + (0.6 * keyword_score)
        quality_score += result_score * weight
    
    return quality_score / total_weight if total_weight > 0 else 0.0

def run_memory_quality_test():
    """Запустить полный тест качества памяти"""
    print("🧠 ТЕСТ РЕАЛЬНОГО КАЧЕСТВА ВОСПОМИНАНИЙ AI MEMORY SERVICE")
    print("=" * 65)
    
    # 1. Сохранение тестовых воспоминаний
    print("\n📝 Этап 1: Сохранение тестовых воспоминаний...")
    stored_count = 0
    for i, memory in enumerate(REAL_TEST_MEMORIES):
        if store_memory(memory):
            stored_count += 1
            print(f"✅ Воспоминание {i+1}/5 сохранено")
        else:
            print(f"❌ Ошибка сохранения воспоминания {i+1}")
        time.sleep(1)  # Пауза между запросами
    
    print(f"\n📊 Сохранено: {stored_count}/5 воспоминаний")
    
    if stored_count == 0:
        print("❌ КРИТИЧЕСКАЯ ОШИБКА: Ни одно воспоминание не сохранено!")
        return
    
    # Пауза для индексации
    print("\n⏳ Ожидание индексации (10 сек)...")
    time.sleep(10)
    
    # 2. Тестирование поиска и качества
    print("\n🔍 Этап 2: Тестирование качества поиска...")
    total_quality = 0.0
    successful_queries = 0
    
    for i, query_test in enumerate(TEST_QUERIES):
        print(f"\n🔍 Запрос {i+1}: {query_test['query']}")
        
        results = search_memory(query_test['query'])
        if results:
            quality = evaluate_memory_quality(query_test, results)
            total_quality += quality
            successful_queries += 1
            
            print(f"📊 Найдено результатов: {len(results)}")
            print(f"⭐ Качество результата: {quality*100:.1f}%")
            
            # Показать лучший результат
            if results:
                best = results[0]
                print(f"🥇 Лучший результат: {best.get('content', 'N/A')[:100]}...")
        else:
            print("❌ Результаты не найдены")
    
    # 3. Итоговая оценка
    print("\n" + "="*65)
    print("🏆 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    
    if successful_queries > 0:
        avg_quality = (total_quality / successful_queries) * 100
        print(f"📊 Успешных запросов: {successful_queries}/{len(TEST_QUERIES)}")
        print(f"⭐ Среднее качество воспоминаний: {avg_quality:.1f}%")
        
        if avg_quality >= 80:
            print("✅ ОТЛИЧНО: Система памяти работает качественно!")
        elif avg_quality >= 60:
            print("⚠️ ХОРОШО: Система работает, но есть потенциал для улучшения")
        else:
            print("❌ ПЛОХО: Качество воспоминаний требует серьезной оптимизации")
    else:
        print("❌ КРИТИЧЕСКАЯ ОШИБКА: Система поиска не функционирует!")
    
    print(f"\n🕒 Тест завершен: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run_memory_quality_test()