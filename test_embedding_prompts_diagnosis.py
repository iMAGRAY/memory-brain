#!/usr/bin/env python3
"""
Диагностический тест для проверки правильности использования промптов
в EmbeddingGemma и их влияния на качество поиска
"""
import asyncio
import aiohttp
import json
import sys
from sentence_transformers import SentenceTransformer

async def test_embedding_prompts():
    """Тест для проверки что embedding server правильно использует промпты"""
    print("🔍 Диагностика Prompts в Embedding Service")
    print("=" * 60)
    
    # 1. Проверяем доступность нативных методов в EmbeddingGemma
    print("\n1️⃣ Проверка нативных методов EmbeddingGemma...")
    model_path = "C:\\Models\\ai-memory-service\\models\\embeddinggemma-300m"
    
    try:
        model = SentenceTransformer(model_path)
        has_encode_query = hasattr(model, 'encode_query')
        has_encode_document = hasattr(model, 'encode_document')
        
        print(f"   hasattr(model, 'encode_query'): {has_encode_query}")
        print(f"   hasattr(model, 'encode_document'): {has_encode_document}")
        
        if has_encode_query:
            print("   ✅ Модель поддерживает нативные методы")
        else:
            print("   📝 Модель использует fallback методы с промптами")
            
    except Exception as e:
        print(f"   ❌ Ошибка загрузки модели: {e}")
        return False
    
    # 2. Тестируем embeddings через HTTP API с разными task_type
    print("\n2️⃣ Тестирование HTTP API с разными task_type...")
    
    test_text = "How to optimize memory usage in AI systems?"
    
    async with aiohttp.ClientSession() as session:
        # Тест query embedding
        print(f"\n   📝 Тестируем query: '{test_text[:50]}...'")
        
        async with session.post(
            "http://localhost:8090/embed",
            json={"text": test_text, "task_type": "query"},
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                query_result = await response.json()
                query_embedding = query_result['embedding']
                print(f"   ✅ Query embedding: dimension={len(query_embedding)}")
                print(f"      First 3 values: {query_embedding[:3]}")
            else:
                print(f"   ❌ Query embedding failed: {response.status}")
                return False
        
        # Тест document embedding
        doc_text = "Memory optimization in AI involves using efficient data structures and algorithms to minimize RAM usage while maintaining performance."
        print(f"\n   📄 Тестируем document: '{doc_text[:50]}...'")
        
        async with session.post(
            "http://localhost:8090/embed", 
            json={"text": doc_text, "task_type": "document"},
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                doc_result = await response.json()
                doc_embedding = doc_result['embedding']
                print(f"   ✅ Document embedding: dimension={len(doc_embedding)}")
                print(f"      First 3 values: {doc_embedding[:3]}")
            else:
                print(f"   ❌ Document embedding failed: {response.status}")
                return False
        
        # Тест general embedding (без промптов)
        async with session.post(
            "http://localhost:8090/embed",
            json={"text": test_text, "task_type": "general"},
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                general_result = await response.json()
                general_embedding = general_result['embedding']
                print(f"   ✅ General embedding: dimension={len(general_embedding)}")
                print(f"      First 3 values: {general_embedding[:3]}")
            else:
                print(f"   ❌ General embedding failed: {response.status}")
                return False
    
    # 3. Сравнение эмбеддингов
    print("\n3️⃣ Анализ различий между эмбеддингами...")
    
    # Простое сравнение первых значений
    query_vs_general_diff = abs(query_embedding[0] - general_embedding[0])
    doc_vs_general_diff = abs(doc_embedding[0] - general_embedding[0])
    query_vs_doc_diff = abs(query_embedding[0] - doc_embedding[0])
    
    print(f"   📊 Разность query vs general: {query_vs_general_diff:.6f}")
    print(f"   📊 Разность doc vs general: {doc_vs_general_diff:.6f}")
    print(f"   📊 Разность query vs doc: {query_vs_doc_diff:.6f}")
    
    if query_vs_general_diff > 0.001 or doc_vs_general_diff > 0.001:
        print("   ✅ ХОРОШО: Промпты создают различающиеся эмбеддинги!")
    else:
        print("   ❌ ПРОБЛЕМА: Промпты не влияют на эмбеддинги!")
        
    print("\n" + "=" * 60)
    print("✅ Диагностика завершена")
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(test_embedding_prompts())
    except KeyboardInterrupt:
        print("\n👋 Тест прерван пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        sys.exit(1)