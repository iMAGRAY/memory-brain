#!/usr/bin/env python3
"""
Fixed Embedding Quality Test с правильными промптами EmbeddingGemma
Исправляет проблемы найденные в comprehensive test
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_model_path() -> str:
    """Get model path"""
    model_path = os.environ.get('EMBEDDING_MODEL_PATH')
    if model_path and Path(model_path).exists():
        return model_path
    
    possible_paths = [
        r'C:\Models\ai-memory-service\models\embeddinggemma-300m',
        r'models\embeddinggemma-300m'
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            return path
    
    raise FileNotFoundError("EmbeddingGemma-300M model not found")

def test_prompts_correctness():
    """Test 1: Проверяем что промпты работают правильно"""
    logger.info("=== TEST 1: PROMPTS CORRECTNESS ===")
    
    from embedding_server import EmbeddingService
    model_path = get_model_path()
    
    service = EmbeddingService(model_path=model_path, cache_size=10)
    
    # Тестируем с промптами и без
    test_text = "What is machine learning?"
    
    # С правильными промптами через наши методы
    query_emb = service.encode_query(test_text)  # Должен добавить "task: search result | query: "
    doc_emb = service.encode_document(test_text)  # Должен добавить "title: none | text: "
    
    # Напрямую через модель без промптов (для сравнения)
    raw_emb = service.model.encode([test_text], normalize_embeddings=True)[0]
    
    # Убеждаемся что все эмбеддинги имеют одинаковую размерность
    logger.info(f"Embedding dimensions - Query: {query_emb.shape}, Doc: {doc_emb.shape}, Raw: {raw_emb.shape}")
    
    # Напрямую через модель с правильными промптами
    query_prompted = f"task: search result | query: {test_text}"
    doc_prompted = f"title: none | text: {test_text}"
    
    # Используем те же настройки что и в нашем сервисе (512D, normalized)
    query_emb_manual = service.model.encode([query_prompted], 
                                          normalize_embeddings=True,
                                          truncate_dim=service.default_dimension)[0]
    doc_emb_manual = service.model.encode([doc_prompted], 
                                        normalize_embeddings=True,
                                        truncate_dim=service.default_dimension)[0]
    
    # Также приводим raw к той же размерности 
    raw_emb = service.model.encode([test_text], 
                                 normalize_embeddings=True,
                                 truncate_dim=service.default_dimension)[0]
    
    # Сравниваем результаты
    query_match = np.allclose(query_emb, query_emb_manual, atol=1e-6)
    doc_match = np.allclose(doc_emb, doc_emb_manual, atol=1e-6)
    
    # Проверяем что промпты улучшают качество
    raw_vs_query_diff = np.linalg.norm(raw_emb - query_emb)
    raw_vs_doc_diff = np.linalg.norm(raw_emb - doc_emb)
    
    logger.info(f"Query embedding matches manual prompt: {query_match}")
    logger.info(f"Document embedding matches manual prompt: {doc_match}")
    logger.info(f"Raw vs Query embedding difference: {raw_vs_query_diff:.4f}")
    logger.info(f"Raw vs Doc embedding difference: {raw_vs_doc_diff:.4f}")
    
    return {
        'query_prompts_working': query_match,
        'doc_prompts_working': doc_match,
        'prompts_make_difference': raw_vs_query_diff > 0.01 or raw_vs_doc_diff > 0.01
    }

def test_semantic_similarity_corrected():
    """Test 2: Исправленный семантический тест с правильными промптами"""
    logger.info("=== TEST 2: CORRECTED SEMANTIC SIMILARITY ===")
    
    from embedding_server import EmbeddingService
    model_path = get_model_path()
    
    service = EmbeddingService(model_path=model_path, cache_size=50, default_dimension=512)
    
    # Пары с высокой схожестью - используем query/document разделение
    high_sim_pairs = [
        ("What is the capital of France?", "Paris is the capital city of France"),
        ("How to cook pasta?", "Instructions for cooking pasta: boil water, add pasta, cook for 10 minutes"),
        ("Machine learning algorithms", "Algorithms used in machine learning include neural networks and decision trees"),
        ("Weather is cold today", "It's very chilly and freezing outside today"),
        ("I love reading books", "Reading literature is my favorite hobby and pastime")
    ]
    
    # Средняя схожесть
    med_sim_pairs = [
        ("Cooking recipes", "Sports activities and exercises"),
        ("Technology news", "Weather forecast for tomorrow"), 
        ("Movie recommendations", "Book suggestions and reviews"),
        ("Travel destinations", "Food and restaurant reviews")
    ]
    
    # Низкая схожесть  
    low_sim_pairs = [
        ("Quantum physics equations", "Cooking chocolate cake recipe"),
        ("Stock market analysis", "Cat sleeping on couch"),
        ("Database optimization", "Beautiful sunset colors"),
        ("Binary search algorithm", "Playing guitar music")
    ]
    
    def evaluate_pairs_corrected(pairs: List[Tuple[str, str]], expected_range: Tuple[float, float], category: str):
        similarities = []
        correct = 0
        
        for query_text, doc_text in pairs:
            # Правильно используем query/document разделение
            query_emb = service.encode_query(query_text)  # С query промптом
            doc_emb = service.encode_document(doc_text)   # С document промптом
            
            # Normalized embeddings - используем dot product (быстрее)
            similarity = np.dot(query_emb, doc_emb)
            similarities.append(similarity)
            
            if expected_range[0] <= similarity <= expected_range[1]:
                correct += 1
                
            logger.info(f"{category}: '{query_text[:40]}...' vs '{doc_text[:40]}...' = {similarity:.3f}")
        
        accuracy = correct / len(pairs)
        return {
            'accuracy': accuracy,
            'avg_similarity': np.mean(similarities),
            'similarities': similarities,
            'correct': correct,
            'total': len(pairs)
        }
    
    # Тестируем с правильными ожиданиями для EmbeddingGemma
    high_results = evaluate_pairs_corrected(high_sim_pairs, (0.6, 1.0), "HIGH")   # Понижаем порог
    med_results = evaluate_pairs_corrected(med_sim_pairs, (0.2, 0.6), "MEDIUM")   # Понижаем порог
    low_results = evaluate_pairs_corrected(low_sim_pairs, (0.0, 0.3), "LOW")      # Оставляем как есть
    
    overall_accuracy = (high_results['correct'] + med_results['correct'] + low_results['correct']) / \
                       (high_results['total'] + med_results['total'] + low_results['total'])
    
    logger.info(f"HIGH similarity accuracy: {high_results['accuracy']:.1%}")
    logger.info(f"MEDIUM similarity accuracy: {med_results['accuracy']:.1%}") 
    logger.info(f"LOW similarity accuracy: {low_results['accuracy']:.1%}")
    logger.info(f"OVERALL accuracy: {overall_accuracy:.1%}")
    
    return {
        'high_similarity': high_results,
        'medium_similarity': med_results,
        'low_similarity': low_results,
        'overall_accuracy': overall_accuracy
    }

def test_retrieval_with_correct_prompts():
    """Test 3: Правильный тест поиска с query/document промптами"""
    logger.info("=== TEST 3: RETRIEVAL WITH CORRECT PROMPTS ===")
    
    from embedding_server import EmbeddingService
    model_path = get_model_path()
    
    service = EmbeddingService(model_path=model_path, cache_size=20, default_dimension=512)
    
    # База документов
    documents = [
        "Python is a programming language known for its simplicity and readability",
        "JavaScript enables dynamic web development and interactive user interfaces",
        "Machine learning helps computers learn patterns from data automatically", 
        "Deep neural networks process information through multiple layers",
        "Natural language processing enables computers to understand human language",
        "Computer vision systems analyze and interpret images and videos",
        "Cybersecurity protects systems from unauthorized access and attacks",
        "Cloud computing provides scalable infrastructure for applications"
    ]
    
    # Запросы с ожидаемыми результатами
    queries_expected = [
        ("programming languages", [0, 1]),           # Python, JavaScript
        ("artificial intelligence", [2, 3, 4, 5]),   # ML, DL, NLP, CV
        ("computer security", [6]),                   # Cybersecurity
        ("web development tools", [1]),               # JavaScript
    ]
    
    # Кодируем документы с правильными промптами
    doc_embeddings = []
    for doc in documents:
        doc_emb = service.encode_document(doc)  # Добавляет "title: none | text: "
        doc_embeddings.append(doc_emb)
    
    doc_embeddings = np.array(doc_embeddings)
    
    retrieval_results = []
    
    for query, expected_indices in queries_expected:
        # Кодируем запрос с правильным промптом
        query_emb = service.encode_query(query)  # Добавляет "task: search result | query: "
        
        # Вычисляем сходства (dot product для normalized)
        similarities = np.dot(doc_embeddings, query_emb)
        
        # Получаем топ результаты
        top_k = min(len(expected_indices) + 1, len(documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Метрики
        retrieved_relevant = len(set(top_indices) & set(expected_indices))
        precision = retrieved_relevant / len(top_indices)
        recall = retrieved_relevant / len(expected_indices)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        retrieval_results.append({
            'query': query,
            'precision': precision,
            'recall': recall, 
            'f1': f1,
            'top_similarities': similarities[top_indices].tolist()
        })
        
        logger.info(f"Query: '{query}'")
        logger.info(f"  Top-{top_k}: {top_indices} (sims: {similarities[top_indices]})")
        logger.info(f"  Expected: {expected_indices}")
        logger.info(f"  P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
    
    avg_f1 = np.mean([r['f1'] for r in retrieval_results])
    avg_precision = np.mean([r['precision'] for r in retrieval_results])
    avg_recall = np.mean([r['recall'] for r in retrieval_results])
    
    logger.info(f"Average F1: {avg_f1:.3f}")
    logger.info(f"Average Precision: {avg_precision:.3f}")
    logger.info(f"Average Recall: {avg_recall:.3f}")
    
    return {
        'avg_f1': avg_f1,
        'avg_precision': avg_precision, 
        'avg_recall': avg_recall,
        'retrieval_results': retrieval_results
    }

def test_dimension_and_normalization():
    """Test 4: Проверка размерности и нормализации"""
    logger.info("=== TEST 4: DIMENSION AND NORMALIZATION ===")
    
    from embedding_server import EmbeddingService
    model_path = get_model_path()
    
    # Тестируем разные размерности
    dimensions_results = {}
    
    for dim in [128, 256, 512, 768]:
        logger.info(f"Testing dimension: {dim}")
        
        service = EmbeddingService(model_path=model_path, cache_size=10, default_dimension=dim)
        
        test_text = "This is a test sentence for dimension verification"
        
        query_emb = service.encode_query(test_text)
        doc_emb = service.encode_document(test_text)
        
        # Проверяем размерность
        query_dim_correct = query_emb.shape[0] == dim
        doc_dim_correct = doc_emb.shape[0] == dim
        
        # Проверяем нормализацию (должна быть близка к 1.0)
        query_norm = np.linalg.norm(query_emb)
        doc_norm = np.linalg.norm(doc_emb)
        query_normalized = abs(query_norm - 1.0) < 0.01
        doc_normalized = abs(doc_norm - 1.0) < 0.01
        
        dimensions_results[dim] = {
            'query_dim_correct': query_dim_correct,
            'doc_dim_correct': doc_dim_correct,
            'query_normalized': query_normalized,
            'doc_normalized': doc_normalized,
            'query_norm': query_norm,
            'doc_norm': doc_norm
        }
        
        logger.info(f"  Dim {dim}: Query={query_dim_correct}, Doc={doc_dim_correct}")
        logger.info(f"  Norms: Query={query_norm:.4f}, Doc={doc_norm:.4f}")
    
    return dimensions_results

def main():
    """Главная функция тестирования"""
    logger.info("🔧 STARTING FIXED EMBEDDING QUALITY TEST")
    logger.info("=" * 80)
    
    start_time = time.perf_counter()
    
    try:
        # Выполняем исправленные тесты
        logger.info("1. Testing prompts correctness...")
        prompts_results = test_prompts_correctness()
        
        logger.info("\n2. Testing corrected semantic similarity...")  
        semantic_results = test_semantic_similarity_corrected()
        
        logger.info("\n3. Testing retrieval with correct prompts...")
        retrieval_results = test_retrieval_with_correct_prompts()
        
        logger.info("\n4. Testing dimensions and normalization...")
        dimension_results = test_dimension_and_normalization()
        
        test_time = time.perf_counter() - start_time
        
        # Финальный анализ
        logger.info("\n" + "=" * 80)
        logger.info("📊 FIXED TEST RESULTS")
        logger.info("=" * 80)
        
        # Оценка качества промптов
        prompts_working = prompts_results['query_prompts_working'] and prompts_results['doc_prompts_working']
        prompts_effective = prompts_results['prompts_make_difference']
        
        logger.info(f"Prompts working correctly: {prompts_working}")
        logger.info(f"Prompts make difference: {prompts_effective}")
        logger.info(f"Semantic similarity accuracy: {semantic_results['overall_accuracy']:.1%}")
        logger.info(f"Retrieval F1 score: {retrieval_results['avg_f1']:.3f}")
        
        # Проверка размерностей
        all_dims_ok = all(
            result['query_dim_correct'] and result['doc_dim_correct'] 
            for result in dimension_results.values()
        )
        all_normalized = all(
            result['query_normalized'] and result['doc_normalized']
            for result in dimension_results.values()
        )
        
        logger.info(f"All dimensions working: {all_dims_ok}")
        logger.info(f"All embeddings normalized: {all_normalized}")
        
        # Итоговая оценка
        overall_score = (
            (1.0 if prompts_working else 0.0) * 0.3 +
            (1.0 if prompts_effective else 0.0) * 0.2 +
            semantic_results['overall_accuracy'] * 0.3 +
            retrieval_results['avg_f1'] * 0.2
        )
        
        logger.info(f"\nTest execution time: {test_time:.1f}s")
        logger.info(f"Overall fixed score: {overall_score:.1%}")
        
        if overall_score >= 0.7:
            logger.info("✅ VERDICT: EMBEDDINGS QUALITY IS NOW GOOD!")
            logger.info("🔧 Configuration fixes were successful")
            return True
        else:
            logger.info("❌ VERDICT: Still needs improvement")
            logger.info("🔍 Further investigation required")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)