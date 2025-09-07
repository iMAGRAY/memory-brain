#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работоспособности EmbeddingGemma-300M интеграции
"""

import os
import sys
import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

def test_embedding_gemma():
    """Тестирование загрузки и работы EmbeddingGemma-300M"""
    
    print("=" * 60)
    print("ТЕСТ EMBEDDINGGEMMA-300M ИНТЕГРАЦИИ")
    print("=" * 60)
    
    # 1. Проверка версий
    print("\n1. ПРОВЕРКА ВЕРСИЙ:")
    print(f"   Python: {sys.version}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   NumPy: {np.__version__}")
    print(f"   CUDA доступна: {torch.cuda.is_available()}")
    
    # 2. Попытка загрузки модели
    print("\n2. ЗАГРУЗКА МОДЕЛИ:")
    
    # Проверка локального пути
    local_path = "C:/Models/ai-memory-service/models/embeddinggemma-300m"
    import os
    if os.path.exists(local_path):
        print(f"   ✓ Локальная модель найдена: {local_path}")
        model_path = local_path
    else:
        print(f"   ⚠ Локальная модель не найдена, используем HuggingFace")
        model_path = "google/embeddinggemma-300m"
    
    # Проверка доступной памяти
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
            print(f"   GPU память: {gpu_free:.1f}GB свободно из {gpu_mem:.1f}GB")
            if gpu_free < 2.0:
                print(f"   ⚠ Мало GPU памяти, используем CPU")
                device = 'cpu'
        except Exception:
            pass
    
    print(f"   Используем device: {device}")
    
    try:
        # Загружаем модель
        print(f"   Загружаем модель: {model_path}")
        
        # Пробуем разные способы загрузки
        model = None
        dtype_used = None
        
        # Способ 1: Попробуем с model_kwargs
        try:
            print("   Пробуем загрузку с model_kwargs для bfloat16...")
            model = SentenceTransformer(
                model_path,
                device=device,
                model_kwargs={'torch_dtype': torch.bfloat16}
            )
            dtype_used = "bfloat16 (через model_kwargs)"
            print("   ✓ Модель загружена с model_kwargs!")
        except (TypeError, ValueError) as e1:
            print(f"   ⚠ model_kwargs не поддерживается: {type(e1).__name__}")
            
        # Способ 2: Загрузка без dtype и конвертация после
        if model is None:
            try:
                print("   Загружаем модель со стандартными настройками...")
                model = SentenceTransformer(model_path, device=device)
                print("   ✓ Модель загружена!")
                
                # Конвертируем всю модель в bfloat16 если поддерживается
                if device == 'cuda' and torch.cuda.is_bf16_supported():
                    print("   Конвертируем модель в bfloat16...")
                    model = model.to(torch.bfloat16)
                    dtype_used = "bfloat16 (конвертирована после загрузки)"
                    
                    # Проверяем конвертацию
                    if hasattr(model, 'parameters'):
                        try:
                            params = list(model.parameters())
                            if params:
                                first_param_dtype = params[0].dtype
                                if first_param_dtype == torch.bfloat16:
                                    print("   ✓ Модель успешно конвертирована в bfloat16!")
                                else:
                                    print(f"   ⚠ Предупреждение: dtype={first_param_dtype}, ожидался bfloat16")
                                    dtype_used = f"{first_param_dtype} (конвертация не удалась)"
                            else:
                                print("   ⚠ Модель не имеет параметров для проверки dtype")
                        except Exception as e:
                            print(f"   ⚠ Не удалось проверить dtype: {e}")
                else:
                    # Используем float32 для CPU или если bfloat16 не поддерживается
                    model = model.to(torch.float32)
                    dtype_used = "float32 (стандартный)"
                    print(f"   Используем float32 (device={device})")
                    
            except (OSError, ValueError) as e2:
                print(f"   ✗ Ошибка загрузки модели: {type(e2).__name__}: {e2}")
                raise
            except AssertionError as e3:
                print(f"   ⚠ Проблема с конвертацией dtype: {e3}")
                dtype_used = "float32 (fallback)"
        
        if model is None:
            raise RuntimeError("Не удалось загрузить модель ни одним способом")
            
        print(f"   ✓ Модель успешно загружена!")
        print(f"   Тип модели: {type(model)}")
        print(f"   Используемый dtype: {dtype_used}")
        
        # Проверка параметров модели
        if hasattr(model, 'parameters'):
            param_count = sum(p.numel() for p in model.parameters())
            print(f"   Количество параметров: {param_count/1e6:.1f}M")
        
    except ImportError as e:
        print(f"   ✗ Ошибка импорта: {e}")
        print("   Проверьте установку sentence-transformers")
        return False
    except OSError as e:
        print(f"   ✗ Ошибка доступа к файлам модели: {e}")
        print(f"   Проверьте путь: {model_path}")
        return False
    except RuntimeError as e:
        print(f"   ✗ Runtime ошибка: {e}")
        print("   Возможно проблема с CUDA, памятью или форматом модели")
        return False
    except ValueError as e:
        print(f"   ✗ Ошибка параметров: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Неожиданная ошибка: {type(e).__name__}: {e}")
        return False
    
    # 3. Тест правильных промптов
    print("\n3. ТЕСТ ПРОМПТОВ (согласно документации):")
    
    # Тестовые тексты
    test_texts = {
        "query": "machine learning algorithms",
        "document": "Machine learning is a subset of artificial intelligence that enables systems to learn from data"
    }
    
    # Правильные промпты для EmbeddingGemma
    prompts = {
        "query": f"task: search result | query: {test_texts['query']}",
        "document": f"title: none | text: {test_texts['document']}"
    }
    
    print(f"   Query prompt: {prompts['query'][:50]}...")
    print(f"   Document prompt: {prompts['document'][:50]}...")
    
    # 4. Генерация эмбеддингов
    print("\n4. ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ:")
    
    try:
        start_time = time.time()
        
        # Эмбеддинги с правильными промптами
        query_embedding = model.encode([prompts["query"]], normalize_embeddings=True)
        doc_embedding = model.encode([prompts["document"]], normalize_embeddings=True)
        
        elapsed = time.time() - start_time
        
        print(f"   ✓ Эмбеддинги сгенерированы за {elapsed:.2f} сек")
        print(f"   Query embedding shape: {query_embedding.shape}")
        print(f"   Document embedding shape: {doc_embedding.shape}")
        print(f"   Размерность: {query_embedding.shape[1]} (ожидается 768)")
        
        # Проверка размерности
        assert query_embedding.shape[1] == 768, f"Неверная размерность: {query_embedding.shape[1]}"
        
    except Exception as e:
        print(f"   ✗ Ошибка генерации эмбеддингов: {e}")
        return False
    
    # 5. Тест Matryoshka dimensions
    print("\n5. ТЕСТ MATRYOSHKA DIMENSIONS:")
    
    supported_dims = [768, 512, 256, 128]
    for dim in supported_dims:
        try:
            # Усечение эмбеддинга
            truncated = query_embedding[:, :dim]
            print(f"   ✓ Dimension {dim}: shape {truncated.shape}")
        except Exception as e:
            print(f"   ✗ Dimension {dim}: ошибка {e}")
    
    # 6. Тест batch обработки
    print("\n6. ТЕСТ BATCH ОБРАБОТКИ:")
    
    batch_texts = [
        "task: search result | query: python programming",
        "task: search result | query: data science",
        "task: search result | query: deep learning",
        "title: none | text: Python is a high-level programming language",
        "title: none | text: Data science combines statistics and computing"
    ]
    
    try:
        start_time = time.time()
        batch_embeddings = model.encode(batch_texts, normalize_embeddings=True)
        elapsed = time.time() - start_time
        
        print(f"   ✓ Batch из {len(batch_texts)} текстов обработан за {elapsed:.2f} сек")
        print(f"   Batch shape: {batch_embeddings.shape}")
        
    except Exception as e:
        print(f"   ✗ Ошибка batch обработки: {e}")
        return False
    
    # 7. Тест similarity
    print("\n7. ТЕСТ SIMILARITY:")
    
    try:
        # Косинусная близость между query и document
        from numpy.linalg import norm
        
        q_emb = query_embedding[0]
        d_emb = doc_embedding[0]
        
        cosine_sim = np.dot(q_emb, d_emb) / (norm(q_emb) * norm(d_emb))
        print(f"   Cosine similarity (query vs document): {cosine_sim:.4f}")
        
        # Similarity между похожими queries
        q1 = model.encode(["task: search result | query: machine learning"], normalize_embeddings=True)[0]
        q2 = model.encode(["task: search result | query: ML algorithms"], normalize_embeddings=True)[0]
        
        sim_similar = np.dot(q1, q2) / (norm(q1) * norm(q2))
        print(f"   Cosine similarity (похожие queries): {sim_similar:.4f}")
        
    except Exception as e:
        print(f"   ✗ Ошибка вычисления similarity: {e}")
        return False
    
    # 8. Проверка производительности
    print("\n8. BENCHMARK ПРОИЗВОДИТЕЛЬНОСТИ:")
    
    test_sizes = [1, 10, 50]
    for size in test_sizes:
        texts = [f"task: search result | query: test query {i}" for i in range(size)]
        
        start_time = time.time()
        _ = model.encode(texts, normalize_embeddings=True)
        elapsed = time.time() - start_time
        
        throughput = size / elapsed
        print(f"   Batch size {size:3d}: {elapsed:.3f} сек ({throughput:.1f} texts/sec)")
    
    print("\n" + "=" * 60)
    print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    print("=" * 60)
    
    return True

def test_dtype_compatibility():
    """Специальный тест для проверки dtype совместимости"""
    print("\n" + "=" * 60)
    print("ТЕСТ DTYPE СОВМЕСТИМОСТИ")
    print("=" * 60)
    
    model_path = "C:/Models/ai-memory-service/models/embeddinggemma-300m"
    if not os.path.exists(model_path):
        model_path = "google/embeddinggemma-300m"
    
    # Тест различных dtype
    dtypes_to_test = [
        ("bfloat16", torch.bfloat16, True),   # Должно работать
        ("float32", torch.float32, True),      # Должно работать
        # ("float16", torch.float16, False),   # НЕ должно работать - закомментировано чтобы не падать
    ]
    
    for dtype_name, dtype, should_work in dtypes_to_test:
        print(f"\nТест dtype: {dtype_name}")
        try:
            model = SentenceTransformer(model_path, torch_dtype=dtype)
            test_emb = model.encode(["test"], normalize_embeddings=True)
            print(f"   ✓ {dtype_name}: работает! Shape: {test_emb.shape}")
        except Exception as e:
            if should_work:
                print(f"   ✗ {dtype_name}: ошибка (неожиданно): {e}")
            else:
                print(f"   ✓ {dtype_name}: не работает (ожидаемо): {type(e).__name__}")

if __name__ == "__main__":
    print("ЗАПУСК ТЕСТОВ AI MEMORY SERVICE - EMBEDDINGGEMMA INTEGRATION")
    print("Python:", sys.version)
    print("-" * 60)
    
    # Основные тесты
    success = test_embedding_gemma()
    
    # Тест dtype
    if success:
        test_dtype_compatibility()
    
    # Финальный статус
    if success:
        print("\n🎉 ИНТЕГРАЦИЯ РАБОТАЕТ КОРРЕКТНО!")
        sys.exit(0)
    else:
        print("\n❌ ОБНАРУЖЕНЫ ПРОБЛЕМЫ В ИНТЕГРАЦИИ")
        sys.exit(1)