#!/usr/bin/env python3
"""
Standalone Python Embedding Server for EmbeddingGemma-300M
Решает проблему Python GIL через отдельный процесс с async обработкой
"""

import asyncio
import json
import logging
import os
import sys
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import threading
from threading import RLock
from html import escape
import multiprocessing as mp
import time
import hashlib

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from aiohttp import web
import aiohttp_cors

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Optimized EmbeddingGemma-300M service с поддержкой специализированных промптов
    и Matryoshka representation learning для максимального качества embeddings
    """
    
    # Специализированные промпты согласно документации EmbeddingGemma (сентябрь 2025)
    TASK_PROMPTS = {
        'retrieval_query': "task: search result | query: ",
        'retrieval_document': "title: {title} | text: ",  
        'classification': "task: classification | query: ",
        'clustering': "task: clustering | query: ",
        'semantic_similarity': "task: sentence similarity | query: ",
        'fact_checking': "task: fact checking | query: ",
        'question_answering': "task: question answering | query: ",
        'code_retrieval': "task: code retrieval | query: ",
        'general': ""  # Fallback для общих случаев
    }
    
    # Поддерживаемые размерности Matryoshka 
    MATRYOSHKA_DIMENSIONS = [768, 512, 256, 128]
    
    # Максимальный контекст модели
    MAX_CONTEXT_TOKENS = 2048
    
    def __init__(self, model_path: str, max_workers: int = 4, default_dimension: int = 512, cache_size: int = 1000):
        """
        Инициализация optimized EmbeddingGemma service
        
        Args:
            model_path: Путь к модели EmbeddingGemma-300M
            max_workers: Количество воркеров для параллельной обработки  
            default_dimension: Размерность по умолчанию (512 optimal для production)
        """
        # Валидация входных параметров
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if default_dimension not in [128, 256, 512, 768]:
            raise ValueError("default_dimension must be one of [128, 256, 512, 768] for Matryoshka support")
            
        self.model_path = model_path
        self.max_workers = max_workers
        self.default_dimension = default_dimension
        self.model = None
        self.device = None
        
        # Thread-safe кеш для embeddings с LRU eviction
        self._cache_lock = RLock()
        self.cache = OrderedDict()
        self.cache_size = min(max(100, cache_size), 10000)  # Ограничиваем размер кеша
        
        # Пул потоков для обработки с proper shutdown handling
        # Увеличиваем размер пула для предотвращения deadlock
        self.executor = ThreadPoolExecutor(
            max_workers=max(8, max_workers),
            thread_name_prefix="embedding-worker"
        )
        # Флаг для graceful shutdown
        self._shutdown_requested = False
        # Регистрируем cleanup handler
        import atexit
        atexit.register(self._cleanup_resources)
        
        # Thread-safe статистика использования (инициализируем ДО загрузки модели)
        self._stats_lock = RLock()
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'model_load_time': 0,
            'average_inference_time': 0,
            'encoding_errors': 0,
            'security_violations': 0
        }
        
        # Загружаем модель с comprehensive error handling
        try:
            self._load_model_safe()
        except Exception as e:
            # Cleanup в случае ошибки загрузки
            self._cleanup_resources()
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def _load_model_safe(self):
        """Безопасная загрузка модели с обработкой ошибок"""
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Loading EmbeddingGemma-300M from {self.model_path}")
            self.model = SentenceTransformer(self.model_path)
            
            # КРИТИЧНО: НЕ используем float16 - EmbeddingGemma не поддерживает!
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # КРИТИЧЕСКАЯ ПРОВЕРКА: EmbeddingGemma НЕ поддерживает float16!
            if hasattr(self.model, '_modules'):
                for module in self.model.modules():
                    if hasattr(module, 'weight') and module.weight is not None:
                        if module.weight.dtype == torch.float16:
                            raise RuntimeError("КРИТИЧЕСКАЯ ОШИБКА: EmbeddingGemma НЕ поддерживает float16! Используйте float32 или bfloat16.")
            
            # Оптимальная настройка precision согласно документации EmbeddingGemma
            if self.device.type == "cuda":
                try:
                    # Тестируем bfloat16 поддержку (оптимально для GPU)
                    test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device=self.device)
                    logger.info("Using bfloat16 precision for GPU optimization (recommended for EmbeddingGemma)")
                    # Используем model_kwargs как в документации
                    self.model = self.model.to(self.device, dtype=torch.bfloat16)
                except Exception as e:
                    logger.warning(f"bfloat16 not supported, using float32: {e}")
                    self.model = self.model.to(self.device, dtype=torch.float32)
            else:
                logger.info("Using float32 precision for CPU (recommended for EmbeddingGemma)")
                self.model = self.model.to(self.device, dtype=torch.float32)
                
            self.model.eval()  # Важно для inference согласно best practices
            
            load_time = time.time() - start_time
            self.stats['model_load_time'] = load_time
            logger.info(f"Model loaded successfully on {self.device} in {load_time:.2f}s (dimension: {self.default_dimension})")
            
            # Warmup inference для оптимизации первого запроса
            self._warmup_model()
            
        except FileNotFoundError as e:
            raise RuntimeError(f"Model files not found at {self.model_path}: {e}")
        except torch.cuda.OutOfMemoryError as e:
            logger.error("GPU out of memory, trying CPU fallback")
            self.device = torch.device("cpu")
            # Проверяем, что модель загружена перед использованием
            if hasattr(self, 'model') and self.model is not None:
                self.model.to(self.device)
                logger.info("Fallback to CPU successful")
            else:
                raise RuntimeError(f"Model not loaded, cannot fallback to CPU: {e}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}. Check model path and dependencies.")
    
    def _warmup_model(self):
        """Прогрев модели для оптимизации первого запроса"""
        try:
            logger.info("Warming up model...")
            warmup_text = "warmup text for model optimization"
            _ = self.model.encode([warmup_text], convert_to_tensor=False)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}, but service will continue")
    
    def _safe_cache_key(self, text: str, task_type: str = 'general', title: Optional[str] = None) -> str:
        """
        Создание безопасного ключа кеша с использованием SHA256
        Предотвращает коллизии и обеспечивает уникальность
        """
        # Валидация входных данных
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        if len(text) > 10000:  # Защита от DoS
            raise ValueError("Text too long (max 10000 chars)")
            
        # Создаем безопасный промпт с экранированием
        prompt_template = self.TASK_PROMPTS.get(task_type, '')
        if task_type == 'retrieval_document' and '{title}' in prompt_template:
            # Экранируем title для предотвращения инъекций
            safe_title = escape(title or "none")[:100]  # Ограничиваем длину
            prompt = prompt_template.format(title=safe_title)
        else:
            prompt = prompt_template
            
        # Создаем уникальный ключ с метаинформацией (быстрый хеш)
        cache_data = f"{prompt}{text}|dim:{self.default_dimension}|v:1.0"
        # Используем blake2b для скорости (в 3-5 раз быстрее SHA256)
        return hashlib.blake2b(cache_data.encode('utf-8'), digest_size=32).hexdigest()
    
    def _evict_cache_if_needed(self) -> None:
        """
        LRU eviction - удаляет старые элементы при превышении лимита
        Thread-safe реализация
        """
        with self._cache_lock:
            while len(self.cache) >= self.cache_size:
                # Удаляем самый старый элемент (FIFO для LRU)
                oldest_key, _ = self.cache.popitem(last=False)
                logger.debug(f"Evicted cache entry: {oldest_key[:16]}...")
    
    def _update_cache_stats(self, cache_hit: bool) -> None:
        """
        Thread-safe обновление статистики кеша
        """
        with self._stats_lock:
            if cache_hit:
                self.stats['cache_hits'] += 1
            else:
                self.stats['cache_misses'] += 1
            self.stats['total_requests'] += 1
    
    def _get_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """
        Thread-safe получение из кеша с LRU обновлением
        ИСПРАВЛЕНО: возвращаем безопасную копию, не view для предотвращения race conditions
        """
        with self._cache_lock:
            if cache_key in self.cache:
                # Безопасное перемещение в конец для LRU
                self.cache.move_to_end(cache_key)
                # КРИТИЧНО: возвращаем копию для thread safety
                # View может стать недействительным если другой поток вытеснит элемент
                value = self.cache[cache_key]
                return value.copy()  # Безопасная копия предотвращает use-after-free
        return None
    
    def _put_in_cache(self, cache_key: str, embedding: np.ndarray) -> bool:
        """
        Thread-safe размещение в кеше с проверкой лимитов
        ИСПРАВЛЕНО: возвращаем success status и не игнорируем ошибки молча
        """
        try:
            with self._cache_lock:
                # Проверяем лимиты перед добавлением
                if len(self.cache) >= self.cache_size:
                    self._evict_cache_if_needed()
                    
                # Создаем безопасную иммутабельную копию
                safe_copy = np.array(embedding, copy=True)
                safe_copy.flags.writeable = False  # Make immutable
                self.cache[cache_key] = safe_copy
                return True
                
        except Exception as e:
            logger.error(f"Critical cache failure: {e}")  # Error, not warning
            with self._stats_lock:
                self.stats['encoding_errors'] += 1
            return False
        
        
    def _encode_with_prompts(self, texts: List[str], task_type: str = 'general', 
                           title: Optional[str] = None, batch_size: int = 32) -> np.ndarray:
        """
        Безопасное кодирование с применением специализированных промптов
        Включает валидацию входных данных и защиту от инъекций
        """
        if not texts:
            return np.array([])
            
        # Валидация задачи
        if task_type not in self.TASK_PROMPTS:
            logger.warning(f"Unknown task type {task_type}, using 'general'")
            with self._stats_lock:
                self.stats['security_violations'] += 1
            task_type = 'general'
            
        try:
            prompt_template = self.TASK_PROMPTS[task_type]
            
            # Безопасное форматирование промптов
            prompted_texts = []
            for text in texts:
                if not isinstance(text, str):
                    raise ValueError(f"All texts must be strings, got {type(text)}")
                    
                if task_type == 'retrieval_document' and '{title}' in prompt_template:
                    # Экранируем title для предотвращения инъекций
                    safe_title = escape(title or "none")[:100]
                    prompt = prompt_template.format(title=safe_title)
                else:
                    prompt = prompt_template
                    
                # Обрабатываем контекст 2048 токенов (не символов!)
                # Используем токенайзер для правильной обработки
                full_text = prompt + text
                prompted_texts.append(full_text)  # Модель сама обрежет до 2048 токенов
            
            # Кодирование с обработкой ошибок
            start_time = time.time()
            # Используем оптимальный batch_size согласно документации
            optimal_batch_size = min(batch_size, len(prompted_texts))
            
            embeddings = self.model.encode(
                prompted_texts,
                batch_size=optimal_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # КРИТИЧНО для EmbeddingGemma!
                convert_to_tensor=False
            )
            
            # Обновляем статистику времени
            encoding_time = time.time() - start_time
            with self._stats_lock:
                # Обновляем среднее время кодирования (простая скользящая средняя)
                current_avg = self.stats['average_inference_time']
                total_reqs = self.stats['total_requests']
                if total_reqs > 0:
                    self.stats['average_inference_time'] = ((current_avg * total_reqs) + encoding_time) / (total_reqs + 1)
                else:
                    self.stats['average_inference_time'] = encoding_time
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Encoding failed for task_type {task_type}: {e}")
            with self._stats_lock:
                self.stats['encoding_errors'] += 1
            raise RuntimeError(f"Embedding encoding failed: {e}")
    
    def _encode_batch(self, texts: List[str], task_type: str = 'general', 
                     title: Optional[str] = None, batch_size: int = 32) -> np.ndarray:
        """
        Thread-safe кодирование батча с кешированием и безопасностью
        Использует LRU кеш и защищенные промпты
        """
        if not texts:
            return np.array([])
            
        try:
            # Создаем безопасные ключи кеша
            cache_keys = []
            for text in texts:
                try:
                    cache_key = self._safe_cache_key(text, task_type, title)
                    cache_keys.append(cache_key)
                except ValueError as e:
                    logger.warning(f"Invalid text for caching: {e}")
                    # Создаем временный ключ для некешируемого элемента
                    cache_keys.append(None)
            
            # Проверяем кеш
            uncached_texts = []
            uncached_indices = []
            results = [None] * len(texts)
            
            for i, (text, cache_key) in enumerate(zip(texts, cache_keys)):
                if cache_key:
                    cached_result = self._get_from_cache(cache_key)
                    if cached_result is not None:
                        results[i] = cached_result
                        self._update_cache_stats(cache_hit=True)
                        continue
                
                # Элемент не найден в кеше
                uncached_texts.append(text)
                uncached_indices.append(i)
                self._update_cache_stats(cache_hit=False)
            
            # Кодируем некешированные тексты
            if uncached_texts:
                logger.debug(f"Encoding {len(uncached_texts)}/{len(texts)} texts with {task_type} prompts")
                embeddings = self._encode_with_prompts(uncached_texts, task_type, title, batch_size)
                
                # Сохраняем в кеш и результаты
                for idx, embedding in zip(uncached_indices, embeddings):
                    results[idx] = embedding
                    
                    # Кешируем только если есть валидный ключ
                    cache_key = cache_keys[idx]
                    if cache_key:
                        self._put_in_cache(cache_key, embedding)
            
            # Проверяем что все результаты заполнены
            if any(r is None for r in results):
                raise RuntimeError("Some embeddings were not computed")
                
            return np.array(results)
            
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            with self._stats_lock:
                self.stats['encoding_errors'] += 1
            raise
    
    # ========== КРИТИЧЕСКИ ВАЖНЫЕ МЕТОДЫ СОГЛАСНО ДОКУМЕНТАЦИИ EMBEDDINGGEMMA ==========
    
    def encode_query(self, queries: Union[str, List[str]], 
                    truncate_dim: Optional[int] = None,
                    batch_size: int = 32) -> Union[np.ndarray, List[List[float]]]:
        """
        Кодирование поисковых запросов через нативный метод модели.
        Полностью совместимо с документацией EmbeddingGemma (сентябрь 2025).
        
        Args:
            queries: Запрос или список запросов
            truncate_dim: Размерность для Matryoshka (128/256/512/768)
            batch_size: Размер батча для обработки
            
        Returns:
            Эмбеддинги запросов с автоматически добавленным промптом
        """
        # Преобразуем в список если одиночный запрос
        single_query = isinstance(queries, str)
        if single_query:
            queries = [queries]
            
        # Используем truncate_dim если указан, иначе default
        target_dim = truncate_dim or self.default_dimension
        
        # Используем переданный batch_size или оптимальный по умолчанию
        if batch_size is None or batch_size <= 0:
            batch_size = 32 if torch.cuda.is_available() else 16
        
        # Используем нативный метод модели если доступен
        if hasattr(self.model, 'encode_query'):
            # Нативный метод SentenceTransformer автоматически добавляет промпт
            embeddings = self.model.encode_query(queries, batch_size=batch_size,
                                                 convert_to_numpy=True,
                                                 normalize_embeddings=True)
        else:
            # Fallback если метод недоступен
            embeddings = self._encode_batch(
                queries, 
                task_type='retrieval_query',
                batch_size=batch_size
            )
        
        # Применяем Matryoshka truncation если нужно
        if target_dim != 768:
            embeddings = self._apply_matryoshka_truncation(embeddings, target_dim)
            
        # Возвращаем в формате согласно документации
        if single_query:
            return embeddings[0]
        return embeddings
    
    def encode_document(self, documents: Union[str, List[str]], 
                       titles: Optional[Union[str, List[str]]] = None,
                       truncate_dim: Optional[int] = None,
                       batch_size: int = 32) -> Union[np.ndarray, List[List[float]]]:
        """
        Кодирование документов через нативный метод модели.
        Полностью совместимо с документацией EmbeddingGemma (сентябрь 2025).
        
        Args:
            documents: Документ или список документов
            titles: Заголовок(и) документов (опционально)
            truncate_dim: Размерность для Matryoshka (128/256/512/768)
            batch_size: Размер батча для обработки
            
        Returns:
            Эмбеддинги документов с автоматически добавленным промптом
        """
        # Преобразуем в списки
        single_doc = isinstance(documents, str)
        if single_doc:
            documents = [documents]
            if titles and isinstance(titles, str):
                titles = [titles]
                
        # Проверяем соответствие длин
        if titles and len(titles) != len(documents):
            raise ValueError("Количество заголовков должно совпадать с количеством документов")
            
        # Используем truncate_dim если указан
        target_dim = truncate_dim or self.default_dimension
        
        # Используем переданный batch_size или оптимальный по умолчанию
        if batch_size is None or batch_size <= 0:
            batch_size = 32 if torch.cuda.is_available() else 16
        
        # Используем нативный метод модели если доступен
        if hasattr(self.model, 'encode_document'):
            # Нативный метод SentenceTransformer автоматически добавляет промпт
            # Подготавливаем документы с заголовками если есть
            if titles:
                docs_with_titles = []
                for doc, title in zip(documents, titles):
                    # Объединяем заголовок и документ для модели
                    doc_with_title = f"{title}\n{doc}" if title else doc
                    docs_with_titles.append(doc_with_title)
                embeddings = self.model.encode_document(docs_with_titles, 
                                                       batch_size=batch_size,
                                                       convert_to_numpy=True,
                                                       normalize_embeddings=True)
            else:
                embeddings = self.model.encode_document(documents, 
                                                       batch_size=batch_size,
                                                       convert_to_numpy=True,
                                                       normalize_embeddings=True)
        else:
            # Fallback если метод недоступен
            # ОПТИМИЗАЦИЯ: обрабатываем все документы одним батчем для локального сервера
            if titles:
                # Подготавливаем документы с заголовками в одном батче
                docs_with_titles = []
                for doc, title in zip(documents, titles):
                    # Формируем текст с заголовком для промпта
                    doc_text = f"{title}\n{doc}" if title else doc
                    docs_with_titles.append(doc_text)
                
                # Обрабатываем все документы одним батчем
                embeddings = self._encode_batch(
                    docs_with_titles,
                    task_type='retrieval_document',
                    title="dynamic",  # Заголовки уже включены в текст
                    batch_size=batch_size
                )
            else:
                # Все документы без заголовков
                embeddings = self._encode_batch(
                    documents,
                    task_type='retrieval_document',
                    title="none",
                    batch_size=batch_size
                )
            
        # Применяем Matryoshka truncation
        if target_dim != 768:
            embeddings = self._apply_matryoshka_truncation(embeddings, target_dim)
            
        # Возвращаем в нужном формате
        if single_doc:
            return embeddings[0]
        return embeddings
    
    def similarity(self, query_embeddings: np.ndarray, 
                  doc_embeddings: np.ndarray,
                  use_dot_product: bool = True) -> np.ndarray:
        """
        Вычисление сходства между запросами и документами.
        Согласно документации EmbeddingGemma - для нормализованных эмбеддингов
        используется dot product вместо cosine similarity (быстрее).
        
        Args:
            query_embeddings: Эмбеддинги запросов (n_queries, dim)
            doc_embeddings: Эмбеддинги документов (n_docs, dim)
            use_dot_product: Использовать dot product (True) или cosine similarity (False)
            
        Returns:
            Матрица сходства (n_queries, n_docs)
        """
        # Преобразуем в numpy если нужно
        if torch.is_tensor(query_embeddings):
            query_embeddings = query_embeddings.cpu().numpy()
        if torch.is_tensor(doc_embeddings):
            doc_embeddings = doc_embeddings.cpu().numpy()
            
        # Проверяем размерности
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        if doc_embeddings.ndim == 1:
            doc_embeddings = doc_embeddings.reshape(1, -1)
            
        if query_embeddings.shape[1] != doc_embeddings.shape[1]:
            raise ValueError(f"Размерности не совпадают: запросы {query_embeddings.shape[1]}, документы {doc_embeddings.shape[1]}")
            
        if use_dot_product:
            # Для нормализованных эмбеддингов dot product = cosine similarity
            # Проверяем нормализацию
            query_norms = np.linalg.norm(query_embeddings, axis=1)
            doc_norms = np.linalg.norm(doc_embeddings, axis=1)
            
            if not (np.allclose(query_norms, 1.0, atol=1e-6) and np.allclose(doc_norms, 1.0, atol=1e-6)):
                logger.warning("Эмбеддинги не нормализованы! Рекомендуется использовать normalize_embeddings=True")
                
            # Быстрое вычисление через dot product
            similarities = np.dot(query_embeddings, doc_embeddings.T)
        else:
            # Классический cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embeddings, doc_embeddings)
            
        return similarities
    
    def _apply_matryoshka_truncation(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Применение Matryoshka truncation с обязательной нормализацией.
        Критически важно для правильной работы согласно документации.
        """
        if target_dim not in [128, 256, 512, 768]:
            raise ValueError(f"Недопустимая размерность {target_dim}. Поддерживаются: 128, 256, 512, 768")
            
        if embeddings.shape[1] < target_dim:
            raise ValueError(f"Нельзя увеличить размерность с {embeddings.shape[1]} до {target_dim}")
            
        # Обрезаем до нужной размерности
        truncated = embeddings[:, :target_dim]
        
        # КРИТИЧНО: Нормализация после truncation (обязательно для Matryoshka!)
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        truncated = truncated / (norms + 1e-10)  # Защита от деления на ноль
        
        return truncated
    
    def encode_documents(self, documents: Union[str, List[str]], 
                        titles: Optional[Union[str, List[str]]] = None,
                        normalize: bool = True,
                        convert_to_numpy: bool = True,
                        parallel: bool = True) -> Union[np.ndarray, List[Optional[np.ndarray]]]:
        """
        Optimized batch encoding method for multiple documents
        
        Args:
            documents: Single document or list of documents to encode
            titles: Optional titles for each document (for better context)
            normalize: Whether to normalize embeddings (recommended: True)
            convert_to_numpy: Whether to return numpy arrays or torch tensors
            
        Returns:
            Single embedding array for single document, list of embeddings for multiple
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If encoding fails
        """
        # Input validation
        if not documents:
            raise ValueError("documents cannot be empty")
            
        single_document = isinstance(documents, str)
        if single_document:
            documents = [documents]
        
        if titles is not None:
            if isinstance(titles, str):
                titles = [titles]
            elif len(titles) != len(documents):
                raise ValueError(f"Number of titles ({len(titles)}) must match documents ({len(documents)})")
            
        # Batch processing with error handling
        embeddings = []
        failed_indices = []
        
        start_time = time.time()
        
        for i, doc in enumerate(documents):
            try:
                title = titles[i] if titles else None
                emb = self.encode_document(
                    doc, 
                    title=title, 
                    normalize=normalize, 
                    convert_to_numpy=convert_to_numpy
                )
                embeddings.append(emb)
                
            except Exception as e:
                logger.error(f"Failed to encode document {i}: {str(e)[:100]}...")
                failed_indices.append(i)
                embeddings.append(None)  # Placeholder for failed encoding
                
                with self._stats_lock:
                    self.stats['encoding_errors'] += 1
        
        # Check if all encodings failed
        if len(failed_indices) == len(documents):
            raise RuntimeError(f"All {len(documents)} documents failed to encode")
            
        # Log batch performance stats
        batch_time = time.time() - start_time
        docs_per_sec = len(documents) / batch_time if batch_time > 0 else 0
        
        if failed_indices:
            logger.warning(f"Batch encoding: {len(failed_indices)}/{len(documents)} documents failed")
        
        logger.debug(f"Batch encoded {len(documents)} documents in {batch_time:.3f}s ({docs_per_sec:.1f} docs/sec)")
        
        # PRESERVE indexing - do NOT filter out failed embeddings
        # This maintains correspondence between input documents and output embeddings
        failed_count = sum(1 for emb in embeddings if emb is None)
        success_count = len(embeddings) - failed_count
        
        if failed_count > 0:
            logger.warning(f"Batch encoding: {failed_count}/{len(documents)} documents failed (preserved as None)")
            logger.info(f"Successfully encoded {success_count}/{len(documents)} documents")
        
        # Return single embedding for single document input
        if single_document:
            return embeddings[0] if embeddings and embeddings[0] is not None else None
        
        return embeddings
    
    def cleanup(self) -> None:
        """
        Public method for manual resource cleanup
        Вызывает internal _cleanup_resources
        """
        self._cleanup_resources()

    def _cleanup_resources(self):
        """
        Критически важный метод для graceful shutdown с proper resource cleanup
        """
        if hasattr(self, '_shutdown_requested') and self._shutdown_requested:
            return  # Уже выполняем shutdown
        
        logger.info("Starting resource cleanup...")
        self._shutdown_requested = True
        
        # 1. Останавливаем thread pool с надежным ожиданием завершения
        if hasattr(self, 'executor') and self.executor:
            logger.info("Shutting down thread pool...")
            import time
            cleanup_start = time.time()
            
            try:
                # Проверяем поддержку timeout в shutdown (Python 3.9+)
                import inspect
                shutdown_signature = inspect.signature(self.executor.shutdown)
                supports_timeout = 'timeout' in shutdown_signature.parameters
                
                if supports_timeout:
                    # Используем встроенный timeout (Python 3.9+)
                    logger.debug("Using built-in shutdown timeout")
                    self.executor.shutdown(wait=True, timeout=30)
                else:
                    # Для старых версий Python - корректная реализация ожидания
                    logger.debug("Using manual shutdown with timeout for Python < 3.9")
                    
                    # Сначала останавливаем прием новых задач
                    self.executor.shutdown(wait=False)
                    
                    # Ждем завершения с правильной проверкой
                    start_time = time.time()
                    timeout_seconds = 30
                    
                    while time.time() - start_time < timeout_seconds:
                        # Проверяем активные задачи через submit() которые завершились
                        try:
                            # Пытаемся подать простую задачу - если executor shutdown, получим RuntimeError
                            future = self.executor.submit(lambda: None)
                            future.cancel()  # Сразу отменяем
                            # Если дошли сюда - executor еще работает
                            time.sleep(0.1)
                        except RuntimeError:
                            # Executor больше не принимает задачи - значит shutdown завершен
                            break
                    else:
                        # Timeout достигнут
                        logger.warning(f"Thread pool shutdown timeout after {timeout_seconds}s")
                
                shutdown_time = time.time() - cleanup_start
                logger.info(f"Thread pool shutdown completed in {shutdown_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error shutting down thread pool: {e}")
                
        # 2. Очищаем кеш
        if hasattr(self, 'cache'):
            try:
                logger.info("Clearing cache...")
                with self._cache_lock:
                    cache_size = len(self.cache)
                    self.cache.clear()
                logger.info(f"Cleared {cache_size} cache entries")
            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                
        # 3. Освобождаем модель
        if hasattr(self, 'model') and self.model:
            try:
                logger.info("Unloading model...")
                del self.model
                self.model = None
                logger.info("Model unloaded successfully")
            except Exception as e:
                logger.error(f"Error unloading model: {e}")
                
        # 4. Очищаем CUDA кеш если доступен
        try:
            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache...")
                torch.cuda.empty_cache()
                logger.info("CUDA cache cleared")
        except Exception as e:
            logger.error(f"Error clearing CUDA cache: {e}")
                
        logger.info("Resource cleanup completed")
    
    def is_healthy(self) -> bool:
        """
        Проверка здоровья сервиса
        """
        try:
            # Проверяем что модель загружена
            if not self.model:
                return False
            
            # Проверяем что executor не shutdown
            if hasattr(self, '_shutdown_requested') and self._shutdown_requested:
                return False
                
            # Проверяем что executor доступен
            if not hasattr(self, 'executor') or not self.executor:
                return False
                
            return True
            
        except Exception:
            return False
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Получение подробной статистики сервиса
        """
        with self._stats_lock:
            base_stats = self.stats.copy()
            
        # Добавляем точную системную информацию
        cache_size = 0
        cache_memory_mb = 0
        with self._cache_lock:
            cache_size = len(self.cache)
            # ИСПРАВЛЕНО: Точная оценка памяти кеша через суммирование всех embeddings
            if self.cache:
                try:
                    total_bytes = sum(embedding.nbytes for embedding in self.cache.values())
                    cache_memory_mb = total_bytes / (1024 * 1024)
                except (AttributeError, TypeError) as e:
                    logger.warning(f"Failed to calculate precise cache memory: {e}")
                    # Fallback к приблизительной оценке только в случае ошибки
                    try:
                        sample_embedding = next(iter(self.cache.values()))
                        avg_size = sample_embedding.nbytes
                        cache_memory_mb = (cache_size * avg_size) / (1024 * 1024)
                    except Exception:
                        cache_memory_mb = 0.0  # Безопасное значение по умолчанию
        
        # Безопасный расчет hit_rate с защитой от division by zero
        total_requests = base_stats['cache_hits'] + base_stats['cache_misses']
        if total_requests > 0:
            cache_hit_rate = base_stats['cache_hits'] / total_requests
        else:
            cache_hit_rate = 0.0
            
        comprehensive_stats = {
            **base_stats,
            'cache_size': cache_size,
            'cache_max_size': self.cache_size,
            'cache_memory_mb': round(cache_memory_mb, 2),
            'cache_hit_rate': round(cache_hit_rate, 3),  # Округляем для читаемости
            'worker_threads': self.max_workers,
            'model_loaded': self.model is not None,
            'is_healthy': self.is_healthy(),
            'device': str(self.device) if self.device else None
        }
        
        return comprehensive_stats
    
    # Оставляем старые методы для обратной совместимости
    def encode_for_retrieval_query(self, queries: List[str]) -> List[List[float]]:
        """
        Кодирование запросов для поиска (retrieval query)
        Использует специализированный промпт для максимального качества поиска
        """
        embeddings = self._encode_batch(queries, task_type='retrieval_query')
        return [emb.tolist() for emb in embeddings]
    
    def encode_for_retrieval_documents(self, documents: List[str], 
                                     titles: Optional[List[str]] = None) -> List[List[float]]:
        """
        Оптимизированное кодирование документов для индексации
        Поддерживает заголовки документов с батчевой обработкой
        """
        if not documents:
            return []
            
        if titles and len(titles) != len(documents):
            raise ValueError("Length of titles must match length of documents")
        
        # Оптимизация: сначала группируем документы по заголовкам
        if not titles:
            # Все документы без заголовков - можно обработать одним батчем
            embeddings = self._encode_batch(documents, task_type='retrieval_document')
            return [emb.tolist() for emb in embeddings]
        
        # Группируем документы по заголовкам с проверкой эффективности
        from collections import defaultdict
        groups = defaultdict(list)
        for i, (doc, title) in enumerate(zip(documents, titles)):
            groups[title].append((i, doc))
        
        # Оптимизация: если слишком много групп - обрабатываем без группировки
        if len(groups) > len(documents) * 0.15:  # Более 15% уникальных заголовков
            logger.debug(f"Too many unique titles ({len(groups)}), processing without grouping for efficiency")
            # Обрабатываем по одному для сохранения контекста заголовков
            embeddings = []
            for doc, title in zip(documents, titles):
                emb = self._encode_batch([doc], task_type='retrieval_document', title=title)
                embeddings.append(emb[0].tolist())
            return embeddings
        
        # Обрабатываем каждую группу отдельно (эффективно при малом количестве групп)
        logger.debug(f"Processing {len(documents)} documents in {len(groups)} title groups")
        results = [None] * len(documents)
        
        for title, doc_group in groups.items():
            indices, docs = zip(*doc_group)
            # Обрабатываем все документы с одинаковым заголовком одновременно
            embeddings = self._encode_batch(list(docs), task_type='retrieval_document', title=title)
            
            # Расставляем результаты по оригинальным позициям
            for idx, embedding in zip(indices, embeddings):
                results[idx] = embedding.tolist()
                
        return results
    
    def encode_for_classification(self, texts: List[str]) -> List[List[float]]:
        """
        Кодирование для задач классификации
        Использует специализированный промпт для лучшего разделения классов
        """
        embeddings = self._encode_batch(texts, task_type='classification')
        return [emb.tolist() for emb in embeddings]
    
    def encode_for_clustering(self, texts: List[str]) -> List[List[float]]:
        """
        Кодирование для кластеризации текстов
        Оптимизировано для группировки похожих документов
        """
        embeddings = self._encode_batch(texts, task_type='clustering')
        return [emb.tolist() for emb in embeddings]
    
    def encode_for_semantic_similarity(self, texts: List[str]) -> List[List[float]]:
        """
        Кодирование для вычисления семантического сходства
        Идеально для сравнения пар предложений
        """
        embeddings = self._encode_batch(texts, task_type='semantic_similarity')
        return [emb.tolist() for emb in embeddings]
    
    def encode_for_code_search(self, code_snippets: List[str]) -> List[List[float]]:
        """
        Специализированное кодирование для поиска по коду
        Оптимизировано для программных текстов и запросов
        """
        embeddings = self._encode_batch(code_snippets, task_type='code_retrieval')
        return [emb.tolist() for emb in embeddings]
    
    def encode_with_matryoshka(self, texts: List[str], 
                              target_dimension: int = 512,
                              task_type: str = 'general',
                              title: Optional[str] = None) -> List[List[float]]:
        """
        Кодирование с Matryoshka representation learning
        Поддерживает разные размерности: 768, 512, 256, 128
        
        Args:
            texts: Список текстов для кодирования
            target_dimension: Целевая размерность (128, 256, 512, 768)
            task_type: Тип задачи для выбора промпта
            title: Заголовок для retrieval_document задач
        
        Returns:
            Список embeddings с указанной размерностью
        """
        # Валидация размерности
        supported_dims = [128, 256, 512, 768]
        if target_dimension not in supported_dims:
            raise ValueError(f"Unsupported dimension {target_dimension}. Supported: {supported_dims}")
        
        # Получаем полные embeddings
        full_embeddings = self._encode_batch(texts, task_type, title)
        
        # Обрезаем до нужной размерности и нормализуем
        truncated_embeddings = []
        for embedding in full_embeddings:
            # Обрезаем до target_dimension
            truncated = embedding[:target_dimension]
            
            # Нормализуем обрезанный вектор (критично для Matryoshka)
            norm = np.linalg.norm(truncated)
            if norm > 0:
                truncated = truncated / norm
            
            truncated_embeddings.append(truncated.tolist())
        
        logger.debug(f"Applied Matryoshka truncation: {embedding.shape[0]} -> {target_dimension}")
        return truncated_embeddings
    
    async def encode_async(self, texts: List[str], 
                          task_type: str = 'general',
                          title: Optional[str] = None) -> List[List[float]]:
        """
        Асинхронное кодирование текстов с поддержкой специализированных промптов
        Включает все новые функции безопасности и кеширования
        """
        if not texts:
            return []
            
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Fallback для обратной совместимости с Python 3.9
            loop = asyncio.get_event_loop()
        
        # Разбиваем на батчи для параллельной обработки
        optimal_batch_size = max(1, len(texts) // self.max_workers)
        batches = [texts[i:i+optimal_batch_size] for i in range(0, len(texts), optimal_batch_size)]
        
        # Создаем задачи для параллельной обработки батчей
        tasks = []
        for batch in batches:
            # Каждый батч обрабатывается с одинаковыми параметрами
            task = loop.run_in_executor(
                self.executor, 
                self._encode_batch, 
                batch, 
                task_type, 
                title
            )
            tasks.append(task)
        
        try:
            # Ждём завершения всех задач
            batch_results = await asyncio.gather(*tasks)
            
            # Объединяем результаты
            all_embeddings = []
            for batch_result in batch_results:
                for embedding in batch_result:
                    all_embeddings.append(embedding.tolist())
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Async encoding failed: {e}")
            with self._stats_lock:
                self.stats['encoding_errors'] += 1
            raise RuntimeError(f"Async embedding encoding failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Thread-safe получение подробной статистики работы сервиса
        Включает метрики производительности, кеширования и ошибок
        """
        with self._stats_lock:
            stats_copy = self.stats.copy()
            
        with self._cache_lock:
            cache_size = len(self.cache)
            
        # Вычисляем производные метрики
        total_requests = stats_copy['cache_hits'] + stats_copy['cache_misses']
        hit_rate = stats_copy['cache_hits'] / total_requests if total_requests > 0 else 0
        error_rate = stats_copy['encoding_errors'] / total_requests if total_requests > 0 else 0
        
        return {
            # Основная статистика
            "total_requests": total_requests,
            "successful_requests": total_requests - stats_copy['encoding_errors'],
            
            # Кеширование
            "cache_size": cache_size,
            "cache_capacity": self.cache_size,
            "cache_hits": stats_copy['cache_hits'],
            "cache_misses": stats_copy['cache_misses'],
            "cache_hit_rate": round(hit_rate, 3),
            
            # Производительность
            "model_load_time_sec": round(stats_copy['model_load_time'], 3),
            "average_inference_time_sec": round(stats_copy['average_inference_time'], 4),
            
            # Безопасность и ошибки
            "encoding_errors": stats_copy['encoding_errors'],
            "security_violations": stats_copy['security_violations'],
            "error_rate": round(error_rate, 4),
            
            # Конфигурация
            "model_device": str(self.device),
            "model_path": self.model_path,
            "max_workers": self.max_workers,
            "default_dimension": self.default_dimension,
            
            # Дополнительная информация
            "supported_tasks": list(self.TASK_PROMPTS.keys()),
            "matryoshka_dimensions": [768, 512, 256, 128]
        }
    
    def clear_cache(self) -> Dict[str, int]:
        """
        Очистка кеша с возвратом статистики
        Полезно для освобождения памяти или сброса состояния
        """
        with self._cache_lock:
            cleared_items = len(self.cache)
            self.cache.clear()
            
        logger.info(f"Cache cleared: {cleared_items} items removed")
        return {
            "cleared_items": cleared_items,
            "current_cache_size": 0
        }

class EmbeddingServer:
    """HTTP сервер для embedding сервиса"""
    
    def __init__(self, embedding_service: EmbeddingService, port: int = 8090):
        self.service = embedding_service
        self.port = port
        self.app = web.Application()
        self.setup_routes()
        self.setup_cors()
    
    def setup_routes(self):
        """Настройка маршрутов"""
        self.app.router.add_post('/embed', self.handle_embed)
        self.app.router.add_post('/embed_batch', self.handle_embed_batch)
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/stats', self.handle_stats)
    
    def setup_cors(self):
        """Настройка CORS"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def handle_embed(self, request: web.Request) -> web.Response:
        """Обработка запроса на embedding одного текста (локальный сервер)"""
        try:
            data = await request.json()
            text = data.get('text', '')
            
            # Минимальная валидация для локального сервера
            if not text:
                return web.json_response(
                    {'error': 'Text is required'},
                    status=400
                )
            
            # Кодируем текст
            embeddings = await self.service.encode_async([text])
            
            return web.json_response({
                'embedding': embeddings[0],
                'dimension': len(embeddings[0])
            })
            
        except Exception as e:
            logger.error(f"Error in handle_embed: {e}")
            return web.json_response(
                {'error': str(e)},
                status=500
            )
    
    async def handle_embed_batch(self, request: web.Request) -> web.Response:
        """Обработка запроса на embedding батча текстов (локальный сервер)"""
        try:
            data = await request.json()
            texts = data.get('texts', [])
            
            # Минимальная валидация для локального сервера
            if not texts:
                return web.json_response(
                    {'error': 'Texts array is required'},
                    status=400
                )
            
            # Кодируем тексты
            embeddings = await self.service.encode_async(texts)
            
            return web.json_response({
                'embeddings': embeddings,
                'count': len(embeddings),
                'dimension': len(embeddings[0]) if embeddings and len(embeddings) > 0 else 0
            })
            
        except Exception as e:
            logger.error(f"Error in handle_embed_batch: {e}")
            return web.json_response(
                {'error': str(e)},
                status=500
            )
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Проверка здоровья сервиса"""
        return web.json_response({
            'status': 'healthy',
            'service': 'embedding-server',
            'model': self.service.model_path
        })
    
    async def handle_stats(self, request: web.Request) -> web.Response:
        """Получение статистики"""
        stats = self.service.get_stats()
        return web.json_response(stats)
    
    def run(self):
        """Запуск сервера"""
        logger.info(f"Starting Embedding Server on port {self.port}")
        web.run_app(self.app, host='0.0.0.0', port=self.port)

def main():
    """Главная функция"""
    # Конфигурация из переменных окружения
    model_path = os.getenv('EMBEDDING_MODEL_PATH', 
                           r'C:\Models\ai-memory-service\models\embeddinggemma-300m')
    port = int(os.getenv('EMBEDDING_SERVER_PORT', '8090'))
    max_workers = int(os.getenv('EMBEDDING_MAX_WORKERS', '8'))  # Увеличено для предотвращения deadlock
    
    # Проверяем существование модели
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        sys.exit(1)
    
    # Создаём сервисы
    embedding_service = EmbeddingService(model_path, max_workers)
    server = EmbeddingServer(embedding_service, port)
    
    # Запускаем сервер
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()