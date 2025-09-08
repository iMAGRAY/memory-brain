//! Performance benchmarks for AI Memory Service
//!
//! Comprehensive benchmarks for testing memory operations, SIMD performance,
//! and overall system throughput on Windows systems.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use ai_memory_service::{
    MemoryService, MemoryConfig, MemoryCell, MemoryType, Priority,
    simd_search::{cosine_similarity_simd, parallel_vector_search_tuples},
    cache::CacheSystem,
    embedding::EmbeddingService,
    storage::GraphStorage,
};
use std::time::Duration;
use uuid::Uuid;
use tokio::runtime::Runtime;

/// Create test memory service
fn create_test_service() -> MemoryService {
    let rt = Runtime::new().unwrap();
    let config = MemoryConfig::default();
    
    rt.block_on(async {
        MemoryService::new(config).await.expect("Failed to create memory service")
    })
}

/// Create test embeddings of different sizes
fn create_test_embeddings(size: usize, count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            (0..size)
                .map(|j| ((i * size + j) as f32).sin())
                .collect()
        })
        .collect()
}

/// Create test memory cell
fn create_test_memory(content: &str, importance: f32) -> MemoryCell {
    MemoryCell {
        id: Uuid::new_v4(),
        content: content.to_string(),
        embedding: vec![0.5; 768],
        memory_type: MemoryType::Semantic {
            facts: vec!["test fact".to_string()],
            concepts: vec!["test".to_string()],
        },
        importance,
        context_path: "/test".to_string(),
        tags: vec!["benchmark".to_string()],
        metadata: None,
        associations: vec![],
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        last_accessed: chrono::Utc::now(),
        access_count: 0,
        decay_rate: 0.01,
    }
}

/// Benchmark SIMD operations with different vector sizes
fn bench_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");
    
    // Test different vector sizes
    for size in [128, 256, 512, 768, 1024, 2048].iter() {
        let vec_a: Vec<f32> = (0..*size).map(|i| (i as f32).sin()).collect();
        let vec_b: Vec<f32> = (0..*size).map(|i| (i as f32).cos()).collect();
        
        group.bench_with_input(
            BenchmarkId::new("cosine_similarity_simd", size),
            size,
            |b, _| {
                b.iter(|| cosine_similarity_simd(&vec_a, &vec_b));
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel vector search with different dataset sizes
fn bench_parallel_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_search");
    group.sample_size(50); // Reduce sample size for expensive operations
    
    let query = vec![1.0, 0.0, 0.5, 0.3, 0.8];
    
    // Test different dataset sizes
    for dataset_size in [100, 500, 1000, 5000, 10000].iter() {
        let embeddings: Vec<(Uuid, Vec<f32>)> = (0..*dataset_size)
            .map(|i| {
                let embedding: Vec<f32> = (0..768)
                    .map(|j| ((i * 768 + j) as f32 * 0.01).sin())
                    .collect();
                (Uuid::new_v4(), embedding)
            })
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("parallel_vector_search", dataset_size),
            dataset_size,
            |b, _| {
                b.iter(|| {
                    parallel_vector_search_tuples(&query, &embeddings, 10)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory store operations
fn bench_memory_store(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("memory_store");
    group.sample_size(30);
    
    // Create service once for all benchmarks
    let service = rt.block_on(async {
        let config = create_test_config();
        MemoryService::new(config).await.expect("Failed to create service")
    });
    
    // Test different content sizes
    for content_size in [100, 500, 1000, 5000, 10000].iter() {
        let content = "a".repeat(*content_size);
        
        group.bench_with_input(
            BenchmarkId::new("store_memory", content_size),
            content_size,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let result = service.store(
                        content.clone(),
                        Some("benchmark".to_string()),
                        None,
                    ).await;
                    // Ensure operation completes
                    result.expect("Store operation failed")
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory recall operations
fn bench_memory_recall(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("memory_recall");
    group.sample_size(20);
    
    let service = rt.block_on(async {
        let config = create_test_config();
        let service = MemoryService::new(config).await.expect("Failed to create service");
        
        // Pre-populate with test memories
        for i in 0..1000 {
            let content = format!("Test memory content number {}", i);
            service.store(content, Some("benchmark".to_string()), None)
                .await
                .expect("Failed to store test memory");
        }
        
        service
    });
    
    // Test different query complexities
    let queries = vec![
        "simple query",
        "more complex query with multiple terms and concepts",
        "very detailed and comprehensive query that should match multiple stored memories with various levels of similarity and relevance",
    ];
    
    for (i, query) in queries.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("recall_memory", i),
            query,
            |b, query| {
                b.to_async(&rt).iter(|| async {
                    let query_obj = ai_memory_service::RecallQuery {
                        query: query.to_string(),
                        context: Some("benchmark".to_string()),
                        limit: 10,
                        memory_types: None,
                        min_importance: None,
                        timeframe: None,
                    };
                    
                    service.recall(query_obj).await.expect("Recall failed")
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark cache operations
fn bench_cache_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_operations");
    
    let cache = CacheSystem::new_default();
    let test_memories: Vec<_> = (0..1000)
        .map(|i| {
            let memory = create_test_memory(&format!("Cache test {}", i), 0.5);
            (memory.id, memory)
        })
        .collect();
    
    group.bench_function("cache_put_l1", |b| {
        b.iter(|| {
            for (id, memory) in &test_memories {
                cache.put_l1(*id, memory.clone());
            }
        });
    });
    
    group.bench_function("cache_get_l1", |b| {
        // Pre-populate cache
        for (id, memory) in &test_memories {
            cache.put_l1(*id, memory.clone());
        }
        
        b.iter(|| {
            for (id, _) in &test_memories {
                let _ = cache.get_l1(id);
            }
        });
    });
    
    group.finish();
}

/// Benchmark embedding generation
fn bench_embedding_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("embedding_generation");
    group.sample_size(10); // Expensive operations
    
    let service = rt.block_on(async {
        let config = ai_memory_service::config::EmbeddingConfig {
            model_path: "./models/embeddinggemma-300m-ONNX/model.onnx".to_string(),
            tokenizer_path: "./models/embeddinggemma-300m-ONNX/tokenizer.json".to_string(),
            batch_size: 32,
            max_sequence_length: 512,
            use_gpu: false,
            cache_embeddings: true,
        };
        
        EmbeddingService::new(config).await.expect("Failed to create embedding service")
    });
    
    // Test different text lengths
    let texts = vec![
        "Short text",
        "Medium length text with several words and some complexity in the content structure",
        "Very long text content that includes multiple sentences, various concepts, and detailed explanations that would require more processing time and computational resources to generate embeddings for this comprehensive textual input".repeat(5),
    ];
    
    for (i, text) in texts.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("generate_embedding", i),
            text,
            |b, text| {
                b.to_async(&rt).iter(|| async {
                    service.embed(text).await.expect("Embedding failed")
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("concurrent_operations");
    group.sample_size(10);
    
    let service = std::sync::Arc::new(rt.block_on(async {
        let config = create_test_config();
        MemoryService::new(config).await.expect("Failed to create service")
    }));
    
    // Test different concurrency levels
    for concurrency in [1, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_stores", concurrency),
            concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| {
                    let service = std::sync::Arc::clone(&service);
                    async move {
                        let mut handles = vec![];
                        
                        for i in 0..concurrency {
                            let service = std::sync::Arc::clone(&service);
                            let handle = tokio::spawn(async move {
                                let content = format!("Concurrent test {}", i);
                                service.store(content, Some("concurrent".to_string()), None).await
                            });
                            handles.push(handle);
                        }
                        
                        futures::future::join_all(handles).await
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark system throughput
fn bench_system_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("system_throughput");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);
    
    let service = std::sync::Arc::new(rt.block_on(async {
        let config = create_test_config();
        MemoryService::new(config).await.expect("Failed to create service")
    }));
    
    group.bench_function("mixed_operations_throughput", |b| {
        b.to_async(&rt).iter(|| {
            let service = std::sync::Arc::clone(&service);
            async move {
                // Mix of store and recall operations
                let store_future = service.store(
                    "Throughput test content".to_string(),
                    Some("throughput".to_string()),
                    None,
                );
                
                let recall_future = service.recall(ai_memory_service::RecallQuery {
                    query: "test".to_string(),
                    context: Some("throughput".to_string()),
                    limit: 5,
                    memory_types: None,
                    min_importance: None,
                    timeframe: None,
                });
                
                let (_store_result, _recall_result) = tokio::join!(store_future, recall_future);
            }
        });
    });
    
    group.finish();
}

/// Helper function to create test configuration
fn create_test_config() -> MemoryConfig {
    MemoryConfig {
        server: ai_memory_service::config::ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            workers: 1,
            enable_tls: false,
            tls_cert: None,
            tls_key: None,
        },
        storage: ai_memory_service::config::StorageConfig {
            neo4j_uri: std::env::var("NEO4J_TEST_URI")
                .unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
            neo4j_user: std::env::var("NEO4J_TEST_USER")
                .unwrap_or_else(|_| "neo4j".to_string()),
            neo4j_password: std::env::var("NEO4J_TEST_PASSWORD")
                .unwrap_or_else(|_| "test_password".to_string()),
            neo4j_database: Some("benchmark_db".to_string()),
            connection_pool_size: 10,
            max_retry_attempts: 3,
            retry_delay_ms: 100,
        },
        embedding: ai_memory_service::config::EmbeddingConfig {
            model_path: std::env::var("TEST_MODEL_PATH")
                .unwrap_or_else(|_| "./models/embeddinggemma-300m-ONNX/model.onnx".to_string()),
            tokenizer_path: std::env::var("TEST_TOKENIZER_PATH")
                .unwrap_or_else(|_| "./models/embeddinggemma-300m-ONNX/tokenizer.json".to_string()),
            batch_size: 16, // Smaller batch size for benchmarks
            max_sequence_length: 256, // Shorter sequences for faster processing
            use_gpu: false,
            cache_embeddings: true,
        },
        cache: ai_memory_service::config::CacheConfig {
            l1_size: 100,
            l2_size: 500,
            ttl_seconds: 300,
            compression_enabled: true,
        },
        brain: ai_memory_service::config::BrainConfig {
            max_memories: 100000,
            importance_threshold: 0.5,
            consolidation_interval: 300,
            decay_rate: 0.01,
        },
    }
}

// Define benchmark groups
criterion_group!(
    benches,
    bench_simd_operations,
    bench_parallel_search,
    bench_memory_store,
    bench_memory_recall,
    bench_cache_operations,
    bench_embedding_generation,
    bench_concurrent_operations,
    bench_system_throughput
);

criterion_main!(benches);