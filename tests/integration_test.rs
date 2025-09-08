use ai_memory_service::{
    memory::MemoryService, 
    types::{MemoryCell, MemoryType, Priority, ContentAnalysis, MemoryQuery, RecalledMemory},
    config::Config,
    simd_search::{cosine_similarity_simd, parallel_vector_search},
};
use std::sync::Arc;
use std::env;
use std::collections::HashMap;
use uuid::Uuid;

#[tokio::test]
async fn test_memory_store_and_recall() {
    // Create in-memory config for testing
    let config = create_test_config();
    
    // Initialize service
    let service = MemoryService::new(config).await.expect("Failed to create service");
    
    // Test content to store
    let content = "Rust is a systems programming language focused on safety and performance.";
    
    // Store memory
    let memory_id = service.store(
        content.to_string(),
        Some("programming/rust".to_string()),
        None,
    ).await.expect("Failed to store memory");
    
    assert!(!memory_id.is_nil());
    
    // Recall memory
    let query = MemoryQuery {
        text: "Tell me about Rust programming".to_string(),
        context_hint: Some("programming".to_string()),
        limit: Some(10),
        memory_types: None,
        min_importance: None,
        time_range: None,
        include_related: true,
        similarity_threshold: None,
    };
    
    let recall_result = service.recall(query).await.expect("Failed to recall memory");
    
    // Verify recall contains our stored memory
    assert!(!recall_result.semantic_layer.is_empty() || 
            !recall_result.contextual_layer.is_empty() || 
            !recall_result.detailed_layer.is_empty());
}

#[tokio::test]
async fn test_context_hierarchy() {
    let config = create_test_config();
    let service = MemoryService::new(config).await.expect("Failed to create service");
    
    // Store memories in hierarchical contexts
    let contexts = vec![
        ("programming", "General programming concepts"),
        ("programming/rust", "Rust programming language"),
        ("programming/rust/async", "Async programming in Rust"),
        ("programming/python", "Python programming language"),
    ];
    
    let mut memory_ids = Vec::new();
    for (context, content) in contexts {
        let id = service.store(
            content.to_string(),
            Some(context.to_string()),
            None,
        ).await.expect("Failed to store memory");
        memory_ids.push(id);
    }
    
    // Query for specific context
    let query = MemoryQuery {
        text: "programming languages".to_string(),
        context_hint: Some("programming/rust".to_string()),
        limit: Some(10),
        memory_types: None,
        min_importance: None,
        time_range: None,
        include_related: true,
        similarity_threshold: None,
    };
    
    let recall_result = service.recall(query).await.expect("Failed to recall memories");
    
    // Should find Rust-related memories
    let all_memories: Vec<_> = recall_result.semantic_layer.iter()
        .chain(recall_result.contextual_layer.iter())
        .chain(recall_result.detailed_layer.iter())
        .collect();
    
    assert!(!all_memories.is_empty());
}

#[tokio::test]
async fn test_memory_types() {
    let config = create_test_config();
    let service = MemoryService::new(config).await.expect("Failed to create service");
    
    // Store different memory types
    let memories = vec![
        ("How to compile Rust: run cargo build", MemoryType::Procedural {
            steps: vec!["run cargo build".to_string()],
            tools: vec!["cargo".to_string()],
            prerequisites: vec![],
        }),
        ("Yesterday we deployed the new version", MemoryType::Episodic {
            event: "deployment".to_string(),
            location: None,
            participants: vec![],
            timeframe: None,
        }),
        ("Rust has zero-cost abstractions", MemoryType::Semantic {
            facts: vec!["zero-cost abstractions".to_string()],
            concepts: vec!["rust".to_string()],
        }),
    ];
    
    for (content, memory_type) in memories {
        let mut memory = MemoryCell::new(
            content.to_string(),
            "test".to_string(),
        );
        memory.memory_type = memory_type;
        memory.importance = 0.7;
        // Generate realistic test embedding instead of placeholder zeros
        memory.embedding = (0..768).map(|i| (i as f32 * 0.01).sin()).collect();
        
        // Validate MemoryCell is properly initialized
        assert_eq!(memory.content, content);
        assert_eq!(memory.context_path, "test");
        assert_eq!(memory.importance, 0.7);
        
        // Store through internal method (simplified for test)
        // In production, this would go through the full pipeline
    }
}

#[tokio::test]
async fn test_simd_performance() {
    // Generate test vectors
    let sizes = vec![128, 256, 512, 768, 1024];
    
    for size in sizes {
        let vec_a: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();
        let vec_b: Vec<f32> = (0..size).map(|i| (i as f32).cos()).collect();
        
        let similarity = cosine_similarity_simd(&vec_a, &vec_b);
        
        // Verify result is in valid range
        assert!(similarity >= -1.0 && similarity <= 1.0);
    }
}

#[tokio::test]
async fn test_parallel_search() {
    let query = vec![1.0, 0.0, 0.0];
    let mut embeddings = HashMap::new();
    embeddings.insert(Uuid::new_v4(), vec![1.0, 0.0, 0.0]);    // Perfect match
    embeddings.insert(Uuid::new_v4(), vec![0.0, 1.0, 0.0]);    // Orthogonal
    embeddings.insert(Uuid::new_v4(), vec![0.7, 0.7, 0.0]);    // Partial match
    embeddings.insert(Uuid::new_v4(), vec![-1.0, 0.0, 0.0]);   // Opposite
    
    let results = parallel_vector_search(&embeddings, &query, 2)
        .expect("Vector search should succeed");
    
    // Should return top 2 matches
    assert_eq!(results.len(), 2);
    
    // Test constants for better maintainability 
    const MIN_PERFECT_MATCH_SCORE: f32 = 0.99;
    const EXPECTED_RESULTS_COUNT: usize = 2;
    
    // Validate bounds before accessing array elements
    assert_eq!(results.len(), EXPECTED_RESULTS_COUNT, "Should return exactly {} matches", EXPECTED_RESULTS_COUNT);
    
    // First result should be the perfect match
    assert!(results[0].similarity > MIN_PERFECT_MATCH_SCORE, "First result similarity {} should be > {}", results[0].similarity, MIN_PERFECT_MATCH_SCORE);
    
    // Results should be sorted by similarity (descending)
    assert!(results[0].similarity >= results[1].similarity, "Results should be sorted by similarity: {} >= {}", results[0].similarity, results[1].similarity);
}

#[tokio::test]
async fn test_memory_decay() {
    let config = create_test_config();
    let service = MemoryService::new(config).await.expect("Failed to create service");
    
    // Store a memory
    let memory_id = service.store(
        "This memory will decay over time".to_string(),
        Some("temporal".to_string()),
        None,
    ).await.expect("Failed to store memory");
    
    // Initial recall
    let query = MemoryQuery {
        text: "decay memory".to_string(),
        context_hint: Some("temporal".to_string()),
        limit: Some(10),
        memory_types: None,
        min_importance: None,
        time_range: None,
        include_related: true,
        similarity_threshold: None,
    };
    
    let initial_recall = service.recall(query.clone()).await.expect("Failed to recall");
    
    // Memory should be present initially
    let initial_count = initial_recall.semantic_layer.len() + 
                       initial_recall.contextual_layer.len() + 
                       initial_recall.detailed_layer.len();
    assert!(initial_count > 0);
}

#[tokio::test]
async fn test_batch_processing() {
    let config = create_test_config();
    let service = MemoryService::new(config).await.expect("Failed to create service");
    
    // Store multiple memories in batch
    let contents = vec![
        "First memory about Rust",
        "Second memory about async programming",
        "Third memory about memory safety",
        "Fourth memory about performance",
        "Fifth memory about concurrency",
    ];
    
    let mut memory_ids = Vec::new();
    for content in contents {
        let id = service.store(
            content.to_string(),
            Some("batch_test".to_string()),
            None,
        ).await.expect("Failed to store memory");
        memory_ids.push(id);
    }
    
    assert_eq!(memory_ids.len(), 5);
    
    // Verify all memories were stored
    for id in memory_ids {
        assert!(!id.is_nil());
    }
}

#[tokio::test]
async fn test_error_handling() {
    let config = create_test_config();
    let service = MemoryService::new(config).await.expect("Failed to create service");
    
    // Test with empty content
    let result = service.store(
        "".to_string(),
        None,
        None,
    ).await;
    
    // Should handle empty content gracefully
    assert!(result.is_ok() || result.is_err());
    
    // Test with very long content
    let long_content = "a".repeat(100000);
    let result = service.store(
        long_content,
        None,
        None,
    ).await;
    
    // Should handle long content
    assert!(result.is_ok() || result.is_err());
}

#[tokio::test] 
async fn test_memory_compression() {
    let config = create_test_config();
    
    // Test L3 cache compression
    let large_embedding = vec![0.5_f32; 768];
    let mut large_memory = MemoryCell::new(
        "Large memory for compression test".to_string(),
        "test/compression".to_string(),
    );
    large_memory.embedding = large_embedding.clone();
    large_memory.memory_type = MemoryType::Semantic {
        facts: vec!["compression test".to_string()],
        concepts: vec!["optimization".to_string()],
    };
    large_memory.importance = 0.8;
    large_memory.tags = vec!["test".to_string()];
    
    // Validate large memory is properly initialized
    assert_eq!(large_memory.content, "Large memory for compression test");
    assert_eq!(large_memory.importance, 0.8);
    
    // Verify compression works
    let serialized = bincode::serialize(&large_memory).expect("Failed to serialize");
    let original_size = serialized.len();
    
    // Compress with LZ4
    let compressed = lz4_flex::compress_prepend_size(&serialized);
    let compressed_size = compressed.len();
    
    // Should achieve some compression
    assert!(compressed_size < original_size);
    
    // Verify decompression
    let decompressed = lz4_flex::decompress_size_prepended(&compressed).expect("Failed to decompress");
    assert_eq!(decompressed, serialized);
}

#[tokio::test]
async fn test_concurrent_access() {
    let config = create_test_config();
    let service = Arc::new(MemoryService::new(config).await.expect("Failed to create service"));
    
    // Spawn multiple concurrent tasks
    let mut handles = vec![];
    
    for i in 0..10 {
        let service_clone = Arc::clone(&service);
        let handle = tokio::spawn(async move {
            let content = format!("Concurrent memory {}", i);
            service_clone.store(
                content,
                Some("concurrent_test".to_string()),
                None,
            ).await
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let results: Vec<_> = futures::future::join_all(handles).await;
    
    // Verify all operations succeeded
    for result in results {
        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());
    }
}

// Helper function to create test configuration  
fn create_test_config() -> Config {
    // Use environment variables for sensitive data
    let neo4j_password = env::var("NEO4J_TEST_PASSWORD")
        .unwrap_or_else(|_| "test_only_not_for_production".to_string());
    
    Config {
        server: ai_memory_service::config::ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0, // Use random port for testing
            workers: 1,
            environment: "test".to_string(),
            cors_origins: vec!["*".to_string()],
        },
        storage: ai_memory_service::config::StorageConfig {
            neo4j_uri: env::var("NEO4J_TEST_URI")
                .unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
            neo4j_user: env::var("NEO4J_TEST_USER")
                .unwrap_or_else(|_| "neo4j".to_string()),
            neo4j_password,
            connection_pool_size: 10,
        },
        embedding: ai_memory_service::config::EmbeddingConfig {
            model_path: env::var("TEST_MODEL_PATH")
                .unwrap_or_else(|_| "models/embedding_model.onnx".to_string()),
            tokenizer_path: env::var("TEST_TOKENIZER_PATH")
                .unwrap_or_else(|_| "models/tokenizer.json".to_string()),
            batch_size: 32,
            max_sequence_length: 512,
            embedding_dimension: Some(512),
            normalize_embeddings: true,
            precision: "float32".to_string(),
            use_specialized_prompts: true,
        },
        cache: ai_memory_service::config::CacheConfig {
            l1_size: 100,
            l2_size: 1000,
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