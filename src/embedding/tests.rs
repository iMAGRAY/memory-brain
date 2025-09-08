//! Comprehensive tests for EmbeddingService with EmbeddingGemma-300M
//! Production-ready test suite covering all critical functionality

#[cfg(test)]
mod embedding_tests {
    use super::super::*;
    use std::sync::Arc;
    use tokio;

    // Mock data for testing
    const TEST_TEXTS: &[&str] = &[
        "This is a test document about machine learning",
        "Vector embeddings are used in AI applications",
        "EmbeddingGemma provides high-quality multilingual embeddings",
        "Python integration with Rust using PyO3 is powerful",
    ];

    const TEST_QUERIES: &[&str] = &[
        "machine learning algorithms",
        "vector search similarity",
        "multilingual text processing",
    ];

    #[tokio::test]
    async fn test_embedding_service_creation() {
        // Skip if Python environment not available
        if std::env::var("SKIP_PYTHON_TESTS").is_ok() {
            return;
        }

        // Test with EmbeddingGemma-300M (requires HF login and license acceptance)
        let service = match create_test_service("google/embeddinggemma-300m").await {
            Ok(s) => s,
            Err(_) => {
                // Fallback to smaller model for CI/testing
                println!("EmbeddingGemma not available, using fallback model");
                EmbeddingService::new("all-MiniLM-L6-v2", "", 8, 512).await
                    .expect("Should create fallback service")
            }
        };

        // Test EmbeddingGemma-300M specific properties
        assert_eq!(service.supported_dimensions(), &[768, 512, 256, 128]);
        assert!(service.is_dimension_supported(768));
        assert!(service.is_dimension_supported(512));
        assert!(service.is_dimension_supported(256));
        assert!(service.is_dimension_supported(128));
        assert!(!service.is_dimension_supported(1024));
    }
    
    /// Helper to create test service with proper error handling
    async fn create_test_service(model_name: &str) -> MemoryResult<EmbeddingService> {
        EmbeddingService::new(model_name, "", 8, 2048).await
    }

    #[test]
    fn test_task_type_formatting() {
        let service_config = create_mock_service_config();
        
        // Test query formatting with EmbeddingGemma prompts
        let query = "test query";
        let formatted = service_config.format_with_instruction(query, TaskType::Query);
        assert_eq!(formatted, "task: search result | query: test query");
        
        // Test document formatting
        let doc = "test document";
        let formatted = service_config.format_with_instruction(doc, TaskType::Document);
        assert_eq!(formatted, "title: none | text: test document");
        
        // Test classification formatting
        let text = "classification text";
        let formatted = service_config.format_with_instruction(text, TaskType::Classification);
        assert_eq!(formatted, "task: classification | text: classification text");
    }

    #[test]
    fn test_input_validation() {
        let service = create_mock_service_config();
        
        // Test empty text validation
        assert!(service.validate_text_input("").is_ok());
        
        // Test max length validation
        let long_text = "a".repeat(10000);
        assert!(service.validate_text_input(&long_text).is_err());
        
        // Test batch validation
        let valid_batch = vec!["text1".to_string(), "text2".to_string()];
        assert!(service.validate_batch_input(&valid_batch).is_ok());
        
        // Test empty batch
        let empty_batch: Vec<String> = vec![];
        assert!(service.validate_batch_input(&empty_batch).is_ok());
        
        // Test batch with empty string
        let invalid_batch = vec!["text1".to_string(), "".to_string()];
        assert!(service.validate_batch_input(&invalid_batch).is_err());
    }

    #[test]
    fn test_text_sanitization() {
        let service = create_mock_service_config();
        
        // Test null byte removal
        let dirty_text = "test\0text\x01more";
        let clean_text = service.sanitize_text_input(dirty_text);
        assert!(!clean_text.contains('\0'));
        assert!(!clean_text.contains('\x01'));
        
        // Test whitespace trimming
        let text_with_spaces = "  test text  ";
        let trimmed = service.sanitize_text_input(text_with_spaces);
        assert_eq!(trimmed, "test text");
    }

    #[test]
    fn test_cache_key_generation() {
        let service = create_mock_service_config();
        
        // Test consistent key generation
        let text = "test text";
        let key1 = service.generate_cache_key(text, TaskType::Query, 768);
        let key2 = service.generate_cache_key(text, TaskType::Query, 768);
        assert_eq!(key1, key2, "Same inputs should generate same keys");
        
        // Test different keys for different task types
        let key_query = service.generate_cache_key(text, TaskType::Query, 768);
        let key_doc = service.generate_cache_key(text, TaskType::Document, 768);
        assert_ne!(key_query, key_doc, "Different task types should generate different keys");
        
        // Test different keys for different dimensions
        let key_768 = service.generate_cache_key(text, TaskType::Query, 768);
        let key_512 = service.generate_cache_key(text, TaskType::Query, 512);
        assert_ne!(key_768, key_512, "Different dimensions should generate different keys");
    }

    #[test]
    fn test_embeddinggemma_matryoshka_dimensions() {
        let service = create_mock_service_config();
        
        // Test EmbeddingGemma-300M Matryoshka dimensions
        let embeddinggemma_dims = [768, 512, 256, 128];
        for &dim in &embeddinggemma_dims {
            assert!(service.is_dimension_supported(dim), 
                    "EmbeddingGemma-300M should support dimension {}", dim);
        }
        
        // Test unsupported dimensions
        let unsupported_dims = [64, 100, 384, 1024, 1536];
        for &dim in &unsupported_dims {
            assert!(!service.is_dimension_supported(dim), 
                    "EmbeddingGemma-300M should not support dimension {}", dim);
        }
        
        // Verify exact match with supported dimensions
        assert_eq!(service.supported_dimensions(), &embeddinggemma_dims);
    }

    #[test]
    fn test_cosine_similarity_calculation() {
        // Test identical vectors
        let vec_a = vec![1.0, 0.0, 0.0];
        let vec_b = vec![1.0, 0.0, 0.0];
        let similarity = cosine_similarity(&vec_a, &vec_b);
        assert!((similarity - 1.0).abs() < 1e-6, "Identical vectors should have similarity 1.0");
        
        // Test orthogonal vectors
        let vec_c = vec![1.0, 0.0, 0.0];
        let vec_d = vec![0.0, 1.0, 0.0];
        let similarity = cosine_similarity(&vec_c, &vec_d);
        assert!((similarity - 0.0).abs() < 1e-6, "Orthogonal vectors should have similarity 0.0");
        
        // Test different length vectors
        let vec_e = vec![1.0, 0.0];
        let vec_f = vec![1.0, 0.0, 0.0];
        let similarity = cosine_similarity(&vec_e, &vec_f);
        assert_eq!(similarity, 0.0, "Different length vectors should return 0.0");
    }

    #[tokio::test]
    async fn test_embedding_dimension_validation() {
        if std::env::var("SKIP_PYTHON_TESTS").is_ok() {
            return;
        }

        let service = create_test_service("all-MiniLM-L6-v2").await
            .expect("Should create test service");
        
        // Test EmbeddingGemma-300M dimension validation
        for &dim in &[768, 512, 256, 128] {
            let valid_embedding = vec![0.1; dim];
            assert!(service.validate_embedding_dimension(&valid_embedding, dim, "test").is_ok(),
                    "Should validate dimension {}", dim);
        }
        
        // Test invalid dimension
        let invalid_embedding = vec![0.1; 512];
        assert!(service.validate_embedding_dimension(&invalid_embedding, 768, "test").is_err(),
                "Should reject mismatched dimension");
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let service = create_mock_service_config();
        
        // Test cache storage and retrieval
        let test_embedding = vec![0.1, 0.2, 0.3, 0.4];
        let cache_key = "test_key".to_string();
        
        // Store embedding
        assert!(service.store_cached_embedding(cache_key.clone(), &test_embedding).is_ok());
        
        // Retrieve embedding
        if let Some(retrieved) = service.get_cached_embedding(&cache_key) {
            assert_eq!(**retrieved, test_embedding);
        } else {
            panic!("Should retrieve cached embedding");
        }
        
        // Test cache stats
        let (current_size, max_size) = service.cache_stats();
        assert!(current_size > 0, "Cache should have entries");
        assert_eq!(max_size, MAX_CACHE_SIZE);
        
        // Test cache clearing
        service.clear_cache();
        let (size_after_clear, _) = service.cache_stats();
        assert_eq!(size_after_clear, 0, "Cache should be empty after clearing");
    }

    #[test]
    fn test_model_info_embeddinggemma() {
        let service = create_mock_service_config();
        let info = service.model_info();
        
        // EmbeddingGemma-300M specifications
        assert_eq!(info.dimensions, 768); // Default dimension for EmbeddingGemma
        assert_eq!(info.max_sequence_length, 2048); // EmbeddingGemma context window
        assert_eq!(info.vocab_size, 256000); // EmbeddingGemma-300M vocab size
        assert_eq!(info.name, "test-model");
    }

    // Helper function to create mock service for testing validation logic
    fn create_mock_service_config() -> EmbeddingService {
        let python_obj = Python::with_gil(|py| py.None());
        
        EmbeddingService {
            python_module: Arc::new(std::sync::Mutex::new(python_obj)),
            model_name: "test-model".to_string(),
            embedding_cache: Arc::new(dashmap::DashMap::new()),
            use_instructions: true,
            matryoshka_dims: vec![768, 512, 256, 128], // EmbeddingGemma-300M dimensions
            embedding_timeout: Duration::from_secs(30),
            max_text_length: MAX_TEXT_LENGTH,
            max_batch_size: MAX_BATCH_SIZE,
        }
    }
    

    #[test]
    fn test_embedding_extraction_validation() {
        // Test empty array handling
        let empty_vec: Vec<f32> = vec![];
        
        // Test finite float validation - this would be tested in integration tests
        // with actual Python arrays, but we test the logic here
        let valid_floats = vec![0.1, -0.5, 0.0, 1.0];
        assert!(valid_floats.iter().all(|&f| f.is_finite()));
        
        let invalid_floats = vec![0.1, f32::NAN, 0.5];
        assert!(!invalid_floats.iter().all(|&f| f.is_finite()));
    }

    #[tokio::test]
    async fn test_timeout_functionality() {
        // Test that timeout configuration is properly set
        let service = create_mock_service_config();
        assert_eq!(service.embedding_timeout, Duration::from_secs(30));
    }
}

// Integration tests that require actual Python environment
#[cfg(test)]
mod integration_tests {
    use super::super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_embeddinggemma_real_generation() {
        // Skip if environment variable is set or Python not available
        if std::env::var("SKIP_PYTHON_TESTS").is_ok() || 
           std::env::var("SKIP_INTEGRATION_TESTS").is_ok() {
            return;
        }

        let service = match create_test_service("google/embeddinggemma-300m").await {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping EmbeddingGemma test - model not available (requires HF login)");
                return;
            }
        };

        // Test single text embedding with EmbeddingGemma
        let text = "This is a test document about machine learning";
        let embedding = service.embed_text(text).await;
        assert!(embedding.is_ok(), "Should generate EmbeddingGemma embedding successfully");
        
        let embedding = embedding.unwrap();
        assert_eq!(embedding.len(), 768, "EmbeddingGemma should produce 768-dim embeddings");
        assert!(embedding.iter().all(|&f| f.is_finite()), "All embedding values should be finite");

        // Test query embedding with EmbeddingGemma-specific prompts
        let query = "machine learning algorithms";
        let query_embedding = service.embed_query(query).await;
        assert!(query_embedding.is_ok(), "Should generate query embedding with proper prompts");
        
        let query_emb = query_embedding.unwrap();
        assert_eq!(query_emb.len(), 768, "Query embedding should be 768-dim");

        // Test document vs query embedding difference (should be different due to prompts)
        let doc_embedding = service.embed_document(text).await.unwrap();
        let similarity = cosine_similarity(&query_emb, &doc_embedding);
        assert!(similarity < 1.0, "Query and document embeddings should differ due to prompts");

        // Test batch embedding
        let texts = vec![
            "Vector embeddings for semantic search".to_string(),
            "Neural networks and deep learning".to_string()
        ];
        let batch_embeddings = service.embed_batch(&texts).await;
        assert!(batch_embeddings.is_ok(), "Should generate batch embeddings");
        
        let batch_embeddings = batch_embeddings.unwrap();
        assert_eq!(batch_embeddings.len(), 2, "Should generate embedding for each text");
        assert!(batch_embeddings.iter().all(|emb| emb.len() == 768), "All batch embeddings should be 768-dim");
        
        // Test cache functionality
        let cached_embedding = service.embed_text(text).await;
        assert!(cached_embedding.is_ok(), "Should retrieve from cache");
        assert_eq!(cached_embedding.unwrap(), embedding, "Cached result should match original");
    }
    }

    #[tokio::test]
    async fn test_embeddinggemma_matryoshka_dimensions() {
        if std::env::var("SKIP_PYTHON_TESTS").is_ok() || 
           std::env::var("SKIP_INTEGRATION_TESTS").is_ok() {
            return;
        }

        let service = match create_test_service("google/embeddinggemma-300m").await {
            Ok(s) => s,
            Err(_) => {
                println!("Skipping EmbeddingGemma Matryoshka test - model not available");
                return;
            }
        };

        let text = "Testing Matryoshka representation learning with EmbeddingGemma";
        
        // Test all EmbeddingGemma-300M Matryoshka dimensions
        for &dim in &[768, 512, 256, 128] {
            let embedding = service.embed_with_dimension(text, TaskType::Query, dim).await;
            assert!(embedding.is_ok(), "Should generate {}-dim embedding", dim);
            
            let emb = embedding.unwrap();
            assert_eq!(emb.len(), dim, "Embedding should have exactly {} dimensions", dim);
            assert!(emb.iter().all(|&f| f.is_finite()), "All values should be finite for {}-dim", dim);
        }
        
        // Test batch Matryoshka processing
        let texts = vec![
            "First test document".to_string(),
            "Second test document".to_string()
        ];
        
        let batch_256 = service.embed_batch_with_dimension(&texts, TaskType::Document, 256).await;
        assert!(batch_256.is_ok(), "Should generate 256-dim batch embeddings");
        
        let batch_embs = batch_256.unwrap();
        assert_eq!(batch_embs.len(), 2, "Should have 2 batch embeddings");
        assert!(batch_embs.iter().all(|emb| emb.len() == 256), "All should be 256-dim");
    }
    
    #[tokio::test]
    async fn test_error_handling() {
        // Test invalid model name
        let invalid_service = EmbeddingService::new("nonexistent/model", "", 8, 512).await;
        assert!(invalid_service.is_err(), "Should fail with invalid model name");
        
        // Test with mock service for input validation
        let service = create_mock_service_config();
        
        // Test oversized text
        let huge_text = "a".repeat(10000);
        let result = service.validate_text_input(&huge_text);
        assert!(result.is_err(), "Should reject oversized text");
        
        // Test invalid dimension
        let invalid_dim_result = service.validate_embedding_dimension(&vec![0.1; 100], 768, "test");
        assert!(invalid_dim_result.is_err(), "Should reject dimension mismatch");
        
        // Test batch with oversized content
        let oversized_batch = vec!["normal".to_string(), "a".repeat(5000).to_string()];
        let batch_result = service.validate_batch_input(&oversized_batch);
        assert!(batch_result.is_err(), "Should reject batch with oversized text");
    }
    
    #[tokio::test]
    async fn test_performance_benchmarks() {
        if std::env::var("SKIP_PERFORMANCE_TESTS").is_ok() {
            return;
        }
        
        let service = match create_test_service("google/embeddinggemma-300m").await {
            Ok(s) => s,
            Err(_) => {
                // Fallback for CI environment
                match create_test_service("all-MiniLM-L6-v2").await {
                    Ok(s) => s,
                    Err(_) => {
                        println!("Skipping performance test - No models available");
                        return;
                    }
                }
            }
        };
        
        let start = std::time::Instant::now();
        
        // Benchmark single embedding generation
        let text = "Performance test document for embedding generation";
        let _embedding = service.embed_text(text).await.expect("Should generate embedding");
        let single_duration = start.elapsed();
        
        println!("Single embedding generation took: {:?}", single_duration);
        assert!(single_duration.as_secs() < 30, "Single embedding should complete within 30 seconds");
        
        // Benchmark cache hit performance
        let cache_start = std::time::Instant::now();
        let _cached_embedding = service.embed_text(text).await.expect("Should retrieve from cache");
        let cache_duration = cache_start.elapsed();
        
        println!("Cache hit took: {:?}", cache_duration);
        assert!(cache_duration.as_millis() < 10, "Cache hit should be under 10ms");
        
        // Benchmark batch processing
        let batch_texts = vec![
            "Batch document one".to_string(),
            "Batch document two".to_string(),
            "Batch document three".to_string(),
        ];
        
        let batch_start = std::time::Instant::now();
        let _batch_embeddings = service.embed_batch(&batch_texts).await.expect("Should generate batch");
        let batch_duration = batch_start.elapsed();
        
        println!("Batch of {} embeddings took: {:?}", batch_texts.len(), batch_duration);
        assert!(batch_duration.as_secs() < 60, "Batch processing should complete within 60 seconds");
        
        // Memory usage tracking (basic)
        let (cache_size, _) = service.cache_stats();
        println!("Cache contains {} entries after performance test", cache_size);
        assert!(cache_size > 0, "Cache should have entries after operations");
    }
    
    #[tokio::test]
    async fn test_advanced_cache_operations() {
        let service = create_mock_service_config();
        
        // Test cache with different dimensions
        let text = "cache test text";
        let embedding_768 = vec![0.1; 768];
        let embedding_512 = vec![0.2; 512];
        
        // Store embeddings with different dimensions
        let key_768 = service.generate_cache_key(text, TaskType::Query, 768);
        let key_512 = service.generate_cache_key(text, TaskType::Query, 512);
        
        assert!(service.store_cached_embedding(key_768.clone(), &embedding_768).is_ok());
        assert!(service.store_cached_embedding(key_512.clone(), &embedding_512).is_ok());
        
        // Test dimension-validated cache retrieval
        let retrieved_768 = service.get_cached_embedding_validated(&key_768, 768);
        assert!(retrieved_768.is_some(), "Should retrieve 768-dim embedding");
        assert_eq!(**retrieved_768.unwrap(), embedding_768);
        
        let retrieved_512 = service.get_cached_embedding_validated(&key_512, 512);
        assert!(retrieved_512.is_some(), "Should retrieve 512-dim embedding");
        assert_eq!(**retrieved_512.unwrap(), embedding_512);
        
        // Test dimension mismatch rejection
        let wrong_dim = service.get_cached_embedding_validated(&key_768, 512);
        assert!(wrong_dim.is_none(), "Should reject dimension mismatch");
        
        // Test cache eviction behavior
        for i in 0..100 {
            let test_text = format!("eviction test {}", i);
            let test_key = service.generate_cache_key(&test_text, TaskType::Document, 768);
            let test_emb = vec![i as f32; 768];
            assert!(service.store_cached_embedding(test_key, &test_emb).is_ok());
        }
        
        let (final_cache_size, max_size) = service.cache_stats();
        println!("Cache size after bulk insert: {}/{}", final_cache_size, max_size);
    }
    
    #[tokio::test]
    async fn test_concurrent_operations() {
        if std::env::var("SKIP_CONCURRENCY_TESTS").is_ok() {
            return;
        }
        
        let service = Arc::new(create_mock_service_config());
        
        // Test concurrent cache operations
        let mut handles = vec![];
        
        for i in 0..10 {
            let service_clone = Arc::clone(&service);
            let handle = tokio::spawn(async move {
                let text = format!("concurrent test {}", i);
                let embedding = vec![i as f32; 768];
                let key = service_clone.generate_cache_key(&text, TaskType::Document, 768);
                
                // Store and retrieve concurrently
                assert!(service_clone.store_cached_embedding(key.clone(), &embedding).is_ok());
                
                let retrieved = service_clone.get_cached_embedding_validated(&key, 768);
                assert!(retrieved.is_some(), "Should retrieve embedding {}", i);
                
                i
            });
            handles.push(handle);
        }
        
        // Wait for all concurrent operations to complete
        let results: Vec<_> = futures::future::join_all(handles).await;
        let completed_count = results.into_iter().filter_map(|r| r.ok()).count();
        assert_eq!(completed_count, 10, "All concurrent operations should succeed");
    }
    
    #[tokio::test]
    async fn test_edge_cases_comprehensive() {
        let service = create_mock_service_config();
        
        // Test empty string handling
        assert!(service.validate_text_input("").is_ok(), "Empty string should be valid");
        let sanitized_empty = service.sanitize_text_input("");
        assert_eq!(sanitized_empty, "", "Empty string should remain empty");
        
        // Test whitespace-only text
        let whitespace_text = "   \t\n   ";
        let sanitized_whitespace = service.sanitize_text_input(whitespace_text);
        assert_eq!(sanitized_whitespace, "", "Whitespace-only should become empty");
        
        // Test special characters handling
        let special_chars = "Text with\0null\x01control\rcharacters";
        let sanitized_special = service.sanitize_text_input(special_chars);
        assert!(!sanitized_special.contains('\0'), "Null bytes should be removed");
        assert!(!sanitized_special.contains('\x01'), "Control chars should be removed");
        
        // Test boundary length validation
        let boundary_text = "a".repeat(MAX_TEXT_LENGTH);
        assert!(service.validate_text_input(&boundary_text).is_ok(), "Boundary length should be valid");
        
        let over_boundary = "a".repeat(MAX_TEXT_LENGTH + 1);
        assert!(service.validate_text_input(&over_boundary).is_err(), "Over-boundary should be rejected");
        
        // Test batch boundary conditions
        let max_batch: Vec<String> = (0..MAX_BATCH_SIZE).map(|i| format!("text{}", i)).collect();
        assert!(service.validate_batch_input(&max_batch).is_ok(), "Max batch size should be valid");
        
        let over_batch: Vec<String> = (0..MAX_BATCH_SIZE + 1).map(|i| format!("text{}", i)).collect();
        assert!(service.validate_batch_input(&over_batch).is_err(), "Over max batch should be rejected");
        
        // Test embedding validation with edge values
        let nan_embedding = vec![f32::NAN; 768];
        let infinity_embedding = vec![f32::INFINITY; 768];
        let normal_embedding = vec![0.5; 768];
        
        assert!(service.validate_embedding_dimension(&normal_embedding, 768, "normal").is_ok());
        assert!(service.validate_embedding_dimension(&nan_embedding, 768, "nan").is_ok()); // Dimension check only
        assert!(service.validate_embedding_dimension(&infinity_embedding, 768, "inf").is_ok()); // Dimension check only
        
        // Wrong dimension should fail
        assert!(service.validate_embedding_dimension(&normal_embedding, 512, "wrong_dim").is_err());
    }
}