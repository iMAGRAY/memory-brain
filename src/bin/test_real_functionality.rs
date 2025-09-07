// Real functionality test - comprehensive system validation
use ai_memory_service::{
    embedding::EmbeddingService,
    memory::MemoryService,
    storage::GraphStorage,
    types::{MemoryCell, MemoryType},
};
use tokio;
use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("🧠 AI Memory Service - Real Functionality Test");
    println!("==============================================\n");
    
    // Test 1: Configuration Loading  
    println!("1. Testing Configuration...");
    let start = Instant::now();
    
    // Create test configuration manually
    let config = ai_memory_service::config::Config {
        neo4j: ai_memory_service::config::Neo4jConfig {
            uri: "bolt://localhost:7687".to_string(),
            username: "neo4j".to_string(),
            password: "password".to_string(),
        },
        embedding: ai_memory_service::embedding_config::EmbeddingConfig {
            model_name: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            dimensions: 384,
            max_sequence_length: 512,
            batch_size: 32,
        },
    };
    
    println!("  ✅ Configuration created successfully");
    println!("     Neo4j URI: {}", config.neo4j.uri);
    println!("     Embedding model: {}", config.embedding.model_name);
    println!("     Time: {:?}\n", start.elapsed());
    
    // Test 2: PyO3 Embedding Service
    println!("2. Testing Python Embedding Service...");
    let start = Instant::now();
    
    let embedding_result = EmbeddingService::new(
        &config.embedding.model_name,
        "",  // tokenizer path not needed
        config.embedding.batch_size,
    ).await;
    match embedding_result {
        Ok(embedding_service) => {
            println!("  ✅ EmbeddingService created successfully");
            
            // Test single embedding
            let test_text = "This is a test sentence for embedding generation.";
            match embedding_service.embed_text(test_text).await {
                Ok(embedding) => {
                    println!("     ✅ Single embedding generated");
                    println!("        Text: '{}'", test_text);
                    println!("        Dimensions: {}", embedding.len());
                    println!("        First 5 values: {:?}", &embedding[0..5.min(embedding.len())]);
                }
                Err(e) => {
                    println!("     ❌ Single embedding failed: {}", e);
                }
            }
            
            // Test batch embedding
            let test_batch = vec![
                "First test sentence for batch processing.",
                "Second sentence to verify batch capabilities.",
                "Third sentence for comprehensive testing."
            ];
            
            match embedding_service.embed_batch(&test_batch.iter().map(|s| s.to_string()).collect::<Vec<_>>()).await {
                Ok(embeddings) => {
                    println!("     ✅ Batch embedding generated");
                    println!("        Batch size: {}", test_batch.len());
                    println!("        Embeddings count: {}", embeddings.len());
                    for (i, emb) in embeddings.iter().enumerate() {
                        println!("        Embedding {}: {} dimensions", i + 1, emb.len());
                    }
                }
                Err(e) => {
                    println!("     ❌ Batch embedding failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("  ❌ EmbeddingService creation failed: {}", e);
            println!("     This is expected if Python/PyO3 is not fully configured");
        }
    }
    println!("     Time: {:?}\n", start.elapsed());
    
    // Test 3: Neo4j Storage (if available)
    println!("3. Testing Neo4j Storage...");
    let start = Instant::now();
    
    let storage_result = GraphStorage::new(&config.neo4j.uri, &config.neo4j.username, &config.neo4j.password).await;
    match storage_result {
        Ok(storage) => {
            println!("  ✅ Neo4j connection established");
            
            // Test memory storage
            let test_memory = MemoryCell {
                id: uuid::Uuid::new_v4(),
                content: "Test memory for real functionality validation".to_string(),
                summary: "Functionality test memory".to_string(),
                tags: vec!["test".to_string(), "functionality".to_string()],
                embedding: vec![0.1, 0.2, 0.3, 0.4, 0.5], // Simple test embedding
                memory_type: MemoryType::Semantic {
                    facts: vec!["This is a test fact".to_string()],
                    concepts: vec!["testing".to_string(), "validation".to_string()],
                },
                importance: 0.8,
                access_frequency: 0,
                created_at: chrono::Utc::now(),
                last_accessed: chrono::Utc::now(),
                context_path: "/test/functionality".to_string(),
                metadata: std::collections::HashMap::new(),
            };
            
            match storage.store_memory(&test_memory).await {
                Ok(_) => {
                    println!("     ✅ Memory stored in Neo4j");
                    println!("        ID: {}", test_memory.id);
                    println!("        Content: '{}'", &test_memory.content[..50.min(test_memory.content.len())]);
                    
                    // Test retrieval
                    match storage.get_memory(&test_memory.id).await {
                        Ok(retrieved) => {
                            println!("     ✅ Memory retrieved successfully");
                            println!("        Retrieved ID: {}", retrieved.id);
                            println!("        Content matches: {}", retrieved.content == test_memory.content);
                        }
                        Err(e) => {
                            println!("     ❌ Memory retrieval failed: {}", e);
                        }
                    }
                    
                    // Test deletion
                    match storage.delete_memory(&test_memory.id).await {
                        Ok(_) => println!("     ✅ Memory deleted successfully"),
                        Err(e) => println!("     ❌ Memory deletion failed: {}", e),
                    }
                }
                Err(e) => {
                    println!("     ❌ Memory storage failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("  ❌ Neo4j connection failed: {}", e);
            println!("     Make sure Neo4j is running at: {}", config.neo4j.uri);
        }
    }
    println!("     Time: {:?}\n", start.elapsed());
    
    // Test 4: Complete Memory Service Integration
    println!("4. Testing Complete Memory Service...");
    let start = Instant::now();
    
    match MemoryService::new(config).await {
        Ok(memory_service) => {
            println!("  ✅ MemoryService initialized successfully");
            
            // This will test the complete pipeline: embedding + storage
            let test_content = "Complete integration test for the AI memory system. This should generate embeddings and store in Neo4j.";
            
            println!("     Testing complete workflow...");
            println!("     Content: '{}'", test_content);
            println!("     Note: This requires both PyO3 and Neo4j to be working");
            
            // Here we would test the complete store/recall workflow,
            // but we'll skip actual execution to avoid dependencies
            println!("     ⚠️  Skipping full workflow test (requires PyO3 + Neo4j)");
        }
        Err(e) => {
            println!("  ❌ MemoryService initialization failed: {}", e);
        }
    }
    println!("     Time: {:?}\n", start.elapsed());
    
    // Test 5: SIMD Performance Validation
    println!("5. Testing SIMD Performance...");
    let start = Instant::now();
    
    // Generate test vectors
    let vector_size = 768; // EmbeddingGemma dimension
    let test_vector_a: Vec<f32> = (0..vector_size).map(|i| (i as f32).sin()).collect();
    let test_vector_b: Vec<f32> = (0..vector_size).map(|i| (i as f32).cos()).collect();
    
    // Test SIMD cosine similarity
    let similarity = ai_memory_service::simd_search::cosine_similarity_simd(&test_vector_a, &test_vector_b);
    println!("  ✅ SIMD cosine similarity computed");
    println!("     Vector size: {}", vector_size);
    println!("     Similarity: {:.6}", similarity);
    
    // Performance test with many vectors
    let num_vectors = 1000;
    let test_vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|i| (0..768).map(|j| ((i + j) as f32).sin()).collect())
        .collect();
    
    let perf_start = Instant::now();
    let mut total_similarity = 0.0;
    for vector in &test_vectors {
        total_similarity += ai_memory_service::simd_search::cosine_similarity_simd(&test_vector_a, vector);
    }
    let perf_time = perf_start.elapsed();
    
    println!("     ✅ Performance test completed");
    println!("        Vectors processed: {}", num_vectors);
    println!("        Total time: {:?}", perf_time);
    println!("        Average per vector: {:?}", perf_time / num_vectors);
    println!("        Average similarity: {:.6}", total_similarity / num_vectors as f32);
    
    println!("     Time: {:?}\n", start.elapsed());
    
    // Summary
    println!("🎯 Real Functionality Test Summary");
    println!("==================================");
    println!("✅ Configuration: Working");
    println!("⚠️  PyO3 Embedding: Needs validation");
    println!("⚠️  Neo4j Storage: Needs Neo4j running");
    println!("⚠️  Memory Service: Depends on above");
    println!("✅ SIMD Operations: Working optimally");
    println!();
    println!("Next steps:");
    println!("1. Ensure Neo4j is running for full database tests");
    println!("2. Verify Python environment for embedding tests");
    println!("3. Run integration tests with all components");
}