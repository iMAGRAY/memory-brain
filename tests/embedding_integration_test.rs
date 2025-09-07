//! Integration tests for EmbeddingGemma model
//! 
//! Tests actual model loading, tokenization, and embedding generation

use ai_memory_service::embedding::EmbeddingService;
use std::path::PathBuf;

const MODEL_PATH: &str = "./models/embeddinggemma-300m-ONNX/model.onnx";
const TOKENIZER_PATH: &str = "./models/embeddinggemma-300m-ONNX/tokenizer.json";

#[tokio::test]
async fn test_embedding_service_initialization() {
    let model_path = PathBuf::from(MODEL_PATH);
    let tokenizer_path = PathBuf::from(TOKENIZER_PATH);
    
    // Check if model files exist
    assert!(model_path.exists(), "Model file not found at: {:?}", model_path);
    assert!(tokenizer_path.exists(), "Tokenizer file not found at: {:?}", tokenizer_path);
    
    // Check for data file
    let data_path = PathBuf::from("./models/embeddinggemma-300m-ONNX/model.onnx_data");
    assert!(data_path.exists(), "Model data file not found at: {:?}", data_path);
    
    // Initialize embedding service
    let result = EmbeddingService::new(
        model_path.to_str().unwrap(),
        tokenizer_path.to_str().unwrap(),
        8,  // batch_size
        2048,  // max_sequence_length
    ).await;
    
    assert!(result.is_ok(), "Failed to initialize EmbeddingService: {:?}", result.err());
    
    let service = result.unwrap();
    
    // Verify model info
    let model_info = service.model_info();
    assert_eq!(model_info.dimensions, 768, "Expected 768 dimensions");
    assert_eq!(model_info.name, "EmbeddingGemma-300m");
}

#[tokio::test]
async fn test_simple_embedding_generation() {
    let model_path = PathBuf::from(MODEL_PATH);
    let tokenizer_path = PathBuf::from(TOKENIZER_PATH);
    
    let service = EmbeddingService::new(
        model_path.to_str().unwrap(),
        tokenizer_path.to_str().unwrap(),
        8,
        2048,
    ).await.expect("Failed to initialize service");
    
    // Test simple text embedding
    let text = "The quick brown fox jumps over the lazy dog";
    let embedding = service.embed_text(text).await;
    
    assert!(embedding.is_ok(), "Failed to generate embedding: {:?}", embedding.err());
    
    let embedding = embedding.unwrap();
    assert_eq!(embedding.len(), 768, "Expected 768-dimensional embedding");
    
    // Check that embedding is normalized (L2 norm ≈ 1.0)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Embedding not normalized: norm = {}", norm);
    
    // Check that values are reasonable (not NaN or infinite)
    for value in &embedding {
        assert!(value.is_finite(), "Found non-finite value in embedding");
        assert!(value.abs() < 10.0, "Found unreasonably large value: {}", value);
    }
}

#[tokio::test]
async fn test_query_document_embeddings() {
    let model_path = PathBuf::from(MODEL_PATH);
    let tokenizer_path = PathBuf::from(TOKENIZER_PATH);
    
    let service = EmbeddingService::new(
        model_path.to_str().unwrap(),
        tokenizer_path.to_str().unwrap(),
        8,
        2048,
    ).await.expect("Failed to initialize service");
    
    // Test query embedding
    let query = "What is machine learning?";
    let query_embedding = service.embed_query(query).await
        .expect("Failed to generate query embedding");
    
    // Test document embedding
    let document = "Machine learning is a subset of artificial intelligence that enables computers to learn from data.";
    let doc_embedding = service.embed_document(document).await
        .expect("Failed to generate document embedding");
    
    // Both should be 768-dimensional
    assert_eq!(query_embedding.len(), 768);
    assert_eq!(doc_embedding.len(), 768);
    
    // Calculate cosine similarity
    let similarity = cosine_similarity(&query_embedding, &doc_embedding);
    println!("Query-Document similarity: {}", similarity);
    
    // Should have reasonable similarity (not 0 or 1)
    assert!(similarity > 0.0 && similarity < 1.0, "Unexpected similarity: {}", similarity);
}

#[tokio::test]
async fn test_matryoshka_dimensions() {
    let model_path = PathBuf::from(MODEL_PATH);
    let tokenizer_path = PathBuf::from(TOKENIZER_PATH);
    
    let service = EmbeddingService::new(
        model_path.to_str().unwrap(),
        tokenizer_path.to_str().unwrap(),
        8,
        2048,
    ).await.expect("Failed to initialize service");
    
    let text = "Test text for dimension reduction";
    
    // Test different Matryoshka dimensions
    let dims = [768, 512, 256, 128];
    for dim in dims {
        // Use batch method with single text to get dimension-specific embedding
        let embeddings = service.embed_batch_with_dimension(
            &[text.to_string()], 
            ai_memory_service::embedding::TaskType::Query, 
            dim
        ).await
            .expect(&format!("Failed to generate {}-dim embedding", dim));
        
        // Add robustness check as recommended by hook
        assert!(!embeddings.is_empty(), "Embeddings vector should not be empty");
        let embedding = &embeddings[0];
        
        assert_eq!(embedding.len(), dim, "Expected {}-dimensional embedding", dim);
        
        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding not normalized for dim {}: norm = {}", dim, norm);
    }
}

#[tokio::test]
async fn test_batch_processing() {
    let model_path = PathBuf::from(MODEL_PATH);
    let tokenizer_path = PathBuf::from(TOKENIZER_PATH);
    
    let service = EmbeddingService::new(
        model_path.to_str().unwrap(),
        tokenizer_path.to_str().unwrap(),
        4,  // smaller batch size for testing
        2048,
    ).await.expect("Failed to initialize service");
    
    let texts = vec![
        "First test sentence".to_string(),
        "Second test sentence with more words".to_string(),
        "Third sentence".to_string(),
        "Fourth and final test sentence in this batch".to_string(),
        "Fifth sentence to test multiple batches".to_string(),
    ];
    
    let embeddings = service.embed_batch(&texts).await
        .expect("Failed to generate batch embeddings");
    
    assert_eq!(embeddings.len(), texts.len(), "Number of embeddings doesn't match input");
    
    for (i, embedding) in embeddings.iter().enumerate() {
        assert_eq!(embedding.len(), 768, "Wrong dimension for text {}", i);
        
        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding {} not normalized: norm = {}", i, norm);
    }
}

#[tokio::test]
async fn test_empty_input_handling() {
    let model_path = PathBuf::from(MODEL_PATH);
    let tokenizer_path = PathBuf::from(TOKENIZER_PATH);
    
    let service = EmbeddingService::new(
        model_path.to_str().unwrap(),
        tokenizer_path.to_str().unwrap(),
        8,
        2048,
    ).await.expect("Failed to initialize service");
    
    // Test empty string
    let result = service.embed_text("").await;
    assert!(result.is_err(), "Should fail on empty input");
    
    // Test whitespace only
    let result = service.embed_text("   \n\t  ").await;
    assert!(result.is_err(), "Should fail on whitespace-only input");
}

#[tokio::test]
async fn test_long_input_truncation() {
    let model_path = PathBuf::from(MODEL_PATH);
    let tokenizer_path = PathBuf::from(TOKENIZER_PATH);
    
    let service = EmbeddingService::new(
        model_path.to_str().unwrap(),
        tokenizer_path.to_str().unwrap(),
        8,
        2048,
    ).await.expect("Failed to initialize service");
    
    // Create very long input (should be truncated to max_sequence_length)
    let long_text = "word ".repeat(5000);
    
    let embedding = service.embed_text(&long_text).await
        .expect("Failed to generate embedding for long text");
    
    assert_eq!(embedding.len(), 768, "Expected 768-dimensional embedding");
    
    // Should still be normalized
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Embedding not normalized: norm = {}", norm);
}

#[tokio::test]
async fn test_cache_effectiveness() {
    let model_path = PathBuf::from(MODEL_PATH);
    let tokenizer_path = PathBuf::from(TOKENIZER_PATH);
    
    let service = EmbeddingService::new(
        model_path.to_str().unwrap(),
        tokenizer_path.to_str().unwrap(),
        8,
        2048,
    ).await.expect("Failed to initialize service");
    
    let text = "This text will be embedded multiple times";
    
    // First embedding (not cached)
    let start = std::time::Instant::now();
    let embedding1 = service.embed_text(text).await
        .expect("Failed to generate first embedding");
    let first_time = start.elapsed();
    
    // Second embedding (should be cached)
    let start = std::time::Instant::now();
    let embedding2 = service.embed_text(text).await
        .expect("Failed to generate second embedding");
    let cached_time = start.elapsed();
    
    // Embeddings should be identical
    for (v1, v2) in embedding1.iter().zip(embedding2.iter()) {
        assert!((v1 - v2).abs() < 1e-6, "Cached embedding differs");
    }
    
    // Cached access should be much faster
    println!("First embedding: {:?}, Cached: {:?}", first_time, cached_time);
    assert!(cached_time < first_time / 2, "Cache doesn't seem to be working effectively");
}

// Helper function for cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same dimension");
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    dot_product / (norm_a * norm_b + 1e-10)
}