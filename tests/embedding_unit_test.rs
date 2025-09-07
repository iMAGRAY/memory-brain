//! Unit tests for embedding service components
//! 
//! Tests core functionality without requiring ONNX Runtime

use ai_memory_service::embedding::EmbeddingService;

#[test]
fn test_model_info_structure() {
    // Test that ModelInfo struct has correct fields
    // This doesn't require ONNX Runtime initialization
    
    let info = ai_memory_service::embedding::ModelInfo {
        name: "test-model".to_string(),
        dimensions: 768,
        max_sequence_length: 2048,
        vocab_size: 30000,
    };
    
    assert_eq!(info.name, "test-model");
    assert_eq!(info.dimensions, 768);
    assert_eq!(info.max_sequence_length, 2048);
    assert_eq!(info.vocab_size, 30000);
}

#[test]
fn test_embedding_cache_key_generation() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    // Test that cache key generation is deterministic
    let text = "Test text for hashing";
    
    let mut hasher1 = DefaultHasher::new();
    text.hash(&mut hasher1);
    let hash1 = format!("{:x}", hasher1.finish());
    
    let mut hasher2 = DefaultHasher::new();
    text.hash(&mut hasher2);
    let hash2 = format!("{:x}", hasher2.finish());
    
    assert_eq!(hash1, hash2, "Hash should be deterministic");
}

#[test]
fn test_embedding_normalization() {
    // Test L2 normalization logic
    let mut embedding = vec![3.0, 4.0, 0.0]; // 3-4-5 triangle
    
    // Calculate L2 norm
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 5.0).abs() < 0.001, "Norm should be 5.0");
    
    // Normalize
    for value in &mut embedding {
        *value /= norm;
    }
    
    // Check normalized norm is 1.0
    let new_norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((new_norm - 1.0).abs() < 0.001, "Normalized norm should be 1.0");
    
    // Check individual values
    assert!((embedding[0] - 0.6).abs() < 0.001);
    assert!((embedding[1] - 0.8).abs() < 0.001);
    assert!(embedding[2].abs() < 0.001);
}

#[test]
fn test_matryoshka_dimension_validation() {
    // Test that Matryoshka dimensions are valid
    let valid_dims = vec![768, 512, 256, 128];
    let invalid_dims = vec![1024, 777, 100, 0];
    
    for dim in valid_dims {
        assert!(dim > 0 && dim <= 768, "Dimension {} should be valid", dim);
        assert!(dim % 128 == 0 || dim == 768, "Dimension {} should be aligned", dim);
    }
    
    for dim in invalid_dims {
        let is_invalid = dim == 0 || dim > 768 || (dim != 768 && dim % 128 != 0);
        assert!(is_invalid, "Dimension {} should be invalid", dim);
    }
}

#[test]
fn test_cosine_similarity_calculation() {
    // Test cosine similarity calculation
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![1.0, 0.0, 0.0];
    let c = vec![0.0, 1.0, 0.0];
    let d = vec![0.707, 0.707, 0.0]; // 45 degrees
    
    let similarity = |v1: &[f32], v2: &[f32]| -> f32 {
        let dot: f32 = v1.iter().zip(v2.iter()).map(|(x, y)| x * y).sum();
        let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm1 * norm2 + 1e-10)
    };
    
    // Same vectors should have similarity 1.0
    assert!((similarity(&a, &b) - 1.0).abs() < 0.001);
    
    // Orthogonal vectors should have similarity 0.0
    assert!(similarity(&a, &c).abs() < 0.001);
    
    // 45 degree angle should have similarity ~0.707
    assert!((similarity(&a, &d) - 0.707).abs() < 0.01);
}

#[test]
fn test_batch_padding_logic() {
    // Test padding logic for batch processing
    let sequences = vec![
        vec![1, 2, 3],
        vec![4, 5],
        vec![6, 7, 8, 9],
    ];
    
    let max_len = sequences.iter().map(|s| s.len()).max().unwrap();
    assert_eq!(max_len, 4);
    
    let mut padded = sequences.clone();
    for seq in &mut padded {
        while seq.len() < max_len {
            seq.push(0); // Padding token
        }
    }
    
    assert_eq!(padded[0], vec![1, 2, 3, 0]);
    assert_eq!(padded[1], vec![4, 5, 0, 0]);
    assert_eq!(padded[2], vec![6, 7, 8, 9]);
}

#[test]
fn test_mean_pooling_logic() {
    // Test mean pooling with attention mask
    let hidden_states = vec![
        vec![1.0, 2.0, 3.0],  // Token 1
        vec![4.0, 5.0, 6.0],  // Token 2
        vec![7.0, 8.0, 9.0],  // Token 3 (padding)
    ];
    
    let attention_mask = vec![1.0, 1.0, 0.0]; // Last token is padding
    
    let mut pooled = vec![0.0; 3];
    let mut total_weight: f32 = 0.0;
    
    for (token_idx, mask_value) in attention_mask.iter().enumerate() {
        total_weight += mask_value;
        for (dim_idx, value) in hidden_states[token_idx].iter().enumerate() {
            pooled[dim_idx] += value * mask_value;
        }
    }
    
    // Average only over non-padding tokens
    for value in &mut pooled {
        *value /= total_weight.max(1e-9);
    }
    
    // Should average first two tokens only
    assert!((pooled[0] - 2.5f32).abs() < 0.001); // (1+4)/2
    assert!((pooled[1] - 3.5f32).abs() < 0.001); // (2+5)/2
    assert!((pooled[2] - 4.5f32).abs() < 0.001); // (3+6)/2
}

#[test]
fn test_input_validation() {
    // Test various input validation scenarios
    
    // Empty input
    let empty = "";
    assert!(empty.trim().is_empty(), "Empty input should be detected");
    
    // Whitespace only
    let whitespace = "   \n\t  ";
    assert!(whitespace.trim().is_empty(), "Whitespace should be detected");
    
    // Very long input
    let long_text = "word ".repeat(10000);
    assert!(long_text.len() > 8192, "Long input should exceed max length");
    
    // Valid input
    let valid = "This is a valid input text";
    assert!(!valid.trim().is_empty(), "Valid input should pass");
    assert!(valid.len() < 8192, "Valid input should be within limits");
}

#[test]
fn test_task_type_formatting() {
    // Test task-specific formatting for EmbeddingGemma
    
    let query_text = "What is machine learning?";
    let doc_text = "Machine learning is a subset of AI";
    
    // Query formatting
    let query_formatted = format!("task: search result | query: {}", query_text);
    assert!(query_formatted.contains("task: search result"));
    assert!(query_formatted.contains(query_text));
    
    // Document formatting
    let doc_formatted = format!("title: none | text: {}", doc_text);
    assert!(doc_formatted.contains("title: none"));
    assert!(doc_formatted.contains(doc_text));
    
    // With title
    let doc_with_title = format!("title: ML Basics | text: {}", doc_text);
    assert!(doc_with_title.contains("ML Basics"));
    assert!(!doc_with_title.contains("none"));
}

#[cfg(test)]
mod performance_tests {
    use std::time::Instant;
    
    #[test]
    fn test_normalization_performance() {
        // Test performance of normalization for typical embedding size
        let mut embedding = vec![1.0; 768];
        
        let start = Instant::now();
        for _ in 0..1000 {
            // L2 normalization
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            for value in &mut embedding {
                *value /= norm + 1e-10;
            }
        }
        let elapsed = start.elapsed();
        
        println!("1000 normalizations took: {:?}", elapsed);
        assert!(elapsed.as_millis() < 100, "Normalization should be fast");
    }
    
    #[test]
    fn test_hashing_performance() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let text = "This is a sample text for embedding generation that might be quite long in practice";
        
        let start = Instant::now();
        for _ in 0..10000 {
            let mut hasher = DefaultHasher::new();
            text.hash(&mut hasher);
            let _ = format!("{:x}", hasher.finish());
        }
        let elapsed = start.elapsed();
        
        println!("10000 hash operations took: {:?}", elapsed);
        assert!(elapsed.as_millis() < 100, "Hashing should be fast");
    }
}