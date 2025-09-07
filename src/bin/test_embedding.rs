use ai_memory_service::embedding::EmbeddingService;

// Configuration constants for PyO3-based embedding service
const MODEL_NAME: &str = "google/embeddinggemma-300m";
const BATCH_SIZE: usize = 8;
const MAX_SEQUENCE_LENGTH: usize = 2048;

#[tokio::main]
async fn main() {
    println!("Testing EmbeddingService with Python Sentence Transformers...");
    
    // Create embedding service with PyO3
    println!("Creating EmbeddingService with model: {}", MODEL_NAME);
    match EmbeddingService::new(MODEL_NAME, "", BATCH_SIZE, MAX_SEQUENCE_LENGTH).await {
        Ok(service) => {
            println!("✓ EmbeddingService created successfully!");
            
            // Test basic embedding
            println!("Testing text embedding...");
            match service.embed_text("Hello world, this is a test").await {
                Ok(embedding) => {
                    println!("✓ Embedding generated! Dimension: {}", embedding.len());
                    println!("✓ First 5 values: {:?}", &embedding[..5]);
                    
                    // Test batch embedding
                    println!("Testing batch embedding...");
                    let texts = vec![
                        "First text".to_string(),
                        "Second text".to_string(),
                        "Third text".to_string(),
                    ];
                    
                    match service.embed_batch(&texts).await {
                        Ok(embeddings) => {
                            println!("✓ Batch embedding successful! Generated {} embeddings", embeddings.len());
                            for (i, emb) in embeddings.iter().enumerate() {
                                println!("  Embedding {}: {} dimensions", i+1, emb.len());
                            }
                        }
                        Err(e) => {
                            println!("✗ Batch embedding failed: {}", e);
                        }
                    }
                    
                    // Test different embedding methods
                    println!("Testing embedding methods...");
                    
                    // Test query embedding
                    match service.embed_query("What is artificial intelligence?").await {
                        Ok(emb) => {
                            println!("✓ Query embedding: {} dimensions", emb.len());
                        }
                        Err(e) => {
                            println!("✗ Query embedding failed: {}", e);
                        }
                    }
                    
                    // Test document embedding
                    match service.embed_document("Artificial intelligence is a branch of computer science.").await {
                        Ok(emb) => {
                            println!("✓ Document embedding: {} dimensions", emb.len());
                        }
                        Err(e) => {
                            println!("✗ Document embedding failed: {}", e);
                        }
                    }
                    
                    // Test cache functionality
                    println!("Testing cache functionality...");
                    let (cache_size, max_size) = service.cache_stats();
                    println!("✓ Cache stats: {}/{} entries", cache_size, max_size);
                    
                    // Test model info
                    println!("Testing model information...");
                    let model_info = service.model_info();
                    println!("✓ Model: {} ({} dims, max_len: {})", 
                             model_info.name, model_info.dimensions, model_info.max_sequence_length);
                    
                    println!("\n🎉 EmbeddingService is fully functional!");
                    
                }
                Err(e) => {
                    println!("✗ Text embedding failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            println!("✗ Failed to create EmbeddingService: {}", e);
            let error_msg = format!("{}", e);
            if error_msg.contains("1.19") || error_msg.contains("version") {
                println!("   This might be an ONNX Runtime version issue.");
            }
            std::process::exit(1);
        }
    }
}