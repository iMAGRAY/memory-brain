//! Integration tests for real HTTP-based Embedding Service (no mocks)

use ai_memory_service::embedding::{EmbeddingService, TaskType};
use std::env;

async fn ensure_embedding_server_available() {
    let url = env::var("EMBEDDING_SERVER_URL").unwrap_or_else(|_| "http://localhost:8090".to_string());
    let health = format!("{}/health", url.trim_end_matches('/'));
    let client = reqwest::Client::new();
    let resp = client.get(&health).send().await;
    if resp.is_err() || !resp.unwrap().status().is_success() {
        panic!("Embedding server health check failed at {}. Start real embedding_server or publish port 8090.", health);
    }
}

#[tokio::test]
async fn test_embedding_service_initialization() {
    ensure_embedding_server_available().await;
    let result = EmbeddingService::new(
        "embeddinggemma-300m",
        "",
        8,
        2048,
    ).await;
    assert!(result.is_ok(), "Failed to initialize EmbeddingService: {:?}", result.err());
}

#[tokio::test]
async fn test_simple_embedding_generation() {
    ensure_embedding_server_available().await;
    let service = EmbeddingService::new("embeddinggemma-300m", "", 8, 2048)
        .await
        .expect("Failed to initialize service");
    let text = "The quick brown fox jumps over the lazy dog";
    let embedding = service.embed(text, TaskType::Document).await
        .expect("Failed to generate embedding");
    assert!(embedding.len() == 768 || embedding.len() == 512, "Unexpected embedding dimension: {}", embedding.len());
    assert!(embedding.iter().all(|v| v.is_finite()), "Embedding contains non-finite values");
}

#[tokio::test]
async fn test_query_document_embeddings_similarity() {
    ensure_embedding_server_available().await;
    let service = EmbeddingService::new("embeddinggemma-300m", "", 8, 2048)
        .await
        .expect("Failed to initialize service");
    let query = "What is machine learning?";
    let document = "Machine learning is a subset of artificial intelligence that enables computers to learn from data.";
    let q = service.embed(query, TaskType::Query).await.expect("Query embed failed");
    let d = service.embed(document, TaskType::Document).await.expect("Doc embed failed");
    assert_eq!(q.len(), d.len(), "Query/Doc dimensions must match");
    let sim = cosine_similarity(&q, &d);
    assert!(sim > 0.0 && sim < 1.0, "Similarity out of range: {}", sim);
}

#[tokio::test]
async fn test_batch_processing() {
    ensure_embedding_server_available().await;
    let service = EmbeddingService::new("embeddinggemma-300m", "", 4, 2048)
        .await
        .expect("Failed to initialize service");
    let texts = vec![
        "First test sentence".to_string(),
        "Second test sentence with more words".to_string(),
        "Third sentence".to_string(),
        "Fourth and final test sentence in this batch".to_string(),
    ];
    let embeddings = service.embed_batch(&texts, TaskType::Document).await
        .expect("Failed to generate batch embeddings");
    assert_eq!(embeddings.len(), texts.len());
    for emb in embeddings {
        assert!(emb.len() == 768 || emb.len() == 512);
        assert!(emb.iter().all(|v| v.is_finite()));
    }
}

#[tokio::test]
async fn test_empty_input_handling() {
    ensure_embedding_server_available().await;
    let service = EmbeddingService::new("embeddinggemma-300m", "", 8, 2048)
        .await
        .expect("Failed to initialize service");
    let result = service.embed("", TaskType::Document).await;
    assert!(result.is_err(), "Should fail on empty input");
}

#[tokio::test]
async fn test_long_input_truncation() {
    ensure_embedding_server_available().await;
    let service = EmbeddingService::new("embeddinggemma-300m", "", 8, 2048)
        .await
        .expect("Failed to initialize service");
    let long_text = "word ".repeat(5000);
    let embedding = service.embed(&long_text, TaskType::Document).await
        .expect("Failed to embed long text");
    assert!(embedding.len() == 768 || embedding.len() == 512);
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x,y)| x*y).sum();
    let na: f32 = a.iter().map(|x| x*x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x*x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-9)
}
