//! Modern API Integration Tests
//! 
//! Comprehensive testing using axum-test framework for HTTP API validation

use ai_memory_service::{api::{self, ApiState}, memory::MemoryService, config::Config};
use axum::http::{StatusCode};
use axum_test::TestServer;
use serde_json::json;
use std::env;
use std::sync::Arc;

// Test constants - consolidated for better organization
const API_BASE_PATH: &str = "/api";
const HEALTH_ENDPOINT: &str = "/health";
const STORE_ENDPOINT: &str = "/memories";
const RECALL_ENDPOINT: &str = "/recall";

#[tokio::test]
async fn test_health_endpoint() {
    let server = create_test_server().await;
    
    let response = server
        .get(HEALTH_ENDPOINT)
        .await;
    
    assert_eq!(response.status_code(), StatusCode::OK);
    
    let json: serde_json::Value = response.json();
    assert_eq!(json["status"], "healthy");
    assert!(json["services"]["embedding"].as_bool().unwrap());
    assert!(json["services"]["storage"].as_bool().unwrap());
    assert!(json["services"]["cache"].as_bool().unwrap());
}

#[tokio::test]
async fn test_store_memory_endpoint() {
    let server = create_test_server().await;
    
    let store_request = json!({
        "content": "Rust is a systems programming language",
        "memory_type": "semantic",
        "importance": 0.8,
        "context": {
            "path": "/programming/rust",
            "tags": ["rust", "programming"]
        }
    });
    
    let response = server
        .post(STORE_ENDPOINT)
        .json(&store_request)
        .await;
    
    assert_eq!(response.status_code(), StatusCode::OK);
    
    let json: serde_json::Value = response.json();
    assert!(json["memory_id"].as_str().is_some());
    assert!(json["success"].as_bool().unwrap_or(false));
}

#[tokio::test]
async fn test_recall_memory_endpoint() {
    let server = create_test_server().await;
    
    // First store a memory
    let store_request = json!({
        "content": "Tokio is an async runtime for Rust",
        "memory_type": "semantic",
        "importance": 0.9,
        "context": {
            "path": "/programming/rust/async",
            "tags": ["tokio", "async", "rust"]
        }
    });
    
    let store_response = server
        .post(STORE_ENDPOINT)
        .json(&store_request)
        .await;
    
    assert_eq!(store_response.status_code(), StatusCode::OK);
    
    // Now recall it
    let recall_request = json!({
        "query": "Tell me about Tokio async runtime",
        "limit": 10,
        "similarity_threshold": 0.5,
        "context_hint": "/programming"
    });
    
    let recall_response = server
        .post(RECALL_ENDPOINT)
        .json(&recall_request)
        .await;
    
    assert_eq!(recall_response.status_code(), StatusCode::OK);
    
    let json: serde_json::Value = recall_response.json();
    assert!(json["semantic_layer"].is_array());
    assert!(json["contextual_layer"].is_array());
    assert!(json["detailed_layer"].is_array());
}

#[tokio::test]
async fn test_error_handling() {
    let server = create_test_server().await;
    
    // Test invalid JSON
    let response = server
        .post(STORE_ENDPOINT)
        .text("invalid json")
        .await;
    
    assert_eq!(response.status_code(), StatusCode::BAD_REQUEST);
    
    let error_json: serde_json::Value = response.json();
    assert!(error_json.get("error").is_some());
    
    // Test missing required fields
    let incomplete_request = json!({
        "content": "Test content"
        // missing memory_type
    });
    
    let response = server
        .post(STORE_ENDPOINT)
        .json(&incomplete_request)
        .await;
    
    assert_eq!(response.status_code(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_metrics_endpoint() {
    let server = create_test_server().await;
    
    let response = server
        .get("/metrics")
        .await;
    
    assert_eq!(response.status_code(), StatusCode::OK);
    
    let metrics_text = response.text();
    
    // Check for expected metric names
    assert!(metrics_text.contains("memory_store_duration_seconds") || 
            metrics_text.contains("# TYPE") || 
            metrics_text.contains("# HELP"));
}

// Helper function to create test server
async fn create_test_server() -> TestServer {
    let config = create_test_config();
    let service = MemoryService::new(config).await
        .expect("Failed to create memory service");
    
    let api_state = ApiState {
        memory_service: Arc::new(service),
    };
    let app = api::create_router(api_state);
    TestServer::new(app).expect("Failed to create test server")
}

fn create_test_config() -> Config {
    let neo4j_password = env::var("NEO4J_TEST_PASSWORD")
        .unwrap_or_else(|_| "test_password".to_string());
    
    Config {
        server: ai_memory_service::config::ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            workers: 1,
            environment: "test".to_string(),
            cors_origins: vec!["http://localhost:3000".to_string()],
        },
        storage: ai_memory_service::config::StorageConfig {
            neo4j_uri: env::var("NEO4J_TEST_URI")
                .unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
            neo4j_user: env::var("NEO4J_TEST_USER")
                .unwrap_or_else(|_| "neo4j".to_string()),
            neo4j_password,
            connection_pool_size: 5,
        },
        embedding: ai_memory_service::config::EmbeddingConfig {
            model_path: env::var("TEST_MODEL_PATH")
                .unwrap_or_else(|_| "./models/embeddinggemma-300m-ONNX/model.onnx".to_string()),
            tokenizer_path: env::var("TEST_TOKENIZER_PATH")
                .unwrap_or_else(|_| "./models/embeddinggemma-300m-ONNX/tokenizer.json".to_string()),
            batch_size: 16,
            max_sequence_length: 512,
        },
        cache: ai_memory_service::config::CacheConfig {
            l1_size: 100,
            l2_size: 1000,
            ttl_seconds: 300,
            compression_enabled: true,
        },
        brain: ai_memory_service::config::BrainConfig {
            model_name: "test_model".to_string(),
            min_importance: 0.5,
            enable_sentiment: true,
        },
    }
}