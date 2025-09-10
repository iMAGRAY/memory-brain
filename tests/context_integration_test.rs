//! Context endpoints and context-aware search integration tests

use ai_memory_service::{api, config::Config, memory::MemoryService};
use axum_test::TestServer;
use axum::http::StatusCode;
use serde_json::json;
use std::env;
use std::sync::Arc;

#[tokio::test]
async fn test_contexts_and_search_by_context() {
    // Build service and router
    let cfg = create_test_config();
    let service = MemoryService::new(cfg).await.expect("service");
    let api_cfg = api::ApiConfig { host: "127.0.0.1".into(), port: 0, max_body_size: 10*1024*1024, enable_cors: true, enable_compression: true, enable_tracing: false };
    let app = api::create_router(Arc::new(service), None, api_cfg);
    let server = TestServer::new(app).expect("server");

    // Store 2 memories in 2 different contexts via compat route
    let ctx_a = "tests_ctxA"; // single-segment for /context/:path
    let ctx_b = "tests_ctxB";
    let store_a = json!({
        "content": "Context A note",
        "memory_type": "semantic",
        "importance": 0.6,
        "context": {"path": ctx_a, "tags": ["a"]}
    });
    let store_b = json!({
        "content": "Context B note",
        "memory_type": "semantic",
        "importance": 0.7,
        "context": {"path": ctx_b, "tags": ["b"]}
    });
    let r1 = server.post("/memories").json(&store_a).await;
    assert_eq!(r1.status_code(), StatusCode::OK);
    let r2 = server.post("/memories").json(&store_b).await;
    assert_eq!(r2.status_code(), StatusCode::OK);

    // Stats should reflect contexts count
    let resp = server.get("/stats").await;
    assert_eq!(resp.status_code(), StatusCode::OK);
    let js: serde_json::Value = resp.json();
    let total_contexts = js["statistics"]["total_contexts"].as_u64().unwrap_or(0);
    assert!(total_contexts >= 1);

    // Search by context A
    let search_req = json!({
        "query": "note",
        "limit": 5,
        "context": ctx_a
    });
    let search_resp = server.post("/search/context").json(&search_req).await;
    assert_eq!(search_resp.status_code(), StatusCode::OK);
    let srch: serde_json::Value = search_resp.json();
    assert!(srch["results"].is_array());
}

fn create_test_config() -> Config {
    let neo4j_password = env::var("NEO4J_TEST_PASSWORD").unwrap_or_else(|_| "test_password".to_string());
    Config {
        server: ai_memory_service::config::ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 0,
            workers: 1,
            environment: "test".to_string(),
            cors_origins: vec!["http://localhost:3000".to_string()],
        },
        storage: ai_memory_service::config::StorageConfig {
            neo4j_uri: env::var("NEO4J_TEST_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
            neo4j_user: env::var("NEO4J_TEST_USER").unwrap_or_else(|_| "neo4j".to_string()),
            neo4j_password,
            connection_pool_size: 5,
        },
        embedding: ai_memory_service::config::EmbeddingConfig {
            model_path: env::var("TEST_MODEL_PATH").unwrap_or_else(|_| "./models/embeddinggemma-300m-ONNX/model.onnx".to_string()),
            tokenizer_path: env::var("TEST_TOKENIZER_PATH").unwrap_or_else(|_| "./models/embeddinggemma-300m-ONNX/tokenizer.json".to_string()),
            batch_size: 16,
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
            importance_threshold: 0.1,
            consolidation_interval: 300,
            decay_rate: 0.01,
        },
    }
}
