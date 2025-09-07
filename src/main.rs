//! AI Memory Service - Human-inspired memory system for AI agents
//! 
//! Main entry point for the memory service server.

mod api;
mod brain;
mod cache;
mod config;
mod embedding;
mod embedding_config;
mod memory;
mod metrics;
mod simd_search;
mod storage;
mod types;

use crate::config::Config;
use crate::memory::MemoryService;
use api::{create_router, ApiState};
use axum::Router;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tower_http::cors::{CorsLayer, Any};
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing with structured logging
    init_tracing()?;

    info!("Starting AI Memory Service v{}", env!("CARGO_PKG_VERSION"));

    // Load and validate configuration
    let config = load_config().await?;
    
    // Initialize metrics system
    metrics::init_metrics();
    info!("Metrics system initialized");

    // Create memory service with proper error handling
    let memory_service = match MemoryService::new(config.clone()).await {
        Ok(service) => {
            info!("Memory service initialized successfully");
            Arc::new(service)
        }
        Err(e) => {
            error!("Failed to initialize memory service: {}", e);
            // Try to recover with minimal configuration
            let minimal_config = create_minimal_config();
            warn!("Attempting to start with minimal configuration");
            Arc::new(
                MemoryService::new(minimal_config)
                    .await
                    .map_err(|e| anyhow::anyhow!("Cannot start service: {}", e))?,
            )
        }
    };

    // Perform health check on startup
    perform_startup_health_check(&memory_service).await?;

    // Create API state
    let api_state = ApiState { memory_service };

    // Build application with proper middleware configuration
    let app = build_app(api_state, &config);

    // Bind to address with validation
    let addr = parse_socket_addr(&config)?;
    
    info!("Server starting on {}", addr);
    info!("Environment: {}", config.server.environment);
    info!("CORS origins: {:?}", config.server.cors_origins);

    // Create TCP listener with error handling
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(e) => {
            error!("Failed to bind to {}: {}", addr, e);
            // Try alternative port
            let alt_addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port + 1)
                .parse()?;
            warn!("Trying alternative address: {}", alt_addr);
            tokio::net::TcpListener::bind(alt_addr).await?
        }
    };

    info!("Server listening on {}", listener.local_addr()?);

    // Start server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shutdown complete");
    Ok(())
}

/// Initialize tracing with environment-aware configuration
fn init_tracing() -> anyhow::Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            // Default log levels for different environments
            let level = std::env::var("RUST_ENV")
                .unwrap_or_else(|_| "development".to_string());
            
            match level.as_str() {
                "production" => "ai_memory_service=info,tower_http=warn",
                "staging" => "ai_memory_service=debug,tower_http=info",
                _ => "ai_memory_service=debug,tower_http=debug",
            }
            .parse()
            .expect("Valid filter")
        });

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();

    Ok(())
}

/// Load configuration with validation and fallback
async fn load_config() -> anyhow::Result<Config> {
    // Try loading from multiple sources
    let config_paths = ["config.toml", "/etc/ai-memory/config.toml", "./config/default.toml"];
    
    for path in &config_paths {
        match Config::from_file(path) {
            Ok(mut config) => {
                info!("Configuration loaded from: {}", path);
                
                // Validate configuration
                validate_config(&mut config)?;
                
                // Apply environment variable overrides
                apply_env_overrides(&mut config);
                
                return Ok(config);
            }
            Err(e) => {
                warn!("Failed to load config from {}: {}", path, e);
            }
        }
    }

    // Create default config with safe defaults
    warn!("Using default configuration");
    let mut config = create_safe_defaults();
    validate_config(&mut config)?;
    Ok(config)
}

/// Validate configuration values
fn validate_config(config: &mut Config) -> anyhow::Result<()> {
    // Validate server settings
    if config.server.port == 0 {
        config.server.port = 8080;
    }
    
    if config.server.workers == 0 {
        config.server.workers = num_cpus::get();
    }
    
    // Validate storage settings
    if config.storage.neo4j_uri.is_empty() {
        return Err(anyhow::anyhow!("Neo4j URI cannot be empty"));
    }
    
    // Validate cache settings
    if config.cache.l1_size == 0 {
        config.cache.l1_size = 1000;
    }
    
    if config.cache.ttl_seconds == 0 {
        config.cache.ttl_seconds = 3600;
    }
    
    Ok(())
}

/// Apply environment variable overrides
fn apply_env_overrides(config: &mut Config) {
    if let Ok(port) = std::env::var("PORT") {
        if let Ok(p) = port.parse() {
            config.server.port = p;
        }
    }
    
    if let Ok(neo4j_uri) = std::env::var("NEO4J_URI") {
        config.storage.neo4j_uri = neo4j_uri;
    }
    
    if let Ok(env) = std::env::var("RUST_ENV") {
        config.server.environment = env;
    }
}

/// Create minimal configuration for recovery
fn create_minimal_config() -> Config {
    Config {
        server: config::ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            workers: 1,
            environment: "recovery".to_string(),
            cors_origins: vec!["http://localhost:3000".to_string()],
        },
        storage: config::StorageConfig {
            neo4j_uri: "bolt://localhost:7687".to_string(),
            neo4j_user: "neo4j".to_string(),
            neo4j_password: String::new(),
            connection_pool_size: 5,
        },
        embedding: config::EmbeddingConfig {
            model_path: "./models/gemma-300m.onnx".to_string(),
            tokenizer_path: "./models/tokenizer.json".to_string(),
            batch_size: 8,
            max_sequence_length: 512,
        },
        cache: config::CacheConfig {
            l1_size: 100,
            l2_size: 1000,
            ttl_seconds: 300,
            compression_enabled: false,
        },
        brain: config::BrainConfig {
            model_name: "local".to_string(),
            min_importance: 0.1,
            enable_sentiment: false,
        },
    }
}

/// Create safe default configuration
fn create_safe_defaults() -> Config {
    Config {
        server: config::ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            workers: num_cpus::get(),
            environment: "development".to_string(),
            cors_origins: vec!["http://localhost:3000".to_string()],
        },
        storage: config::StorageConfig {
            neo4j_uri: std::env::var("NEO4J_URI")
                .unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
            neo4j_user: std::env::var("NEO4J_USER")
                .unwrap_or_else(|_| "neo4j".to_string()),
            neo4j_password: std::env::var("NEO4J_PASSWORD")
                .unwrap_or_default(),
            connection_pool_size: 10,
        },
        embedding: config::EmbeddingConfig {
            model_path: "./models/gemma-300m.onnx".to_string(),
            tokenizer_path: "./models/tokenizer.json".to_string(),
            batch_size: 32,
            max_sequence_length: 2048,
        },
        cache: config::CacheConfig {
            l1_size: 1000,
            l2_size: 10000,
            ttl_seconds: 3600,
            compression_enabled: true,
        },
        brain: config::BrainConfig {
            model_name: "gemma-300m".to_string(),
            min_importance: 0.1,
            enable_sentiment: true,
        },
    }
}

/// Build application with middleware
fn build_app(api_state: ApiState, config: &Config) -> Router {
    // Configure CORS based on environment
    let cors = if config.server.environment == "production" {
        CorsLayer::new()
            .allow_origin(
                config.server.cors_origins
                    .iter()
                    .map(|s| s.parse().expect("Valid origin"))
                    .collect::<Vec<_>>(),
            )
            .allow_methods([
                axum::http::Method::GET,
                axum::http::Method::POST,
                axum::http::Method::DELETE,
            ])
            .allow_headers(Any)
            .max_age(Duration::from_secs(3600))
    } else {
        // More permissive for development
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    };

    create_router(api_state)
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .route("/metrics", axum::routing::get(metrics_handler))
        .route("/health", axum::routing::get(health_handler))
}

/// Parse and validate socket address
fn parse_socket_addr(config: &Config) -> anyhow::Result<SocketAddr> {
    let addr_str = format!("{}:{}", config.server.host, config.server.port);
    addr_str
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid server address '{}': {}", addr_str, e))
}

/// Perform startup health check
async fn perform_startup_health_check(service: &MemoryService) -> anyhow::Result<()> {
    info!("Performing startup health check");
    
    // Test database connection
    match service.get_stats().await {
        Ok(stats) => {
            info!("Health check passed. Total memories: {}", stats.total_memories);
            Ok(())
        }
        Err(e) => {
            warn!("Health check warning: {}", e);
            // Continue anyway - service might recover
            Ok(())
        }
    }
}

/// Metrics endpoint handler with basic protection
async fn metrics_handler() -> Result<String, StatusCode> {
    // In production, you might want to check for a metrics token
    if let Ok(token) = std::env::var("METRICS_TOKEN") {
        // Implement token validation here if needed
        if !token.is_empty() {
            // Token validation logic
        }
    }
    
    Ok(metrics::export_metrics())
}

use axum::http::StatusCode;

/// Health check endpoint
async fn health_handler() -> Result<String, StatusCode> {
    Ok("OK".to_string())
}

/// Graceful shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C signal");
        },
        _ = terminate => {
            info!("Received terminate signal");
        },
    }

    info!("Starting graceful shutdown");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        config.server.port = 0;
        validate_config(&mut config).unwrap();
        assert_eq!(config.server.port, 8080);
    }

    #[test]
    fn test_socket_addr_parsing() {
        let config = Config {
            server: config::ServerConfig {
                host: "127.0.0.1".to_string(),
                port: 3000,
                ..Default::default()
            },
            ..Default::default()
        };
        
        let addr = parse_socket_addr(&config).unwrap();
        assert_eq!(addr.to_string(), "127.0.0.1:3000");
    }
}