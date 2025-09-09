//! AI Memory Service - Main Entry Point
//! 
//! Advanced intelligent memory system with GPT-5-nano orchestration

use ai_memory_service::{
    MemoryService, MemoryOrchestrator, OrchestratorConfig,
    api::{run_server, ApiConfig},
    config::{Config, ServerConfig, StorageConfig, EmbeddingConfig, CacheConfig, BrainConfig},
};
use std::sync::Arc;
use std::env;
use std::path::Path;
use tracing::{info, warn, error};
use tracing_subscriber;
use anyhow::Result;
use tokio::signal;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .init();

    info!("🚀 Starting AI Memory Service with GPT-4 Orchestrator");

    // Load .env file if it exists - with enhanced diagnostics
    if Path::new(".env").exists() {
        match dotenv::dotenv() {
            Ok(_) => {
                info!("✅ Loaded .env configuration");
                
                // Secure diagnostics - only check presence without logging values
                let env_vars = ["NEO4J_PASSWORD", "NEO4J_URI", "NEO4J_USER", "SERVICE_HOST", "SERVICE_PORT"];
                let mut found_count = 0;
                for var_name in &env_vars {
                    if env::var(var_name).is_ok() {
                        found_count += 1;
                        info!("🔍 Debug: {} present in environment", var_name);
                    } else {
                        warn!("🔍 Debug: {} not found in environment", var_name);
                    }
                }
                info!("🔍 Environment variables loaded: {}/{}", found_count, env_vars.len());
            },
            Err(e) => warn!("⚠️ Failed to load .env: {}", e),
        }
    } else {
        warn!("⚠️ .env file not found at current directory");
        
        // Force load from system environment as fallback
        info!("🔄 Attempting to use system environment variables");
    }

    // Create configuration
    let config = load_configuration()?;
    
    // Initialize Memory Service
    info!("📦 Initializing Memory Service...");
    let memory_service = Arc::new(
        MemoryService::new(config.clone()).await?
    );
    info!("✅ Memory Service initialized");

    // Initialize GPT-4 Orchestrator if API key is available
    let orchestrator = if let Ok(api_key) = env::var("OPENAI_API_KEY") {
        if api_key.starts_with("sk-") {
            info!("🧠 Initializing GPT-4 Orchestrator...");
            let orch_config = OrchestratorConfig {
                api_key: api_key.clone(),
                model: env::var("ORCHESTRATOR_MODEL").unwrap_or_else(|_| "gpt-4-turbo-preview".to_string()),
                max_input_tokens: env::var("MAX_INPUT_TOKENS")
                    .unwrap_or_else(|_| "128000".to_string())
                    .parse()
                    .unwrap_or_else(|e| {
                        warn!("Invalid MAX_INPUT_TOKENS: {}, using default 128000", e);
                        128000
                    }),
                max_output_tokens: env::var("MAX_OUTPUT_TOKENS")
                    .unwrap_or_else(|_| "4096".to_string())
                    .parse()
                    .unwrap_or_else(|e| {
                        warn!("Invalid MAX_OUTPUT_TOKENS: {}, using default 4096", e);
                        4096
                    }),
                temperature: env::var("ORCHESTRATOR_TEMPERATURE")
                    .unwrap_or_else(|_| "0.7".to_string())
                    .parse()
                    .unwrap_or_else(|e| {
                        warn!("Invalid ORCHESTRATOR_TEMPERATURE: {}, using default 0.7", e);
                        0.7
                    }),
                timeout_seconds: env::var("ORCHESTRATOR_TIMEOUT")
                    .unwrap_or_else(|_| "120".to_string())
                    .parse()
                    .unwrap_or_else(|e| {
                        warn!("Invalid ORCHESTRATOR_TIMEOUT: {}, using default 120", e);
                        120
                    }),
            };
            
            match MemoryOrchestrator::new(orch_config) {
                Ok(orch) => {
                    info!("✅ GPT-4 Orchestrator ready");
                    Some(Arc::new(orch))
                }
                Err(e) => {
                    error!("❌ Failed to initialize Orchestrator: {}", e);
                    None
                }
            }
        } else {
            warn!("⚠️ Invalid OPENAI_API_KEY format (should start with 'sk-')");
            None
        }
    } else {
        warn!("⚠️ OPENAI_API_KEY not set, running without orchestrator");
        None
    };

    // Create API configuration
    let api_config = ApiConfig {
        host: config.server.host.clone(),
        port: config.server.port,
        cors_enabled: true,
        request_body_limit: 10 * 1024 * 1024, // 10MB
        enable_tracing: true,
    };

    // Start the REST API server
    info!("🌐 Starting REST API server on {}:{}", api_config.host, api_config.port);
    
    // Clone for the shutdown handler
    let shutdown_service = memory_service.clone();
    
    // Spawn the API server
    let api_handle = tokio::spawn(async move {
        if let Err(e) = run_server(memory_service, orchestrator, api_config).await {
            error!("API server error: {}", e);
        }
    });

    // Setup graceful shutdown
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("💤 Received Ctrl+C, shutting down...");
        },
        _ = terminate => {
            info!("💤 Received terminate signal, shutting down...");
        },
    }

    // Graceful shutdown
    info!("🔄 Performing graceful shutdown...");
    
    // Abort the API server
    api_handle.abort();
    
    // Shutdown memory service
    if let Err(e) = shutdown_service.shutdown().await {
        error!("Error during shutdown: {}", e);
    }
    
    info!("👋 AI Memory Service stopped");
    Ok(())
}

/// Load configuration from environment and/or config file
fn load_configuration() -> Result<Config> {
    info!("🔍 Starting configuration loading process...");
    
    // Check if config file exists
    let config_path = env::var("CONFIG_FILE").unwrap_or_else(|_| "config.toml".to_string());
    info!("🔍 Config file path: {}", config_path);
    
    let mut config = if Path::new(&config_path).exists() {
        info!("📄 Loading configuration from {}", config_path);
        let config_str = std::fs::read_to_string(&config_path)?;
        toml::from_str(&config_str)?
    } else {
        info!("🔧 Using default configuration with environment overrides");
        
        // Build configuration from environment variables
        Config {
            server: ServerConfig {
                host: env::var("SERVICE_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
                port: env::var("SERVICE_PORT")
                    .unwrap_or_else(|_| "8080".to_string())
                    .parse()
                    .unwrap_or(8080),
                workers: env::var("WORKERS")
                    .unwrap_or_else(|_| "4".to_string())
                    .parse()
                    .unwrap_or(4),
                environment: env::var("ENVIRONMENT").unwrap_or_else(|_| "development".to_string()),
                cors_origins: env::var("CORS_ORIGINS")
                    .unwrap_or_else(|_| "*".to_string())
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
            },
            storage: StorageConfig {
                neo4j_uri: env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
                neo4j_user: env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string()),
                neo4j_password: env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".to_string()),
                connection_pool_size: env::var("NEO4J_POOL_SIZE")
                    .unwrap_or_else(|_| "10".to_string())
                    .parse()
                    .unwrap_or(10),
            },
            embedding: EmbeddingConfig {
                model_path: env::var("EMBEDDING_MODEL_PATH")
                    .unwrap_or_else(|_| "onnx-community/embeddinggemma-300m-ONNX".to_string()),
                tokenizer_path: env::var("TOKENIZER_PATH")
                    .unwrap_or_else(|_| "tokenizer".to_string()),
                batch_size: env::var("EMBEDDING_BATCH_SIZE")
                    .unwrap_or_else(|_| "32".to_string())
                    .parse()
                    .unwrap_or(32),
                max_sequence_length: env::var("MAX_SEQUENCE_LENGTH")
                    .unwrap_or_else(|_| "2048".to_string())
                    .parse()
                    .unwrap_or(2048),
            },
            cache: CacheConfig {
                l1_size: env::var("L1_CACHE_SIZE")
                    .unwrap_or_else(|_| "1000".to_string())
                    .parse()
                    .unwrap_or(1000),
                l2_size: env::var("L2_CACHE_SIZE")
                    .unwrap_or_else(|_| "10000".to_string())
                    .parse()
                    .unwrap_or(10000),
                ttl_seconds: env::var("CACHE_TTL")
                    .unwrap_or_else(|_| "3600".to_string())
                    .parse()
                    .unwrap_or(3600),
                compression_enabled: env::var("CACHE_COMPRESSION")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()
                    .unwrap_or(true),
            },
            brain: BrainConfig {
                max_memories: env::var("MAX_MEMORIES")
                    .unwrap_or_else(|_| "100000".to_string())
                    .parse()
                    .unwrap_or(100000),
                importance_threshold: env::var("IMPORTANCE_THRESHOLD")
                    .unwrap_or_else(|_| "0.3".to_string())
                    .parse()
                    .unwrap_or(0.3),
                consolidation_interval: env::var("CONSOLIDATION_INTERVAL")
                    .unwrap_or_else(|_| "300".to_string())
                    .parse()
                    .unwrap_or(300),
                decay_rate: env::var("MEMORY_DECAY_RATE")
                    .unwrap_or_else(|_| "0.01".to_string())
                    .parse()
                    .unwrap_or(0.01),
            },
        }
    };

    info!("🔍 Starting environment variable override process...");
    
    // Apply environment variable overrides to config loaded from file
    // Neo4j URI with validation
    if let Ok(neo4j_uri) = env::var("NEO4J_URI") {
        info!("🔍 Found NEO4J_URI in environment");
        config.storage.neo4j_uri = neo4j_uri;
        info!("🔐 Applied NEO4J_URI from environment");
    } else {
        info!("🔍 NEO4J_URI not found in environment");
    }
    
    // Neo4j user
    if let Ok(neo4j_user) = env::var("NEO4J_USER") {
        config.storage.neo4j_user = neo4j_user;
        info!("🔐 Applied NEO4J_USER from environment: {}", config.storage.neo4j_user);
    }
    
    // Neo4j password with proper validation
    if let Ok(neo4j_password) = env::var("NEO4J_PASSWORD") {
        if !neo4j_password.is_empty() {
            config.storage.neo4j_password = neo4j_password;
            info!("🔐 Applied NEO4J_PASSWORD from environment (length: {})", config.storage.neo4j_password.len());
        } else {
            warn!("⚠️ NEO4J_PASSWORD is empty, using config file value");
        }
    } else {
        warn!("⚠️ NEO4J_PASSWORD not found in environment, using config file value");
    }
    
    // Final validation - ensure password is not empty
    if config.storage.neo4j_password.is_empty() {
        return Err(anyhow::anyhow!(
            "Neo4j password is empty. Please set NEO4J_PASSWORD environment variable or configure in config.toml"
        ));
    }
    
    // Apply all other environment overrides
    if let Ok(host) = env::var("SERVICE_HOST") {
        config.server.host = host;
        info!("🔐 Applied SERVICE_HOST from environment: {}", config.server.host);
    }
    
    if let Ok(port) = env::var("SERVICE_PORT") {
        if let Ok(port_num) = port.parse::<u16>() {
            config.server.port = port_num;
            info!("🔐 Applied SERVICE_PORT from environment: {}", config.server.port);
        }
    }

    // Validate configuration
    validate_config(&config)?;
    
    Ok(config)
}

/// Validate configuration for correctness
fn validate_config(config: &Config) -> Result<()> {
    // Validate server config
    if config.server.port == 0 {
        return Err(anyhow::anyhow!("Invalid port number"));
    }
    
    // Validate Neo4j URI
    if !config.storage.neo4j_uri.starts_with("bolt://") && 
       !config.storage.neo4j_uri.starts_with("neo4j://") {
        return Err(anyhow::anyhow!("Invalid Neo4j URI format"));
    }
    
    // Validate cache sizes
    if config.cache.l1_size == 0 || config.cache.l2_size == 0 {
        return Err(anyhow::anyhow!("Cache sizes must be greater than 0"));
    }
    
    // Validate brain config
    if config.brain.importance_threshold < 0.0 || config.brain.importance_threshold > 1.0 {
        return Err(anyhow::anyhow!("Importance threshold must be between 0.0 and 1.0"));
    }
    
    Ok(())
}