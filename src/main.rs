//! AI Memory Service - Main Entry Point
//! 
//! Advanced intelligent memory system with GPT-4 orchestration

use ai_memory_service::{
    MemoryService, MemoryOrchestrator, OrchestratorConfig,
    api::{run_server, ApiConfig},
    config::{Config, ServerConfig, StorageConfig, EmbeddingConfig, CacheConfig, BrainConfig},
    distillation::{MemoryDistillationEngine, DistillationConfig},
    ShutdownManager,
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

    // Load .env file if it exists
    if Path::new(".env").exists() {
        dotenv::dotenv().ok();
        info!("✅ Loaded .env configuration");
    }

    // Create configuration
    let config = load_configuration()?;
    
    // Initialize Memory Service
    info!("📦 Initializing Memory Service...");
    let memory_service = Arc::new(
        MemoryService::new(config.clone()).await?
    );
    info!("✅ Memory Service initialized");

    // Initialize GPT-5-nano Orchestrator
    let orchestrator = if let Ok(api_key) = env::var("OPENAI_API_KEY") {
        if api_key.starts_with("sk-") {
            info!("🧠 Initializing GPT-5-nano Orchestrator...");
            let orch_config = OrchestratorConfig {
                api_key: api_key.clone(),
                model: env::var("ORCHESTRATOR_MODEL").unwrap_or_else(|_| "gpt-5-nano".to_string()),
                max_input_tokens: env::var("MAX_INPUT_TOKENS")
                    .unwrap_or_else(|_| "400000".to_string())
                    .parse()
                    .unwrap_or_else(|e| {
                        warn!("Invalid MAX_INPUT_TOKENS: {}, using default 400000", e);
                        400000
                    }),
                max_output_tokens: env::var("MAX_OUTPUT_TOKENS")
                    .unwrap_or_else(|_| "12000".to_string())
                    .parse()
                    .unwrap_or_else(|e| {
                        warn!("Invalid MAX_OUTPUT_TOKENS: {}, using default 12000", e);
                        12000
                    }),
                temperature: env::var("ORCHESTRATOR_TEMPERATURE")
                    .unwrap_or_else(|_| "1.0".to_string())
                    .parse()
                    .unwrap_or_else(|e| {
                        warn!("Invalid ORCHESTRATOR_TEMPERATURE: {}, using default 1.0", e);
                        1.0
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
                    info!("✅ GPT-5-nano Orchestrator ready");
                    Some(Arc::new(orch))
                }
                Err(e) => {
                    error!("❌ Failed to initialize GPT-5-nano Orchestrator: {}", e);
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

    // Initialize Memory Distillation Engine (if orchestrator is available)
    let distillation_engine = if let Some(ref orch) = orchestrator {
        info!("🧠 Initializing Autonomous Memory Distillation Engine...");
        
        let distillation_config = DistillationConfig {
            enabled: env::var("DISTILLATION_ENABLED")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            daily_hour: env::var("DISTILLATION_DAILY_HOUR")
                .unwrap_or_else(|_| "2".to_string())
                .parse()
                .unwrap_or(2),
            weekly_day: env::var("DISTILLATION_WEEKLY_DAY")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .unwrap_or(0),
            monthly_day: env::var("DISTILLATION_MONTHLY_DAY")
                .unwrap_or_else(|_| "1".to_string())
                .parse()
                .unwrap_or(1),
            min_importance_threshold: env::var("DISTILLATION_MIN_IMPORTANCE")
                .unwrap_or_else(|_| "0.3".to_string())
                .parse()
                .unwrap_or(0.3),
            max_memories_per_batch: env::var("DISTILLATION_MAX_BATCH_SIZE")
                .unwrap_or_else(|_| "1000".to_string())
                .parse()
                .unwrap_or(1000),
            operation_timeout_minutes: env::var("DISTILLATION_TIMEOUT_MINUTES")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .unwrap_or(30),
        };
        
        let engine = MemoryDistillationEngine::new(
            orch.clone(),
            memory_service.clone(),
            distillation_config,
        );
        
        // Start autonomous distillation
        if let Err(e) = engine.start_autonomous_distillation().await {
            error!("Failed to start distillation engine: {}", e);
            None
        } else {
            info!("✅ Autonomous Memory Distillation Engine started");
            Some(Arc::new(engine))
        }
    } else {
        warn!("⚠️ Memory Distillation Engine disabled (no orchestrator available)");
        None
    };

    // Create API configuration
    let api_config = ApiConfig {
        host: config.server.host.clone(),
        port: config.server.port,
        max_body_size: 10 * 1024 * 1024, // 10MB
        enable_cors: true,
        enable_compression: true,
        enable_tracing: true,
    };

    // Start the REST API server
    info!("🌐 Starting REST API server on {}:{}", api_config.host, api_config.port);
    
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
    
    // Shutdown distillation engine first
    if let Some(ref engine) = distillation_engine {
        info!("🧠 Shutting down Memory Distillation Engine...");
        if let Err(e) = engine.shutdown().await {
            error!("Error shutting down distillation engine: {}", e);
        } else {
            info!("✅ Memory Distillation Engine shutdown complete");
        }
    }
    
    // Abort the API server
    api_handle.abort();
    
    // Perform graceful shutdown using ShutdownManager
    if let Err(e) = ShutdownManager::shutdown().await {
        error!("Error during shutdown: {}", e);
    }
    
    info!("👋 AI Memory Service stopped");
    Ok(())
}

/// Load configuration from environment and/or config file
fn load_configuration() -> Result<Config> {
    // Check if config file exists
    let config_path = env::var("CONFIG_FILE").unwrap_or_else(|_| "config.toml".to_string());
    
    let config = if Path::new(&config_path).exists() {
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
                    .unwrap_or_else(|_| "./models/embeddinggemma-300m".to_string()),
                tokenizer_path: env::var("TOKENIZER_PATH")
                    .unwrap_or_else(|_| "./models/embeddinggemma-300m/tokenizer.json".to_string()),
                batch_size: env::var("EMBEDDING_BATCH_SIZE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(32),
                max_sequence_length: env::var("MAX_SEQUENCE_LENGTH")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2048),
                // New EmbeddingGemma-specific fields
                embedding_dimension: env::var("EMBEDDING_DIMENSION")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .or(Some(512)),
                normalize_embeddings: env::var("NORMALIZE_EMBEDDINGS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(true),
                precision: env::var("EMBEDDING_PRECISION")
                    .unwrap_or_else(|_| "float32".to_string()),
                use_specialized_prompts: env::var("USE_SPECIALIZED_PROMPTS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(true),
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