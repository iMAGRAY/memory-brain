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
// tracing_subscriber is used via fully qualified path; no need to import as a single item
use anyhow::Result;
use tokio::signal;
use tokio::time::{sleep, Duration};
use tokio::net::TcpStream;

// serde_json и toml уже используются в проекте; новых внешних зависимостей не добавляем.
use serde_json::Value as Json;

/// Унифицированный парсер булевых флагов окружения
fn env_flag(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            matches!(s.as_str(), "1" | "true" | "t" | "yes" | "y")
        }
        Err(_) => default,
    }
}

/// Пытается определить размерность эмбеддинга с embedding-сервера (HTTP) и вернуть её.
/// Безопасно падает в None при любой ошибке. Не использует новые внешние зависимости.
fn autodetect_embedding_dimension_from_server() -> Option<usize> {
    let base = env::var("EMBEDDING_SERVER_URL").ok()?;
    let api_key = env::var("EMBEDDING_API_KEY").ok();

    // Поддерживаем только http:// без TLS, чтобы не тянуть зависимости на TLS.
    let base = base.trim();
    if !base.starts_with("http://") {
        warn!("Embedding autodetect: only plain http:// is supported (got '{}'), skipping autodetect", base);
        return None;
    }

    // Находим хост, порт, базовый путь
    let without_scheme = &base["http://".len()..];
    let (host_port, _path_prefix) = match without_scheme.split_once('/') {
        Some((hp, p)) => (hp, format!("/{}", p)),
        None => (without_scheme, String::from("")),
    };

    // IPv6 в квадратных скобках: [::1]:8000
    let (host, port) = if host_port.starts_with('[') {
        // Ищем закрывающую скобку
        if let Some(idx) = host_port.find(']') {
            let host_inner = &host_port[1..idx];
            let rest = &host_port[idx + 1..];
            let port = if let Some((_colon, p)) = rest.split_once(':') {
                p.parse::<u16>().unwrap_or(80)
            } else {
                80
            };
            (host_inner.to_string(), port)
        } else {
            (host_port.to_string(), 80)
        }
    } else if let Some((h, p)) = host_port.rsplit_once(':') {
        // h может тоже содержать двоеточия, но для простоты этого достаточно
        let port = p.parse::<u16>().unwrap_or(80);
        (h.to_string(), port)
    } else {
        (host_port.to_string(), 80)
    };

    // Кандидатные эндпоинты, где может быть размерность
    let candidates = [
        "/stats",      // contains default_dimension, matryoshka_dimensions
        "/dimensions", // hypothetical
        "/health",
        "/info",
        "/config",
        "/v1/health",
        "/v1/info",
    ];

    for path in candidates {
        // Открываем TCP и делаем простой HTTP/1.1 GET
        let addr = format!("{}:{}", host, port);
        let stream = std::net::TcpStream::connect_timeout(
            &addr.parse().ok()?,
            std::time::Duration::from_millis(700),
        );
        if stream.is_err() {
            continue;
        }
        let mut stream = stream.ok()?;
        stream
            .set_read_timeout(Some(std::time::Duration::from_millis(700)))
            .ok()?;
        stream
            .set_write_timeout(Some(std::time::Duration::from_millis(700)))
            .ok()?;

        let auth_header = api_key
            .as_ref()
            .map(|k| format!("Authorization: Bearer {}\r\n", k))
            .unwrap_or_default();

        let request = format!(
            "GET {} HTTP/1.1\r\nHost: {}\r\nAccept: application/json\r\n{}Connection: close\r\n\r\n",
            path, host, auth_header
        );

        use std::io::{Read, Write};
        if stream.write_all(request.as_bytes()).is_err() {
            continue;
        }

        let mut buf = String::new();
        if stream.read_to_string(&mut buf).is_err() {
            continue;
        }

        // Выделяем тело ответа
        let body = if let Some(idx) = buf.find("\r\n\r\n") {
            &buf[idx + 4..]
        } else {
            buf.as_str()
        };

        if let Ok(json) = serde_json::from_str::<Json>(body) {
            if let Some(n) = parse_embedding_dimension_from_json(&json) {
                return Some(n);
            }
        }
    }

    None
}

/// Выделяет размерность эмбеддинга из JSON, отдаваемого embedding-сервером (/stats, /info, и т.п.).
/// Поддерживает ключи: default_dimension, embedding_dimension, dimension, dim, embedding_dim, dims[], size,
/// а также пример: {"example_embedding":[..]}.
pub(crate) fn parse_embedding_dimension_from_json(json: &Json) -> Option<usize> {
    // Прямые числовые ключи
    let try_keys = [
        "default_dimension",
        "embedding_dimension",
        "dimension",
        "dim",
        "embedding_dim",
        "size",
    ];
    for k in try_keys {
        if let Some(v) = json.get(k) {
            if let Some(n) = v.as_u64() {
                return Some(n as usize);
            }
            if let Some(arr) = v.as_array() {
                if let Some(n) = arr.first().and_then(|x| x.as_u64()) {
                    return Some(n as usize);
                }
            }
        }
    }
    // Вариант, где ключ содержит массив поддерживаемых размерностей
    if let Some(arr) = json.get("matryoshka_dimensions").and_then(|v| v.as_array()) {
        // Берём первую (наибольшую) как дефолт — если список убывающий
        if let Some(n) = arr.first().and_then(|x| x.as_u64()) {
            return Some(n as usize);
        }
    }
    // Пример вектора
    if let Some(arr) = json.get("example_embedding").and_then(|v| v.as_array()) {
        return Some(arr.len());
    }
    None
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .init();

    // Initialize Prometheus metrics registry
    ai_memory_service::metrics::init_metrics();

    info!("🚀 Starting AI Memory Service with GPT-4 Orchestrator");

    // Load .env file if it exists
    if Path::new(".env").exists() {
        dotenv::dotenv().ok();
        info!("✅ Loaded .env configuration");
    }

    // Create configuration
    let mut config = load_configuration()?;

    // Автоопределение размерности эмбеддинга (безопасный фолбэк на конфиг, по умолчанию 512)
    if env_flag("EMBEDDING_AUTODETECT", true) {
        if let Some(dim) = autodetect_embedding_dimension_from_server() {
            if dim > 0 {
                info!("📐 Resolved embedding dimension from embedding server: {}", dim);
                config.embedding.embedding_dimension = Some(dim);
                ai_memory_service::metrics::set_embedding_dimension("autodetect", dim);
                ai_memory_service::metrics::record_autodetect_result("success");
            } else {
                warn!("📐 Embedding autodetect returned non-positive value, keeping config value {:?}", config.embedding.embedding_dimension);
                ai_memory_service::metrics::record_autodetect_result("failure");
            }
        } else {
            info!(
                "📐 Embedding autodetect skipped or failed, using config value {:?}",
                config.embedding.embedding_dimension
            );
            ai_memory_service::metrics::record_autodetect_result("failure");
            ai_memory_service::metrics::set_embedding_dimension(
                "config",
                config.embedding.embedding_dimension.unwrap_or(512),
            );
        }
    } else {
        info!(
            "📐 Embedding autodetect disabled, using config value {:?}",
            config.embedding.embedding_dimension
        );
        ai_memory_service::metrics::record_autodetect_result("disabled");
        ai_memory_service::metrics::set_embedding_dimension(
            "config",
            config.embedding.embedding_dimension.unwrap_or(512),
        );
    }
    
    // Initialize Memory Service
    info!("📦 Initializing Memory Service...");
    let memory_service = Arc::new(
        MemoryService::new(config.clone()).await?
    );
    info!("✅ Memory Service initialized (embedding_dimension={})",
        config.embedding.embedding_dimension.unwrap_or(512)
    );
    // Update metrics: embedding service availability and dimension source
    ai_memory_service::metrics::set_service_available("embedding", memory_service.embedding_available());

    // Флаг принудительного отключения оркестратора
    let orchestrator_force_disabled = env_flag("ORCHESTRATOR_FORCE_DISABLE", false);

    // Initialize GPT-5-nano Orchestrator (с учётом форс-отключения)
    let orchestrator = if orchestrator_force_disabled {
        info!("🧠 Orchestrator is force-disabled by ORCHESTRATOR_FORCE_DISABLE=true");
        None
    } else if let Ok(api_key) = env::var("OPENAI_API_KEY") {
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

    // Глобальный флаг для отключения всех планировщиков (включая дистилляцию)
    let disable_schedulers = env_flag("DISABLE_SCHEDULERS", false);

    // Initialize Memory Distillation Engine (если оркестратор доступен и планировщики не отключены)
    let distillation_engine = if orchestrator.is_some() && !disable_schedulers {
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
            orchestrator.as_ref().unwrap().clone(),
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
        if disable_schedulers {
            warn!("⚠️ All schedulers are disabled by DISABLE_SCHEDULERS=true");
        } else {
            warn!("⚠️ Memory Distillation Engine disabled (no orchestrator available)");
        }
        None
    };

    // Create API configuration (SERVICE_HOST / SERVICE_PORT используются в load_configuration)
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
    
    // Запускаем сервер REST API
    let memory_service_for_server = memory_service.clone();
    let orchestrator_for_server = orchestrator.clone();
    let api_config_for_server = api_config.clone();

    let api_handle = tokio::spawn(async move {
        if let Err(e) = run_server(memory_service_for_server, orchestrator_for_server, api_config_for_server).await {
            error!("API server error: {}", e);
        }
    });

    // Ожидаем реальный bind сокета и пишем явный лог «API bound on ...»
    let host_for_wait = api_config.host.clone();
    let port_for_wait = api_config.port;
    tokio::spawn(async move {
        // Если бинды происходят на 0.0.0.0 или ::, пробуем локалхосты
        let mut candidates: Vec<String> = vec![host_for_wait.clone()];
        if host_for_wait == "0.0.0.0" || host_for_wait == "127.0.0.1" || host_for_wait.is_empty() || host_for_wait == "*" {
            candidates.push("127.0.0.1".to_string());
            candidates.push("localhost".to_string());
        }
        if host_for_wait == "::" {
            candidates.push("::1".to_string());
            candidates.push("localhost".to_string());
            candidates.push("127.0.0.1".to_string());
        }

        let timeout_ms: u64 = env::var("WAIT_FOR_BIND_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30000);

        let start = tokio::time::Instant::now();
        let mut logged = false;

        while start.elapsed() < Duration::from_millis(timeout_ms) {
            for h in &candidates {
                if TcpStream::connect((h.as_str(), port_for_wait)).await.is_ok() {
                    info!("✅ API bound on {}:{}", host_for_wait, port_for_wait);
                    logged = true;
                    break;
                }
            }
            if logged {
                break;
            }
            sleep(Duration::from_millis(60)).await;
        }
        if !logged {
            warn!("⏱️ Could not confirm port bind within {} ms (server may still be starting)", timeout_ms);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_default_dimension_key() {
        let j: Json = serde_json::json!({"default_dimension": 512});
        assert_eq!(parse_embedding_dimension_from_json(&j), Some(512));
    }

    #[test]
    fn parse_matryoshka_dimensions_array() {
        let j: Json = serde_json::json!({"matryoshka_dimensions": [768,512,256,128]});
        assert_eq!(parse_embedding_dimension_from_json(&j), Some(768));
    }

    #[test]
    fn parse_example_embedding_len() {
        let j: Json = serde_json::json!({"example_embedding": [0.1,0.2,0.3,0.4]});
        assert_eq!(parse_embedding_dimension_from_json(&j), Some(4));
    }
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
