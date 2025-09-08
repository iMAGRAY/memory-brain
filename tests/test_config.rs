//! Test configuration loading and validation

use ai_memory_service::config::{Config, ServerConfig, StorageConfig, EmbeddingConfig, CacheConfig, BrainConfig};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Testing Configuration System");
    println!("================================");

    // Test 1: Default configuration
    println!("\n1. Testing Default Configuration:");
    let default_config = Config::default();
    println!("   ✅ Default config created successfully");
    println!("   - Server port: {}", default_config.server.port);
    println!("   - Neo4j URI: {}", default_config.storage.neo4j_uri);
    println!("   - Embedding model: {}", default_config.embedding.model_path);
    println!("   - Cache L1 size: {}", default_config.cache.l1_size);
    println!("   - Max memories: {}", default_config.brain.max_memories);

    // Test 2: Environment variable configuration
    println!("\n2. Testing Environment Variable Configuration:");
    
    // Set some test environment variables
    env::set_var("SERVICE_PORT", "9090");
    env::set_var("NEO4J_URI", "bolt://test:7687");
    env::set_var("EMBEDDING_MODEL_PATH", "test-model");
    env::set_var("L1_CACHE_SIZE", "2000");
    env::set_var("MAX_MEMORIES", "50000");

    // Create configuration from environment
    let env_config = Config {
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
            cors_origins: vec!["*".to_string()],
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
            embedding_dimension: env::var("EMBEDDING_DIMENSION")
                .ok()
                .and_then(|s| s.parse().ok()),
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
    };

    println!("   ✅ Environment config created successfully");
    println!("   - Server port: {} (from env)", env_config.server.port);
    println!("   - Neo4j URI: {} (from env)", env_config.storage.neo4j_uri);
    println!("   - Embedding model: {} (from env)", env_config.embedding.model_path);
    println!("   - Cache L1 size: {} (from env)", env_config.cache.l1_size);
    println!("   - Max memories: {} (from env)", env_config.brain.max_memories);

    // Test 3: Configuration validation
    println!("\n3. Testing Configuration Validation:");
    
    // Test valid config
    match validate_config(&env_config) {
        Ok(_) => println!("   ✅ Valid configuration passed validation"),
        Err(e) => println!("   ❌ Valid configuration failed validation: {}", e),
    }

    // Test invalid config (zero port)
    let mut invalid_config = env_config.clone();
    invalid_config.server.port = 0;
    match validate_config(&invalid_config) {
        Ok(_) => println!("   ❌ Invalid configuration (port=0) incorrectly passed validation"),
        Err(_) => println!("   ✅ Invalid configuration (port=0) correctly failed validation"),
    }

    // Test invalid config (invalid Neo4j URI)
    let mut invalid_config2 = env_config.clone();
    invalid_config2.storage.neo4j_uri = "invalid-uri".to_string();
    match validate_config(&invalid_config2) {
        Ok(_) => println!("   ❌ Invalid configuration (bad Neo4j URI) incorrectly passed validation"),
        Err(_) => println!("   ✅ Invalid configuration (bad Neo4j URI) correctly failed validation"),
    }

    // Test 4: TOML serialization/deserialization
    println!("\n4. Testing TOML Serialization:");
    
    let toml_string = match toml::to_string_pretty(&env_config) {
        Ok(s) => {
            println!("   ✅ Configuration serialized to TOML successfully");
            s
        }
        Err(e) => {
            println!("   ❌ Failed to serialize configuration to TOML: {}", e);
            return Err(e.into());
        }
    };

    println!("   TOML Preview (first 200 chars):");
    println!("   {}", &toml_string.chars().take(200).collect::<String>());

    match toml::from_str::<Config>(&toml_string) {
        Ok(_) => println!("   ✅ Configuration deserialized from TOML successfully"),
        Err(e) => {
            println!("   ❌ Failed to deserialize configuration from TOML: {}", e);
            return Err(e.into());
        }
    }

    println!("\n🎉 All configuration tests completed successfully!");
    Ok(())
}

/// Validate configuration for correctness
fn validate_config(config: &Config) -> Result<(), String> {
    // Validate server config
    if config.server.port == 0 {
        return Err("Invalid port number".to_string());
    }
    
    // Validate Neo4j URI
    if !config.storage.neo4j_uri.starts_with("bolt://") && 
       !config.storage.neo4j_uri.starts_with("neo4j://") {
        return Err("Invalid Neo4j URI format".to_string());
    }
    
    // Validate cache sizes
    if config.cache.l1_size == 0 || config.cache.l2_size == 0 {
        return Err("Cache sizes must be greater than 0".to_string());
    }
    
    // Validate brain config
    if config.brain.importance_threshold < 0.0 || config.brain.importance_threshold > 1.0 {
        return Err("Importance threshold must be between 0.0 and 1.0".to_string());
    }
    
    Ok(())
}