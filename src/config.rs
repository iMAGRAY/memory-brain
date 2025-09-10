//! Configuration module for AI Memory Service
//!
//! Provides configuration structures for all subsystems.

use serde::{Deserialize, Serialize};
use std::env;
use std::path::Path;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub server: ServerConfig,
    pub storage: StorageConfig,
    pub embedding: EmbeddingConfig,
    pub cache: CacheConfig,
    pub brain: BrainConfig,
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub workers: usize,
    pub environment: String,
    pub cors_origins: Vec<String>,
}

/// Storage configuration for Neo4j
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub neo4j_uri: String,
    pub neo4j_user: String,
    pub neo4j_password: String,
    pub connection_pool_size: usize,
}

/// Embedding service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model_path: String,
    pub tokenizer_path: String,
    pub batch_size: usize,
    pub max_sequence_length: usize,
    /// Embedding dimension for Matryoshka truncation (128, 256, 512, 768)
    pub embedding_dimension: Option<usize>,
    /// Whether to normalize embeddings (critical for Matryoshka truncation)
    pub normalize_embeddings: bool,
    /// Precision: float32 (recommended), bfloat16 (modern GPU), NOT float16
    pub precision: String,
    /// Use specialized prompts for better quality
    pub use_specialized_prompts: bool,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub l1_size: usize,
    pub l2_size: usize,
    pub ttl_seconds: u64,
    pub compression_enabled: bool,
}

/// AI Brain configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainConfig {
    pub max_memories: usize,
    pub importance_threshold: f32,
    pub consolidation_interval: u64,
    pub decay_rate: f32,
}

#[allow(clippy::derivable_impls)]
impl Default for Config {
    fn default() -> Self {
        Self {
            server: ServerConfig::default(),
            storage: StorageConfig::default(),
            embedding: EmbeddingConfig::default(),
            cache: CacheConfig::default(),
            brain: BrainConfig::default(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            workers: num_cpus::get(),
            environment: "development".to_string(),
            cors_origins: vec!["http://localhost:3000".to_string()],
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            neo4j_uri: "bolt://localhost:7687".to_string(),
            neo4j_user: "neo4j".to_string(),
            neo4j_password: String::new(),
            connection_pool_size: 10,
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
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
            // Optimal settings according to EmbeddingGemma documentation
            embedding_dimension: env::var("EMBEDDING_DIMENSION")
                .ok()
                .and_then(|s| s.parse().ok())
                .or(Some(512)), // Optimal balance quality/speed
            normalize_embeddings: env::var("NORMALIZE_EMBEDDINGS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(true), // Critical for Matryoshka truncation
            precision: env::var("EMBEDDING_PRECISION")
                .unwrap_or_else(|_| "float32".to_string()), // NEVER use float16 - not supported!
            use_specialized_prompts: env::var("USE_SPECIALIZED_PROMPTS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(true), // Critical for quality
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_size: 1000,
            l2_size: 10000,
            ttl_seconds: 3600,
            compression_enabled: true,
        }
    }
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            max_memories: 100000,
            importance_threshold: 0.3,
            consolidation_interval: 300,
            decay_rate: 0.01,
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn from_file(path: &str) -> Result<Self, ConfigError> {
        if !Path::new(path).exists() {
            return Err(ConfigError::FileNotFound(path.to_string()));
        }

        let contents = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::IoError(e.to_string()))?;

        toml::from_str(&contents)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    /// Save configuration to file
    pub fn to_file(&self, path: &str) -> Result<(), ConfigError> {
        let contents = toml::to_string_pretty(self)
            .map_err(|e| ConfigError::SerializeError(e.to_string()))?;

        std::fs::write(path, contents)
            .map_err(|e| ConfigError::IoError(e.to_string()))
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate server config
        if self.server.port == 0 {
            return Err(ConfigError::ValidationError("Server port cannot be 0".to_string()));
        }

        // Validate storage config
        if self.storage.neo4j_uri.is_empty() {
            return Err(ConfigError::ValidationError("Neo4j URI cannot be empty".to_string()));
        }

        // Validate embedding config
        if self.embedding.batch_size == 0 {
            return Err(ConfigError::ValidationError("Batch size cannot be 0".to_string()));
        }
        
        // Validate model and tokenizer file paths (skip in unit-test builds)
        if !cfg!(test) {
            Self::validate_file_path(&self.embedding.model_path, "Model")?;
            Self::validate_file_path(&self.embedding.tokenizer_path, "Tokenizer")?;
        }

        if self.embedding.max_sequence_length == 0 {
            return Err(ConfigError::ValidationError("Max sequence length cannot be 0".to_string()));
        }
        
        // Validate new EmbeddingGemma-specific fields
        if let Some(dim) = self.embedding.embedding_dimension {
            if ![128, 256, 512, 768].contains(&dim) {
                return Err(ConfigError::ValidationError(
                    format!("Invalid embedding dimension {}. Must be one of: 128, 256, 512, 768", dim)
                ));
            }
        }
        
        // Validate precision setting
        if !["float32", "bfloat16"].contains(&self.embedding.precision.as_str()) {
            return Err(ConfigError::ValidationError(
                format!("Unsupported precision '{}'. Must be 'float32' or 'bfloat16' (NOT float16)", self.embedding.precision)
            ));
        }

        // Validate cache config
        if self.cache.l1_size == 0 {
            return Err(ConfigError::ValidationError("L1 cache size cannot be 0".to_string()));
        }

        Ok(())
    }
    
    /// Helper function to validate file paths
    fn validate_file_path(path_str: &str, file_type: &str) -> Result<(), ConfigError> {
        // Check for empty path
        if path_str.is_empty() {
            return Err(ConfigError::ValidationError(
                format!("{} path cannot be empty", file_type)
            ));
        }
        
        let path = std::path::PathBuf::from(path_str);
        
        // Check if file exists
        if !path.exists() {
            return Err(ConfigError::ValidationError(
                format!("{} file not found: {}", file_type, path_str)
            ));
        }
        
        // Check if path points to a file (not directory)
        if !path.is_file() {
            return Err(ConfigError::ValidationError(
                format!("{} path is not a file: {}", file_type, path_str)
            ));
        }
        
        // Check if file is readable
        match std::fs::File::open(&path) {
            Ok(_) => {},
            Err(e) => {
                return Err(ConfigError::ValidationError(
                    format!("{} file is not readable: {} - {}", file_type, path_str, e)
                ));
            }
        }
        
        Ok(())
    }
}

/// Configuration error types
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Configuration file not found: {0}")]
    FileNotFound(String),
    
    #[error("IO error: {0}")]
    IoError(String),
    
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("Serialize error: {0}")]
    SerializeError(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.cache.l1_size, 1000);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        assert!(config.validate().is_ok());

        config.server.port = 0;
        assert!(config.validate().is_err());
    }
}
