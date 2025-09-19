//! Configuration module for EmbeddingGemma model integration
//! Provides cross-platform model path resolution and configuration management

use crate::types::{MemoryError, MemoryResult};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tracing::{debug, info, warn};

// Cache for home directory to avoid repeated lookups
static HOME_DIR_CACHE: OnceLock<Option<PathBuf>> = OnceLock::new();

fn get_home_dir() -> Option<PathBuf> {
    HOME_DIR_CACHE.get_or_init(dirs::home_dir).clone()
}

/// EmbeddingGemma model configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingConfig {
    pub model: ModelConfig,
    pub embedding: EmbeddingSettings,
    pub prompts: PromptTemplates,
    pub performance: PerformanceConfig,
    pub cache: CacheConfig,
    pub validation: ValidationConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    pub name: String,
    pub huggingface_id: String,
    pub version: String,
    pub local_search_paths: Vec<String>,
    pub required_files: Vec<String>,
    pub min_model_size: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingSettings {
    pub default_dimension: usize,
    pub matryoshka_dimensions: Vec<usize>,
    pub max_sequence_length: usize,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptTemplates {
    pub query: String,
    pub document: String,
    pub classification: String,
    pub similarity: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PerformanceConfig {
    pub default_device: String,
    pub torch_dtype: String,
    pub default_batch_size: usize,
    pub max_batch_size: usize,
    pub embedding_timeout: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CacheConfig {
    pub enabled: bool,
    pub ttl_seconds: u64,
    pub max_entries: usize,
    pub eviction_strategy: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ValidationConfig {
    pub max_text_length: usize,
    pub max_total_batch_chars: usize,
    pub sanitize_input: bool,
    pub allow_empty_text: bool,
}

impl EmbeddingConfig {
    /// Load configuration from file or use defaults
    pub fn load() -> MemoryResult<Self> {
        // Try to load from config file
        let config_paths = vec![
            PathBuf::from("config/embeddinggemma.toml"),
            PathBuf::from("../config/embeddinggemma.toml"),
            PathBuf::from("/etc/ai-memory-service/embeddinggemma.toml"),
        ];
        
        for path in config_paths {
            if path.exists() {
                match std::fs::read_to_string(&path) {
                    Ok(content) => {
                        match toml::from_str::<EmbeddingConfig>(&content) {
                            Ok(config) => {
                                info!("Loaded EmbeddingGemma config from: {}", path.display());
                                return Ok(config);
                            }
                            Err(e) => {
                                warn!("Failed to parse config file {}: {}", path.display(), e);
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to read config file {}: {}", path.display(), e);
                    }
                }
            }
        }
        
        // Return default configuration if no file found
        Ok(Self::default())
    }
    
    /// Resolve model path with cross-platform support and security validation
    pub fn resolve_model_path(&self, model_name: &str) -> MemoryResult<String> {
        // Security: Comprehensive path traversal prevention
        if self.is_path_traversal_attempt(model_name) {
            return Err(MemoryError::Embedding(
                "Invalid model name: potential path traversal detected".to_string()
            ));
        }
        
        // Check if this is EmbeddingGemma model
        if model_name.contains("embeddinggemma") || 
           model_name == self.model.name || 
           model_name == self.model.huggingface_id {
            
            // Priority 1: Environment variable
            if let Ok(custom_path) = std::env::var("EMBEDDINGGEMMA_MODEL_PATH") {
                let path = Path::new(&custom_path);
                if self.validate_model_directory(path)? {
                    info!("Using EmbeddingGemma from EMBEDDINGGEMMA_MODEL_PATH: {}", custom_path);
                    return Ok(custom_path);
                } else {
                    warn!("EMBEDDINGGEMMA_MODEL_PATH set but invalid: {}", custom_path);
                }
            }
            
            // Priority 2: Search local paths
            let search_paths = self.build_search_paths();
            
            for path in search_paths {
                if self.validate_model_directory(&path)? {
                    info!("Found local EmbeddingGemma-300m model at: {}", path.display());
                    return Ok(path.to_string_lossy().to_string());
                }
            }
            
            // Priority 3: Fallback to HuggingFace
            info!("No local EmbeddingGemma model found, using HuggingFace: {}", self.model.huggingface_id);
            return Ok(self.model.huggingface_id.clone());
        }
        
        // For other models, return as-is after validation
        if model_name.is_empty() || model_name.len() > 256 {
            return Err(MemoryError::Embedding(
                "Model name must be non-empty and less than 256 characters".to_string()
            ));
        }
        
        Ok(model_name.to_string())
    }
    
    /// Build search paths for local model
    fn build_search_paths(&self) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        
        // Add configured paths
        for path_str in &self.model.local_search_paths {
            let path = if let Some(stripped) = path_str.strip_prefix("~/") {
                // Expand home directory using cached value
                if let Some(home) = get_home_dir() {
                    home.join(stripped)
                } else {
                    continue; // Skip invalid path
                }
            } else if path_str.starts_with("./") || path_str.starts_with("../") {
                // Relative paths
                if let Ok(current_dir) = std::env::current_dir() {
                    current_dir.join(path_str)
                } else {
                    continue; // Skip if can't get current dir
                }
            } else {
                PathBuf::from(path_str)
            };
            
            paths.push(path);
        }
        
        // Add dynamic paths based on current context (with error handling)
        if let Ok(current_dir) = std::env::current_dir() {
            paths.push(current_dir.join("models").join(&self.model.name));
        }
        
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(parent) = exe_path.parent() {
                paths.push(parent.join("models").join(&self.model.name));
            }
        }
        
        // Normalize and validate paths
        paths.into_iter()
            .filter_map(|p| self.normalize_path(&p).ok())
            .collect()
    }
    
    /// Validate that a directory contains required model files
    pub fn validate_model_directory(&self, path: &Path) -> MemoryResult<bool> {
        if !path.exists() || !path.is_dir() {
            return Ok(false);
        }
        
        // Check for required model files
        for required_file in &self.model.required_files {
            let file_path = path.join(required_file);
            if !file_path.exists() || !file_path.is_file() {
                debug!("Missing required model file: {}", file_path.display());
                return Ok(false);
            }
        }
        
        // Check model file size
        let model_file = path.join("model.safetensors");
        if let Ok(metadata) = std::fs::metadata(&model_file) {
            if metadata.len() < self.model.min_model_size {
                debug!(
                    "Model file too small: {} bytes (minimum: {} bytes)", 
                    metadata.len(), 
                    self.model.min_model_size
                );
                return Ok(false);
            }
        } else {
            return Ok(false);
        }
        
        info!("Validated model directory: {}", path.display());
        Ok(true)
    }
    
    /// Get device configuration (CPU/GPU)
    pub fn get_device(&self) -> String {
        std::env::var("EMBEDDING_DEVICE")
            .unwrap_or_else(|_| self.performance.default_device.clone())
    }
    
    /// Get torch dtype for model loading
    pub fn get_torch_dtype(&self) -> &str {
        &self.performance.torch_dtype
    }
    
    /// Check if a path contains traversal attempts (cross-platform)
    /// 
    /// This function validates paths to prevent directory traversal attacks
    /// by checking for various malicious patterns across different platforms.
    fn is_path_traversal_attempt(&self, path: &str) -> bool {
        // Decode potential URL-encoded characters first
        let decoded = path.replace("%2e", ".").replace("%2E", ".");
        let check_path = decoded.as_str();
        
        // Check for various path traversal patterns (deduplicated)
        let dangerous_patterns = [
            "..",       // Parent directory reference
            "~",        // Home directory expansion
            "%",        // Environment variable (remaining after decode)
            "$",        // Shell variable
            "\\UNC\\",  // Windows UNC path
            "\\\\?\\",   // Windows extended path
        ];
        
        // Check for absolute paths (Unix and Windows)
        if check_path.starts_with('/') || 
           check_path.starts_with('\\') ||
           check_path.starts_with("\\\\?") {
            return true;
        }
        
        // Check Windows drive letters (e.g., C:, D:)
        if check_path.len() >= 2 {
            let chars: Vec<char> = check_path.chars().take(2).collect();
            if chars.len() == 2 && chars[0].is_ascii_alphabetic() && chars[1] == ':' {
                return true;
            }
        }
        
        // Check for dangerous patterns
        dangerous_patterns.iter().any(|pattern| check_path.contains(pattern))
    }
    
    /// Normalize and validate a path for security
    /// 
    /// This function attempts to canonicalize paths when they exist,
    /// or normalizes path components when they don't, preventing traversal attacks.
    /// 
    /// # Returns
    /// - `Ok(PathBuf)` with normalized path if valid
    /// - `Err` if path contains traversal attempts
    fn normalize_path(&self, path: &Path) -> MemoryResult<PathBuf> {
        // Security pre-check: reject leading parent traversals like ../../.. even if canonicalize would succeed
        // This ensures tests that expect traversal to be rejected will pass regardless of filesystem layout.
        {
            use std::path::Component;
            let mut depth: i32 = 0;
            for component in path.components() {
                match component {
                    Component::ParentDir => {
                        depth -= 1;
                        if depth < 0 {
                            return Err(MemoryError::Embedding(
                                "Path traversal attempt detected: too many parent directory references".to_string()
                            ));
                        }
                    }
                    Component::Normal(_) => { depth += 1; }
                    Component::RootDir | Component::CurDir | Component::Prefix(_) => { /* ignore for depth */ }
                }
            }
        }

        // Try to canonicalize the path (resolves symlinks and normalizes)
        match path.canonicalize() {
            Ok(canonical) => {
                debug!("Canonicalized path: {} -> {}", path.display(), canonical.display());
                Ok(canonical)
            }
            Err(_) => {
                // If canonicalization fails (path doesn't exist yet), 
                // manually normalize the path components for security
                let mut normalized = PathBuf::new();
                let mut depth = 0i32; // Track directory depth to prevent going above root
                
                for component in path.components() {
                    match component {
                        std::path::Component::ParentDir => {
                            // Prevent going above root directory
                            depth -= 1;
                            if depth < 0 {
                                return Err(MemoryError::Embedding(
                                    "Path traversal attempt detected: too many parent directory references".to_string()
                                ));
                            }
                            normalized.pop();
                        }
                        std::path::Component::Normal(name) => {
                            // Validate component name doesn't contain dangerous characters
                            let name_str = name.to_string_lossy();
                            if name_str.contains("\0") || name_str.contains("/") || name_str.contains("\\") {
                                return Err(MemoryError::Embedding(
                                    format!("Invalid path component: {}", name_str)
                                ));
                            }
                            normalized.push(name);
                            depth += 1;
                        }
                        std::path::Component::RootDir => {
                            normalized.push("/");
                            depth = 0; // Reset depth at root
                        }
                        std::path::Component::CurDir => {
                            // Current directory reference, skip
                        }
                        std::path::Component::Prefix(prefix) => {
                            // Windows drive prefix (e.g., C:)
                            normalized.push(prefix.as_os_str());
                            depth = 0; // Reset depth at drive root
                        }
                    }
                }
                
                debug!("Normalized path: {} -> {}", path.display(), normalized.display());
                Ok(normalized)
            }
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig {
            model: ModelConfig {
                name: "embeddinggemma-300m".to_string(),
                huggingface_id: "google/embeddinggemma-300m".to_string(),
                version: "300m".to_string(),
                local_search_paths: vec![
                    "./models/embeddinggemma-300m".to_string(),
                    "~/.cache/models/embeddinggemma-300m".to_string(),
                ],
                required_files: vec![
                    "model.safetensors".to_string(),
                    "config.json".to_string(),
                    "tokenizer.json".to_string(),
                    "tokenizer_config.json".to_string(),
                ],
                min_model_size: 1_000_000, // 1MB
            },
            embedding: EmbeddingSettings {
                default_dimension: 512,
                matryoshka_dimensions: vec![768, 512, 256, 128],
                max_sequence_length: 2048,
                vocab_size: 256000,
            },
            prompts: PromptTemplates {
                query: "task: search result | query: {}".to_string(),
                document: "title: none | text: {}".to_string(),
                classification: "task: classification | text: {}".to_string(),
                similarity: "task: similarity | text: {}".to_string(),
            },
            performance: PerformanceConfig {
                default_device: "cpu".to_string(),
                torch_dtype: "bfloat16".to_string(),
                default_batch_size: 8,
                max_batch_size: 128,
                embedding_timeout: 30,
            },
            cache: CacheConfig {
                enabled: true,
                ttl_seconds: 3600,
                max_entries: 10000,
                eviction_strategy: "lru".to_string(),
            },
            validation: ValidationConfig {
                max_text_length: 8192,
                max_total_batch_chars: 1_048_576,
                sanitize_input: true,
                allow_empty_text: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.model.name, "embeddinggemma-300m");
        assert_eq!(config.embedding.default_dimension, 512);
        assert_eq!(config.performance.torch_dtype, "bfloat16");
    }
    
    #[test]
    fn test_path_traversal_prevention() {
        let config = EmbeddingConfig::default();
        
        // Test Unix-style path traversal attempts
        assert!(config.resolve_model_path("../../../etc/passwd").is_err());
        assert!(config.resolve_model_path("~/../../root").is_err());
        assert!(config.resolve_model_path("/etc/passwd").is_err());
        
        // Test Windows-style path traversal attempts
        assert!(config.resolve_model_path("..\\..\\windows\\system32").is_err());
        assert!(config.resolve_model_path("C:\\Windows\\System32").is_err());
        
        // Test URL-encoded traversal attempts
        assert!(config.resolve_model_path("%2e%2e/%2e%2e/etc/passwd").is_err());
        
        // Test valid model names
        assert!(config.resolve_model_path("embeddinggemma-300m").is_ok());
        assert!(config.resolve_model_path("all-MiniLM-L6-v2").is_ok());
    }
    
    #[test]
    fn test_normalize_path() {
        let config = EmbeddingConfig::default();
        
        // Test normal path normalization
        let normal_path = Path::new("./models/embeddinggemma");
        assert!(config.normalize_path(normal_path).is_ok());
        
        // Test path with parent directory references
        let parent_path = Path::new("models/../models/embeddinggemma");
        assert!(config.normalize_path(parent_path).is_ok());
        
        // Test excessive parent directory references (should fail)
        let invalid_path = Path::new("../../../../../../etc/passwd");
        assert!(config.normalize_path(invalid_path).is_err());
    }
    
    #[test]
    fn test_device_configuration() {
        let config = EmbeddingConfig::default();
        
        // Test default device
        assert_eq!(config.get_device(), "cpu");
        
        // Test environment override
        std::env::set_var("EMBEDDING_DEVICE", "cuda");
        assert_eq!(config.get_device(), "cuda");
        std::env::remove_var("EMBEDDING_DEVICE");
    }
}
