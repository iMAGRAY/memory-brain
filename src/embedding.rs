//! Embedding system for AI memory service using HTTP API to Python embedding server
//! 
//! This module provides vector embedding generation via HTTP API calls to a standalone
//! Python server running Sentence Transformers with the EmbeddingGemma-300M model.
//! 
//! Features:
//! - HTTP-based communication to bypass Python GIL limitations
//! - Parallel request handling for local usage (2-4 concurrent requests)
//! - EmbeddingGemma-300M support with optimal prompts
//! - Comprehensive caching with TTL and dimension validation
//! - Robust error handling and input validation
//! - Async-first design with timeout protection

use crate::types::{MemoryError, MemoryResult};
use crate::embedding_config::EmbeddingConfig;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use tokio::sync::Semaphore;
use tracing::{debug, info, warn, error};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use serde::{Deserialize, Serialize};
use reqwest::Client;

use serde_json::Value as JsonValue; // for robust /stats parsing

// Cache configuration constants
const CACHE_TTL_SECONDS: u64 = 3600; // 1 hour
const CACHE_KEY_PREFIX: &str = "emb";
const MAX_CACHE_SIZE: usize = 10000; // Maximum number of cached embeddings

// Input validation constants (reserved for future validation features)
#[allow(dead_code)]
const MAX_TEXT_LENGTH: usize = 8192; // Maximum characters per text
#[allow(dead_code)]
const MAX_BATCH_SIZE: usize = 128; // Maximum texts in single batch
#[allow(dead_code)]
const MAX_TOTAL_BATCH_CHARS: usize = 1048576; // 1MB total character limit per batch

// EmbeddingGemma dimension validation constants (reserved for future validation)
#[allow(dead_code)]
const MAX_EMBEDDING_DIMENSION_LIMIT: usize = 4096; // Safety upper bound

// Concurrency and timeout constants
const MAX_EMBEDDING_TIMEOUT_SECS: u64 = 30; // Maximum timeout for embedding operations
const DEFAULT_MAX_CONCURRENT_OPS: usize = 4; // Default max concurrent operations for local use

// Default embedding server URL
const DEFAULT_EMBEDDING_SERVER_URL: &str = "http://localhost:8090";

/// Task type for instruction formatting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskType {
    Query,
    Document,
    Classification,
    Similarity,
}

impl TaskType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::Query => "query",
            TaskType::Document => "document", 
            TaskType::Classification => "classification",
            TaskType::Similarity => "similarity",
        }
    }
}

/// Request structure for single text embedding
#[derive(Debug, Serialize)]
struct EmbedRequest {
    text: String,
}

/// Request structure for batch text embedding
#[derive(Debug, Serialize)]
struct EmbedBatchRequest {
    texts: Vec<String>,
}

/// Response structure for single text embedding
#[derive(Debug, Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
    dimension: usize,
}

/// Response structure for batch text embedding
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct EmbedBatchResponse {
    embeddings: Vec<Vec<f32>>,
    count: usize,
    dimension: usize,
}

/// HTTP-based embedding service using external Python server
pub struct EmbeddingService {
    client: Client,
    server_url: String,
    model_name: String,
    config: EmbeddingConfig,
    embedding_cache: Arc<dashmap::DashMap<String, (Arc<Vec<f32>>, u64)>>, // (embedding, timestamp)
    use_instructions: bool,
    matryoshka_dims: Vec<usize>, // Supported dimensions: [768, 512, 256, 128]
    #[allow(dead_code)]
    embedding_timeout: Duration,
    #[allow(dead_code)]
    max_text_length: usize,
    max_batch_size: usize,
    // Concurrency control for parallel requests
    concurrency_semaphore: Arc<Semaphore>,
    #[allow(dead_code)]
    max_concurrent_operations: usize,
    // Actual embedding dimension detected from server or fallback to config
    actual_dimension: usize,
}

/// Model information for compatibility
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub dimensions: usize,
    pub vocab_size: usize,
    pub max_sequence_length: usize,
}

impl EmbeddingService {
    /// Create new EmbeddingService with HTTP client for embedding server
    pub async fn new(
        model_name: &str,
        _tokenizer_path: &str, // Not needed for HTTP API
        _batch_size: usize,    // Handled by server
        _max_sequence_length: usize, // Handled by server
    ) -> MemoryResult<Self> {
        // Load configuration
        let config = EmbeddingConfig::load()?;
        
        // Calculate optimal max concurrent operations
        let max_concurrent_operations = std::cmp::max(
            1, // Ensure at least 1 concurrent operation
            std::cmp::min(config.performance.max_batch_size / 2, DEFAULT_MAX_CONCURRENT_OPS)
        );
        
        info!("Initializing HTTP-based EmbeddingService");
        info!("Configured max concurrent operations: {} (based on batch_size={}, default_max={})", 
              max_concurrent_operations, config.performance.max_batch_size, DEFAULT_MAX_CONCURRENT_OPS);
        
        // Create HTTP client with timeout
        let client = Client::builder()
            .timeout(Duration::from_secs(MAX_EMBEDDING_TIMEOUT_SECS))
            .build()
            .map_err(|e| MemoryError::Embedding(format!("Failed to create HTTP client: {}", e)))?;
        
        // Get server URL from environment or use default
        let server_url = std::env::var("EMBEDDING_SERVER_URL")
            .unwrap_or_else(|_| DEFAULT_EMBEDDING_SERVER_URL.to_string());
        
        info!("Using embedding server at: {}", server_url);
        
        // Test connection to embedding server
        let health_url = format!("{}/health", server_url);
        match client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => {
                info!("Successfully connected to embedding server");
            }
            Ok(response) => {
                warn!("Embedding server returned status: {}", response.status());
            }
            Err(e) => {
                error!("Failed to connect to embedding server: {}", e);
                return Err(MemoryError::Embedding(
                    format!("Embedding server not available at {}: {}", server_url, e)
                ));
            }
        }

        // Detect actual embedding dimension
        let mut actual_dimension = config.embedding.default_dimension;
        let stats_url = format!("{}/stats", server_url);
        let mut detected_from = "config";

        match client.get(&stats_url).send().await {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<JsonValue>().await {
                    Ok(json) => {
                        // Try multiple likely paths: "dimension", "embedding_dimension", "model.dimension"
                        let dim = json.get("dimension")
                            .and_then(|v| v.as_u64())
                            .or_else(|| json.get("embedding_dimension").and_then(|v| v.as_u64()))
                            .or_else(|| json.get("model").and_then(|m| m.get("dimension")).and_then(|v| v.as_u64()))
                            .map(|v| v as usize);
                        if let Some(d) = dim {
                            actual_dimension = d;
                            detected_from = "stats";
                            info!("Detected embedding dimension from /stats: {}", actual_dimension);
                        } else {
                            warn!("No 'dimension' field found in /stats response; will probe via /embed");
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse /stats JSON: {}; will probe via /embed", e);
                    }
                }
            }
            Ok(resp) => {
                warn!("Embedding server /stats returned non-success status: {}. Will probe via /embed.", resp.status());
            }
            Err(e) => {
                warn!("Failed to call /stats: {}. Will probe via /embed.", e);
            }
        }

        // If /stats didn't yield a dimension, probe with a single /embed call
        if detected_from != "stats" {
            let probe_url = format!("{}/embed", server_url);
            let probe_req = EmbedRequest { text: "dimension probe".to_string() };
            match client.post(&probe_url).json(&probe_req).send().await {
                Ok(resp) if resp.status().is_success() => {
                    match resp.json::<EmbedResponse>().await {
                        Ok(r) => {
                            // Basic sanity: make sure vector length matches reported dimension
                            if r.embedding.len() == r.dimension && r.dimension > 0 && r.dimension <= MAX_EMBEDDING_DIMENSION_LIMIT {
                                actual_dimension = r.dimension;
                                detected_from = "embed";
                                info!("Detected embedding dimension from /embed: {}", actual_dimension);
                            } else {
                                warn!(
                                    "Probe /embed returned inconsistent dimension (len={} vs dimension={}); using config fallback {}",
                                    r.embedding.len(),
                                    r.dimension,
                                    actual_dimension
                                );
                            }
                        }
                        Err(e) => {
                            warn!("Failed to parse /embed probe response: {}; using config fallback {}", e, actual_dimension);
                        }
                    }
                }
                Ok(resp) => {
                    warn!("Probe /embed returned status {}. Using config fallback {}.", resp.status(), actual_dimension);
                }
                Err(e) => {
                    warn!("Failed to call probe /embed: {}. Using config fallback {}.", e, actual_dimension);
                }
            }
        }

        info!("Embedding dimension ready: {} (source: {})", actual_dimension, detected_from);
        
        Ok(Self {
            client,
            server_url,
            model_name: model_name.to_string(),
            config: config.clone(),
            embedding_cache: Arc::new(dashmap::DashMap::new()),
            use_instructions: true, // Always use specialized prompts for EmbeddingGemma
            matryoshka_dims: vec![768, 512, 256, 128],
            embedding_timeout: Duration::from_secs(MAX_EMBEDDING_TIMEOUT_SECS),
            max_text_length: config.embedding.max_sequence_length,
            max_batch_size: config.performance.max_batch_size,
            concurrency_semaphore: Arc::new(Semaphore::new(max_concurrent_operations)),
            max_concurrent_operations,
            actual_dimension,
        })
    }
    
    /// Generate embedding for a single text with task type
    pub async fn embed(&self, text: &str, task: TaskType) -> MemoryResult<Vec<f32>> {
        // Format text with task-specific prompt
        let formatted_text = self.format_with_task(text, task);
        
        // Check cache first
        let cache_key = self.generate_cache_key(&formatted_text);
        if let Some(cached) = self.get_cached_embedding(&cache_key) {
            debug!("Cache hit for embedding");
            return Ok(cached.to_vec());
        }
        
        // Acquire semaphore permit for concurrency control
        let _permit = self.concurrency_semaphore.acquire().await
            .map_err(|e| MemoryError::Embedding(format!("Failed to acquire semaphore: {}", e)))?;
        
        // Make HTTP request to embedding server
        let url = format!("{}/embed", self.server_url);
        let request = EmbedRequest {
            text: formatted_text.clone(),
        };
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| MemoryError::Embedding(format!("HTTP request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(MemoryError::Embedding(
                format!("Embedding server error {}: {}", status, error_text)
            ));
        }
        
        let embed_response: EmbedResponse = response.json().await
            .map_err(|e| MemoryError::Embedding(format!("Failed to parse response: {}", e)))?;
        
        // Validate embedding dimension
        if embed_response.embedding.len() != embed_response.dimension {
            return Err(MemoryError::Embedding(
                format!("Dimension mismatch: {} vs {}", 
                       embed_response.embedding.len(), embed_response.dimension)
            ));
        }
        
        // Cache the result
        self.cache_embedding(cache_key, embed_response.embedding.clone());
        
        Ok(embed_response.embedding)
    }
    
    /// Generate embeddings for a batch of texts with task type
    pub async fn embed_batch(&self, texts: &[String], task: TaskType) -> MemoryResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        // Validate batch size
        if texts.len() > self.max_batch_size {
            return Err(MemoryError::Embedding(
                format!("Batch size {} exceeds maximum {}", texts.len(), self.max_batch_size)
            ));
        }
        
        // Format texts with task-specific prompts
        let formatted_texts: Vec<String> = texts.iter()
            .map(|t| self.format_with_task(t, task))
            .collect();
        
        // Check cache for each text
        let mut results = Vec::with_capacity(texts.len());
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();
        
        for (i, text) in formatted_texts.iter().enumerate() {
            let cache_key = self.generate_cache_key(text);
            if let Some(cached) = self.get_cached_embedding(&cache_key) {
                results.push(Some(cached.to_vec()));
            } else {
                results.push(None);
                uncached_indices.push(i);
                uncached_texts.push(text.clone());
            }
        }
        
        // If all texts were cached, return immediately
        if uncached_texts.is_empty() {
            debug!("All {} texts found in cache", texts.len());
            return Ok(results.into_iter().map(|r| r.unwrap()).collect());
        }
        
        debug!("Generating embeddings for {} uncached texts", uncached_texts.len());
        
        // Acquire semaphore permit for concurrency control
        let _permit = self.concurrency_semaphore.acquire().await
            .map_err(|e| MemoryError::Embedding(format!("Failed to acquire semaphore: {}", e)))?;
        
        // Make HTTP request to embedding server for uncached texts
        let url = format!("{}/embed_batch", self.server_url);
        let request = EmbedBatchRequest {
            texts: uncached_texts.clone(),
        };
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| MemoryError::Embedding(format!("HTTP request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(MemoryError::Embedding(
                format!("Embedding server error {}: {}", status, error_text)
            ));
        }
        
        let embed_response: EmbedBatchResponse = response.json().await
            .map_err(|e| MemoryError::Embedding(format!("Failed to parse response: {}", e)))?;
        
        // Validate response
        if embed_response.embeddings.len() != uncached_texts.len() {
            return Err(MemoryError::Embedding(
                format!("Response count mismatch: {} vs {}", 
                       embed_response.embeddings.len(), uncached_texts.len())
            ));
        }
        
        // Cache the new embeddings and fill in results
        for (idx, embedding) in uncached_indices.iter().zip(embed_response.embeddings.iter()) {
            let cache_key = self.generate_cache_key(&formatted_texts[*idx]);
            self.cache_embedding(cache_key, embedding.clone());
            results[*idx] = Some(embedding.clone());
        }
        
        // Convert results to final format
        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }
    
    /// Format text with task-specific prompt
    fn format_with_task(&self, text: &str, task: TaskType) -> String {
        if self.use_instructions {
            match task {
                TaskType::Query => format!("task: search result | query: {}", text),
                TaskType::Document => format!("title: none | text: {}", text),
                TaskType::Classification => format!("task: classification | text: {}", text),
                TaskType::Similarity => format!("task: similarity | text: {}", text),
            }
        } else {
            text.to_string()
        }
    }
    
    /// Generate cache key for text
    fn generate_cache_key(&self, text: &str) -> String {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        self.model_name.hash(&mut hasher);
        format!("{}:{}:{}", CACHE_KEY_PREFIX, self.model_name, hasher.finish())
    }
    
    /// Get embedding from cache if valid
    fn get_cached_embedding(&self, key: &str) -> Option<Arc<Vec<f32>>> {
        if let Some(entry) = self.embedding_cache.get(key) {
            let (embedding, timestamp) = entry.value();
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            
            if now - timestamp < CACHE_TTL_SECONDS {
                return Some(Arc::clone(embedding));
            } else {
                // Remove expired entry
                drop(entry);
                self.embedding_cache.remove(key);
            }
        }
        None
    }
    
    /// Cache embedding with timestamp
    fn cache_embedding(&self, key: String, embedding: Vec<f32>) {
        // Check cache size limit
        if self.embedding_cache.len() >= MAX_CACHE_SIZE {
            // Simple eviction: remove some old entries
            let to_remove: Vec<_> = self.embedding_cache
                .iter()
                .take(MAX_CACHE_SIZE / 10) // Remove 10% of cache
                .map(|entry| entry.key().clone())
                .collect();
            
            for key in to_remove {
                self.embedding_cache.remove(&key);
            }
        }
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        self.embedding_cache.insert(key, (Arc::new(embedding), timestamp));
    }
    
    /// Truncate text to specified dimension (Matryoshka Representation Learning)
    pub fn truncate_to_dimension(&self, embedding: &[f32], target_dim: usize) -> MemoryResult<Vec<f32>> {
        if !self.matryoshka_dims.contains(&target_dim) {
            return Err(MemoryError::Embedding(
                format!("Unsupported dimension {}. Supported: {:?}", target_dim, self.matryoshka_dims)
            ));
        }
        
        if embedding.len() < target_dim {
            return Err(MemoryError::Embedding(
                format!("Embedding dimension {} is less than target {}", embedding.len(), target_dim)
            ));
        }
        
        Ok(embedding[..target_dim].to_vec())
    }
    
    /// Get model information
    pub fn get_model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.model_name.clone(),
            dimensions: self.actual_dimension, // return detected/fallback actual dimension
            vocab_size: 256000, // EmbeddingGemma-300M vocab size
            max_sequence_length: self.config.embedding.max_sequence_length,
        }
    }
    
    /// Clear embedding cache
    pub fn clear_cache(&self) {
        self.embedding_cache.clear();
        info!("Cleared embedding cache");
    }
    
    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize) {
        let total = self.embedding_cache.len();
        let mut valid = 0;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        for entry in self.embedding_cache.iter() {
            let (_, timestamp) = entry.value();
            if now - timestamp < CACHE_TTL_SECONDS {
                valid += 1;
            }
        }
        
        (total, valid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_embedding_service_creation() {
        // This test requires the embedding server to be running
        match EmbeddingService::new(
            "embeddinggemma-300m",
            "",
            32,
            2048
        ).await {
            Ok(service) => {
                let info = service.get_model_info();
                assert_eq!(info.name, "embeddinggemma-300m");
                // Dimension is detected from /stats or /embed, or falls back to config; just ensure it's sane
                assert!(info.dimensions > 0 && info.dimensions <= 4096);
            }
            Err(e) => {
                // Expected if server is not running
                assert!(e.to_string().contains("not available"));
            }
        }
    }
    
    #[tokio::test]
    async fn test_cache_key_generation() {
        let service = match EmbeddingService::new(
            "test-model",
            "",
            32,
            2048
        ).await {
            Ok(s) => s,
            Err(_) => return, // Skip if server not available
        };
        
        let key1 = service.generate_cache_key("test text");
        let key2 = service.generate_cache_key("test text");
        let key3 = service.generate_cache_key("different text");
        
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
}
