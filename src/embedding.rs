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
use tracing::{debug, info, warn};
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
    // Text that will be embedded (with task-specific prompt already applied)
    text: String,
    // Explicit task type hint for the embedding server (ensures correct prompts + Matryoshka)
    task_type: String,
}

/// Request structure for batch text embedding
#[derive(Debug, Serialize)]
struct EmbedBatchRequest {
    // Batch of formatted texts
    texts: Vec<String>,
    // Task type applied uniformly to the batch
    task_type: String,
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
    #[allow(clippy::type_complexity)]
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
    // Service availability flag (graceful degradation) with interior mutability for late startup
    available: std::sync::atomic::AtomicBool,
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
        let max_concurrent_operations = (config.performance.max_batch_size / 2)
            .clamp(1, DEFAULT_MAX_CONCURRENT_OPS);
        
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
        
        // Test connection to embedding server (graceful degradation)
        let health_url = format!("{}/health", server_url);
        let mut available = false;
        match client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => {
                info!("Successfully connected to embedding server");
                available = true;
            }
            Ok(response) => {
                warn!("Embedding server returned status: {} — degraded mode (503 for embed ops)", response.status());
            }
            Err(e) => {
                warn!("Embedding server not available at {}: {} — degraded mode (503 for embed ops)", server_url, e);
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

        // If /stats didn't yield a dimension, probe with a single /embed call (only if available)
        if detected_from != "stats" && available {
            let probe_url = format!("{}/embed", server_url);
            let probe_req = EmbedRequest { text: "dimension probe".to_string(), task_type: TaskType::Query.as_str().to_string() };
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

        info!("Embedding dimension ready: {} (source: {}, available: {})", actual_dimension, detected_from, available);
        
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
            available: std::sync::atomic::AtomicBool::new(available),
        })
    }
    
    /// Generate embedding for a single text with task type
    pub async fn embed(&self, text: &str, task: TaskType) -> MemoryResult<Vec<f32>> {
        if !self.available.load(std::sync::atomic::Ordering::Relaxed) {
            // Attempt late health check (embedding server might have started after us)
            let health_url = format!("{}/health", self.server_url);
            match self.client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    self.available.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                _ => {
                    return Err(MemoryError::Embedding("Embedding server unavailable".to_string()));
                }
            }
        }
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
            task_type: task.as_str().to_string(),
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
        if !self.available.load(std::sync::atomic::Ordering::Relaxed) {
            // Attempt late health check
            let health_url = format!("{}/health", self.server_url);
            match self.client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    self.available.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                _ => {
                    return Err(MemoryError::Embedding("Embedding server unavailable".to_string()));
                }
            }
        }
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
            task_type: task.as_str().to_string(),
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
    
    /// Attempt to derive a stable title and body from input text.
    /// Heuristics:
    /// - If first non-empty line is a Markdown header (# ...), use it as title and drop from body
    /// - Else, if first line is reasonably short (5..=120 chars) and text has multiple lines, use it as title
    /// - Otherwise, None
    fn derive_title_and_body(&self, text: &str) -> (Option<String>, String) {
        let trimmed = text.trim();
        if trimmed.is_empty() { return (None, String::new()); }
        let mut lines = trimmed.lines();
        let mut first_nonempty = None;
        let mut collected: Vec<&str> = Vec::new();
        for (i, line) in lines.clone().enumerate() {
            if !line.trim().is_empty() { first_nonempty = Some((i, line)); break; }
        }
        let (idx, first_line) = match first_nonempty { Some(v) => v, None => (0, trimmed) };
        let is_md_header = first_line.trim_start().starts_with('#');
        let header_text = first_line.trim().trim_start_matches('#').trim();

        // Collect remaining lines as body (skipping first line if used as title)
        let mut body_lines: Vec<&str> = Vec::new();
        for (i, line) in trimmed.lines().enumerate() {
            if i == idx { continue; }
            body_lines.push(line);
        }
        let body_joined = if is_md_header {
            body_lines.join("\n").trim().to_string()
        } else {
            trimmed.to_string()
        };

        // Secondary heuristic: plain first-line title
        if !is_md_header {
            // Extract first physical line
            if let Some((first_line_full, rest)) = trimmed.split_once('\n') {
                let fl = first_line_full.trim();
                let len = fl.chars().count();
                if len >= 5 && len <= 120 {
                    return (Some(fl.to_string()), rest.trim().to_string());
                }
            }
            return (None, body_joined);
        }

        let title = if header_text.is_empty() { None } else { Some(header_text.to_string()) };
        (title, body_joined)
    }

    /// Sanitize a snippet for prompt insertion: collapse whitespace; replace pipes with slashes to avoid prompt syntax conflicts
    fn sanitize_for_prompt(&self, s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut prev_ws = false;
        for ch in s.chars() {
            let is_ws = ch.is_whitespace();
            if is_ws {
                if !prev_ws { out.push(' '); }
                prev_ws = true;
            } else {
                prev_ws = false;
                out.push(if ch == '|' { '/' } else { ch });
            }
        }
        out.trim().to_string()
    }

    /// Format text with task-specific prompt using EmbeddingGemma best-practice templates (docs/embeddinggemma-context.md)
    fn format_with_task(&self, text: &str, task: TaskType) -> String {
        if !self.use_instructions { return text.to_string(); }
        match task {
            TaskType::Query => {
                let q = self.sanitize_for_prompt(text);
                format!("task: search result | query: {}", q)
            }
            TaskType::Document => {
                let (title_opt, body) = self.derive_title_and_body(text);
                let title = title_opt.unwrap_or_else(|| "none".to_string());
                let t = self.sanitize_for_prompt(&title);
                let b = self.sanitize_for_prompt(&body);
                format!("title: {} | text: {}", t, b)
            }
            TaskType::Classification => {
                let s = self.sanitize_for_prompt(text);
                format!("task: classification | text: {}", s)
            }
            TaskType::Similarity => {
                let s = self.sanitize_for_prompt(text);
                format!("task: similarity | text: {}", s)
            }
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

    /// Embedding server availability
    pub fn is_available(&self) -> bool { self.available.load(std::sync::atomic::Ordering::Relaxed) }
    
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

    #[test]
    fn test_derive_title_and_body_md_header() {
        let svc = tokio_test::block_on(EmbeddingService::new("test","",32,2048)).ok();
        if svc.is_none() { return; }
        let svc = svc.unwrap();
        let text = "# Заголовок документа\nЭто тело документа.\nСодержимое.";
        let (title, body) = svc.derive_title_and_body(text);
        assert_eq!(title.as_deref(), Some("Заголовок документа"));
        assert!(body.contains("Это тело документа."));
    }

    #[test]
    fn test_derive_title_and_body_plain_first_line() {
        let svc = tokio_test::block_on(EmbeddingService::new("test","",32,2048)).ok();
        if svc.is_none() { return; }
        let svc = svc.unwrap();
        let text = "Заголовок в первой строке\nДальше идёт содержимое без markdown";
        let (title, body) = svc.derive_title_and_body(text);
        assert_eq!(title.as_deref(), Some("Заголовок в первой строке"));
        assert!(body.starts_with("Дальше"));
    }

    #[test]
    fn test_format_with_task_document() {
        let svc = tokio_test::block_on(EmbeddingService::new("test","",32,2048)).ok();
        if svc.is_none() { return; }
        let svc = svc.unwrap();
        let text = "# Title | Pipe\nBody with  multiple\tspaces.";
        let formatted = svc.format_with_task(text, TaskType::Document);
        assert!(formatted.starts_with("title:"));
        assert!(formatted.contains("text:"));
        // Pipes are sanitized to slashes
        // Our formatting uses a single '|' between title and text; ensure sanitized body/title don't introduce extras
        let first_pipe = formatted.find('|').unwrap_or(0);
        let remaining = &formatted[first_pipe+1..];
        assert!(remaining.find('|').is_none());
    }
}
