//! Embedding system for AI memory service using Python Sentence Transformers via PyO3
//! 
//! This module provides vector embedding generation using Sentence Transformers
//! with the EmbeddingGemma-300M model for high-quality multilingual text embeddings.
//! 
//! Features:
//! - Production-ready PyO3 integration with proper GIL management
//! - EmbeddingGemma-300M support with optimal prompts
//! - Matryoshka Representation Learning (768, 512, 256, 128 dimensions)
//! - Comprehensive caching with TTL and dimension validation
//! - Robust error handling and input validation
//! - Async-first design with timeout protection

use crate::types::{MemoryError, MemoryResult};
use crate::embedding_config::EmbeddingConfig;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use tracing::{debug, info, warn};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};

// Cache configuration constants
const CACHE_TTL_SECONDS: u64 = 3600; // 1 hour
const CACHE_KEY_PREFIX: &str = "emb";
const MAX_CACHE_SIZE: usize = 10000; // Maximum number of cached embeddings
// Input validation constants
#[cfg(test)]
const MAX_TEXT_LENGTH: usize = 8192; // Maximum characters per text
#[cfg(test)]
const MAX_BATCH_SIZE: usize = 128; // Maximum texts in single batch
const MAX_TOTAL_BATCH_CHARS: usize = 1048576; // 1MB total character limit per batch

// EmbeddingGemma dimension validation constants
const MAX_EMBEDDING_DIMENSION_LIMIT: usize = 4096; // Safety upper bound to prevent memory exhaustion attacks (far above 768)

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

/// Python-based embedding service using Sentence Transformers
pub struct EmbeddingService {
    python_module: Arc<Mutex<PyObject>>,
    model_name: String,
    config: EmbeddingConfig, // Configuration for EmbeddingGemma
    embedding_cache: Arc<dashmap::DashMap<String, (Arc<Vec<f32>>, u64)>>, // (embedding, timestamp)
    use_instructions: bool,
    matryoshka_dims: Vec<usize>, // Supported dimensions: [768, 512, 256, 128]
    embedding_timeout: Duration,
    max_text_length: usize,
    max_batch_size: usize,
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
    /// Create new EmbeddingService with Python Sentence Transformers and local model support
    pub async fn new(
        model_name: &str,
        _tokenizer_path: &str, // Not needed for Sentence Transformers
        _batch_size: usize,    // Handled by Python
        _max_sequence_length: usize,
    ) -> MemoryResult<Self> {
        // Load configuration
        let config = EmbeddingConfig::load()?;
        
        // Resolve model path with security validation and cross-platform support
        let actual_model_path = config.resolve_model_path(model_name)?;
        info!("Using model: {}", actual_model_path);
        
        // Validate model path
        if actual_model_path.is_empty() || actual_model_path.len() > 512 {
            return Err(MemoryError::Embedding(
                "Model path must be non-empty and less than 512 characters".to_string()
            ));
        }
        
        info!("Initializing EmbeddingService with model: {}", actual_model_path);
        
        // Capture config for use in closure
        let config_clone = config.clone();
        let actual_model_path_clone = actual_model_path.clone();
        
        // Initialize Python interpreter and import required modules with proper dtype for EmbeddingGemma
        let python_module = Python::with_gil(move |py| -> MemoryResult<PyObject> {
            // Import required modules
            let st_module = py.import_bound("sentence_transformers")
                .map_err(|e| MemoryError::Embedding(format!(
                    "Failed to import sentence_transformers: {}. Please install: pip install sentence-transformers>=3.2.0",
                    e
                )))?;
            
            // For EmbeddingGemma, we need to set proper device and model kwargs
            let kwargs = pyo3::types::PyDict::new_bound(py);
            
            // Check if this is EmbeddingGemma and set appropriate configuration
            if actual_model_path_clone.contains("embeddinggemma") {
                // Try to import torch and set model_kwargs
                match py.import_bound("torch") {
                    Ok(torch_module) => {
                        // Create model_kwargs dict for dtype configuration
                        let model_kwargs = pyo3::types::PyDict::new_bound(py);
                        
                        // Use configured dtype (bfloat16 or float32)
                        let dtype_str = config_clone.get_torch_dtype();
                        if dtype_str == "bfloat16" {
                            if let Ok(bfloat16) = torch_module.getattr("bfloat16") {
                                model_kwargs.set_item("torch_dtype", bfloat16)?;
                                info!("Using bfloat16 dtype for EmbeddingGemma-300m");
                            }
                        } else if dtype_str == "float32" {
                            if let Ok(float32) = torch_module.getattr("float32") {
                                model_kwargs.set_item("torch_dtype", float32)?;
                                info!("Using float32 dtype for EmbeddingGemma-300m");
                            }
                        }
                        
                        kwargs.set_item("model_kwargs", model_kwargs)?;
                    
                        // Set device from configuration with validation
                        let device = config_clone.get_device();
                        // Validate device string for security
                        let valid_device = match device.as_str() {
                            "cpu" => "cpu",
                            "cuda" => "cuda",
                            d if d.starts_with("cuda:") && d.len() == 6 && d.chars().nth(5).map_or(false, |c| c.is_ascii_digit()) => d,
                            "auto" => "cpu", // Default to CPU for auto
                            _ => {
                                warn!("Invalid device '{}', defaulting to CPU", device);
                                "cpu"
                            }
                        };
                        kwargs.set_item("device", valid_device)?;
                        info!("Using device: {}", valid_device);
                    }
                    Err(e) => {
                        warn!("Failed to import torch for dtype configuration: {}. Using default SentenceTransformer settings.", e);
                    }
                }
            }
            
            // Create SentenceTransformer model with proper configuration
            let model = if kwargs.len() > 0 {
                st_module
                    .getattr("SentenceTransformer")?
                    .call((&actual_model_path_clone,), Some(&kwargs))
                    .map_err(|e| MemoryError::Embedding(format!(
                        "Failed to load model '{}': {}. For local models, ensure all files are present.",
                        actual_model_path_clone, e
                    )))?
            } else {
                st_module
                    .getattr("SentenceTransformer")?
                    .call1((&actual_model_path_clone,))
                    .map_err(|e| MemoryError::Embedding(format!(
                        "Failed to load model '{}': {}.",
                        actual_model_path_clone, e
                    )))?
            };
            
            info!("Successfully loaded Sentence Transformers model: {}", actual_model_path_clone);
            Ok(model.to_object(py))
        })?;
        
        Ok(EmbeddingService {
            python_module: Arc::new(Mutex::new(python_module)),
            model_name: actual_model_path.to_string(),
            config: config.clone(),
            embedding_cache: Arc::new(dashmap::DashMap::new()),
            use_instructions: true,
            matryoshka_dims: config.embedding.matryoshka_dimensions.clone(),
            embedding_timeout: Duration::from_secs(config.performance.embedding_timeout),
            max_text_length: config.validation.max_text_length,
            max_batch_size: config.performance.max_batch_size,
        })
    }
    
    /// Generate cache key with proper hashing and task context
    fn generate_cache_key(&self, text: &str, task_type: TaskType, dimensions: usize) -> String {
        Self::hash_cache_key(text, task_type, dimensions, self.use_instructions)
    }
    
    /// Static method to generate consistent cache keys
    fn hash_cache_key(text: &str, task_type: TaskType, dimensions: usize, use_instructions: bool) -> String {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        task_type.hash(&mut hasher);
        dimensions.hash(&mut hasher);
        use_instructions.hash(&mut hasher);
        format!("{}_{:x}_{}", CACHE_KEY_PREFIX, hasher.finish(), dimensions)
    }
    
    /// Get cached embedding with TTL validation
    fn get_cached_embedding(&self, key: &str) -> Option<Arc<Vec<f32>>> {
        if let Some(entry) = self.embedding_cache.get(key) {
            let (embedding, timestamp) = entry.value();
            
            match SystemTime::now().duration_since(UNIX_EPOCH) {
                Ok(duration) => {
                    let current_time = duration.as_secs();
                    
                    if *timestamp > current_time {
                        debug!("Invalid cache entry with future timestamp, removing");
                        drop(entry);
                        self.embedding_cache.remove(key);
                        return None;
                    }
                    
                    match current_time.checked_sub(*timestamp) {
                        Some(age) if age < CACHE_TTL_SECONDS => {
                            Some(embedding.clone())
                        }
                        Some(_) => {
                            debug!("Cache entry expired, removing");
                            drop(entry);
                            self.embedding_cache.remove(key);
                            None
                        }
                        None => {
                            debug!("Cache entry timestamp error, removing");
                            drop(entry);
                            self.embedding_cache.remove(key);
                            None
                        }
                    }
                }
                Err(e) => {
                    debug!("System time error, skipping cache: {}", e);
                    None
                }
            }
        } else {
            None
        }
    }
    
    /// Store embedding in cache with current timestamp and size control
    fn store_cached_embedding(&self, key: String, embedding: &[f32]) -> MemoryResult<()> {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => {
                let timestamp = duration.as_secs();
                
                // Enforce cache size limit
                if self.embedding_cache.len() >= MAX_CACHE_SIZE {
                    self.evict_oldest_entries(MAX_CACHE_SIZE / 4);
                }
                
                let cached_embedding = Arc::new(embedding.to_vec());
                self.embedding_cache.insert(key, (cached_embedding, timestamp));
                Ok(())
            }
            Err(e) => {
                debug!("Failed to store cache entry due to time error: {}", e);
                Ok(())
            }
        }
    }
    
    /// Evict oldest cache entries
    fn evict_oldest_entries(&self, count: usize) {
        if count == 0 || self.embedding_cache.is_empty() {
            return;
        }
        
        let actual_count = count.min(self.embedding_cache.len());
        let mut entries: Vec<_> = self.embedding_cache
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().1))
            .collect();
        
        entries.select_nth_unstable_by_key(actual_count.saturating_sub(1), |(_, timestamp)| *timestamp);
        
        for (key, _) in entries.into_iter().take(actual_count) {
            self.embedding_cache.remove(&key);
        }
        
        debug!("Evicted {} oldest cache entries", actual_count);
    }
    
    /// Generate embeddings for single text using Sentence Transformers with EmbeddingGemma prompts
    pub async fn embed_text(&self, text: &str) -> MemoryResult<Vec<f32>> {
        if text.is_empty() {
            return Ok(vec![]);
        }
        
        self.validate_text_input(text)?;
        debug!("Processing text for embedding: length={}", text.len());
        
        let cache_key = self.generate_cache_key(text, TaskType::Document, self.matryoshka_dims[0]);
        if let Some(cached) = self.get_cached_embedding(&cache_key) {
            debug!("Cache hit for text embedding");
            return Ok((*cached).clone());
        }
        
        let sanitized_text = self.sanitize_text_input(text);
        let formatted_text = self.format_with_instruction(&sanitized_text, TaskType::Document);
        let embedding = self.encode_single_text(formatted_text).await?;
        self.validate_embedding_dimension(&embedding, self.matryoshka_dims[0], "Single text embedding")?;
        self.store_cached_embedding(cache_key, &embedding)?;
        
        debug!("Generated embedding with dimension: {}", embedding.len());
        Ok(embedding)
    }
    
    /// Internal method for encoding single text with Python Sentence Transformers and EmbeddingGemma prompts
    async fn encode_single_text(&self, text: String) -> MemoryResult<Vec<f32>> {
        if text.is_empty() {
            debug!("Empty text provided to encode_single_text, returning empty vector");
            return Ok(vec![]);
        }
        
        debug!("Encoding text with length: {}", text.len());
        
        let text_clone = text.clone();
        let python_module = self.python_module.clone();
        self.execute_python_task(move || -> MemoryResult<Vec<f32>> {
            Python::with_gil(move |py| -> MemoryResult<Vec<f32>> {
                let model_guard = python_module.lock()
                    .map_err(|e| MemoryError::Embedding(format!("Failed to lock Python module: {}", e)))?;
                let model = model_guard.bind(py);
                
                // Additional security validation
                if text_clone.len() > 8192 {
                    return Err(MemoryError::Embedding("Text too long for encoding".to_string()));
                }
                
                let texts = PyList::new_bound(py, &[text_clone.as_str()]);
                let kwargs = pyo3::types::PyDict::new_bound(py);
                kwargs.set_item("normalize_embeddings", true)?;
                kwargs.set_item("convert_to_tensor", false)?;
                
                debug!("Calling Python model.encode with text length: {}", text.len());
                
                // Call model.encode without releasing GIL (PyO3 0.20 limitation)
                let result = model
                    .call_method("encode", (texts,), Some(&kwargs))
                    .map_err(|e| MemoryError::Embedding(format!("Python encode failed: {}", e)))?;
                
                debug!("Python encode completed successfully");
                Self::extract_1d_embedding(result, py)
            })
        }, "Single text encoding").await
    }
    
    /// Extract 1D embedding from Python result with comprehensive validation
    fn extract_1d_embedding(result: Bound<'_, PyAny>, _py: Python<'_>) -> MemoryResult<Vec<f32>> {
        // Simple approach: convert to Python list first, then extract
        let list_obj = result.call_method0("tolist")
            .map_err(|e| MemoryError::Embedding(format!("Failed to convert to Python list: {}", e)))?;
        
        // Handle potential 2D array (batch dimension) by taking the first element
        let embedding: Vec<f32> = if let Ok(vec2d) = list_obj.extract::<Vec<Vec<f32>>>() {
            // 2D array case - take first batch element
            if vec2d.is_empty() {
                return Err(MemoryError::Embedding("Model returned empty batch".to_string()));
            }
            vec2d.into_iter().next().unwrap_or_default()
        } else {
            // 1D array case - direct extraction
            list_obj.extract()
                .map_err(|e| MemoryError::Embedding(format!("Failed to extract Vec<f32> from Python list: {}", e)))?
        };
        
        // Validate embedding is not empty
        if embedding.is_empty() {
            return Err(MemoryError::Embedding("Empty embedding vector returned from model".to_string()));
        }
        
        // Security check BEFORE further processing to prevent DoS attacks
        if embedding.len() > MAX_EMBEDDING_DIMENSION_LIMIT {
            return Err(MemoryError::Embedding(format!(
                "Embedding dimension {} exceeds safety limit {} (potential DoS attack)", 
                embedding.len(), MAX_EMBEDDING_DIMENSION_LIMIT
            )));
        }
        
        // Validate embedding contains finite float values for robustness
        if embedding.iter().any(|&f| !f.is_finite()) {
            return Err(MemoryError::Embedding("Embedding contains invalid float values (NaN or infinite)".to_string()));
        }
        
        debug!("Successfully extracted 1D embedding with dimension: {}", embedding.len());
        Ok(embedding)
    }
    
    /// Generate embeddings for batch of texts with EmbeddingGemma prompts, caching and validation
    pub async fn embed_batch(&self, texts: &[String]) -> MemoryResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        self.validate_batch_input(texts)?;
        debug!("Processing batch with {} texts", texts.len());
        
        let embeddings = self.embed_batch_with_task_type(texts, TaskType::Document).await?;
        debug!("Generated batch embeddings: {} vectors of dimension {}", 
               embeddings.len(), 
               embeddings.first().map(|v| v.len()).unwrap_or(0));
        Ok(embeddings)
    }
    
    /// Generate batch embeddings for queries with EmbeddingGemma query prompts
    pub async fn embed_batch_queries(&self, queries: &[String]) -> MemoryResult<Vec<Vec<f32>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        
        self.validate_batch_input(queries)?;
        debug!("Processing query batch with {} texts", queries.len());
        
        let embeddings = self.embed_batch_with_task_type(queries, TaskType::Query).await?;
        debug!("Generated query batch embeddings: {} vectors", embeddings.len());
        Ok(embeddings)
    }
    
    /// Internal batch processing with task-specific prompts and caching
    async fn embed_batch_with_task_type(&self, texts: &[String], task_type: TaskType) -> MemoryResult<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        let mut uncached_texts = Vec::new();
        let mut uncached_indices = Vec::new();
        
        // Check cache for each text individually to maximize cache hits
        for (i, text) in texts.iter().enumerate() {
            let sanitized_text = self.sanitize_text_input(text);
            let cache_key = self.generate_cache_key(&sanitized_text, task_type, self.matryoshka_dims[0]);
            
            if let Some(cached) = self.get_cached_embedding_validated(&cache_key, self.matryoshka_dims[0]) {
                debug!("Cache hit for batch index {} with validated dimension", i);
                embeddings.push((*cached).clone());
            } else {
                embeddings.push(vec![]); // Placeholder
                uncached_texts.push(sanitized_text);
                uncached_indices.push(i);
            }
        }
        
        // Process uncached texts in batch if any
        if !uncached_texts.is_empty() {
            debug!("Processing {} uncached texts from batch", uncached_texts.len());
            let formatted_texts: Vec<String> = uncached_texts.iter()
                .map(|text| self.format_with_instruction(text, task_type))
                .collect();
            
            let new_embeddings = self.encode_batch_texts(formatted_texts).await?;
            
            // Store in cache and update results
            for (idx, &batch_index) in uncached_indices.iter().enumerate() {
                let sanitized_text = &uncached_texts[idx];
                let cache_key = self.generate_cache_key(sanitized_text, task_type, self.matryoshka_dims[0]);
                let embedding = &new_embeddings[idx];
                
                self.store_cached_embedding(cache_key, embedding)?;
                embeddings[batch_index] = embedding.clone();
            }
        }
        
        Ok(embeddings)
    }
    
    /// Internal method for batch encoding with EmbeddingGemma optimization and proper error handling
    async fn encode_batch_texts(&self, texts: Vec<String>) -> MemoryResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        debug!("Batch encoding {} texts with total length: {}", 
               texts.len(), 
               texts.iter().map(|t| t.len()).sum::<usize>());
        
        let texts_clone = texts.clone();
        let python_module = self.python_module.clone();
        self.execute_python_task(move || -> MemoryResult<Vec<Vec<f32>>> {
            Python::with_gil(move |py| -> MemoryResult<Vec<Vec<f32>>> {
                let model_guard = python_module.lock()
                    .map_err(|e| MemoryError::Embedding(format!("Failed to lock Python module: {}", e)))?;
                let model = model_guard.bind(py);
                
                let py_texts: Vec<&str> = texts_clone.iter().map(|s| s.as_str()).collect();
                let texts_list = PyList::new_bound(py, &py_texts);
                let kwargs = pyo3::types::PyDict::new_bound(py);
                kwargs.set_item("normalize_embeddings", true)?;
                kwargs.set_item("convert_to_tensor", false)?;
                
                debug!("Calling Python model.encode for batch of {} texts", texts_clone.len());
                
                // Call model.encode without releasing GIL (PyO3 0.20 limitation)
                let result = model
                    .call_method("encode", (texts_list,), Some(&kwargs))
                    .map_err(|e| MemoryError::Embedding(format!("Python batch encode failed: {}", e)))?;
                
                debug!("Python batch encode completed successfully");
                Self::extract_2d_embeddings(result, py, texts_clone.len())
            })
        }, "Batch text encoding").await
    }
    
    /// Generate batch embeddings with specified Matryoshka dimension
    pub async fn embed_batch_with_dimension(&self, texts: &[String], task_type: TaskType, dimension: usize) -> MemoryResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        if !self.matryoshka_dims.contains(&dimension) {
            return Err(MemoryError::Embedding(format!(
                "Unsupported dimension {}. EmbeddingGemma supports: {:?}", 
                dimension, self.matryoshka_dims
            )));
        }
        
        self.validate_batch_input(texts)?;
        debug!("Processing batch with {} texts for dimension {}", texts.len(), dimension);
        
        let mut embeddings = Vec::with_capacity(texts.len());
        let mut uncached_texts = Vec::new();
        let mut uncached_indices = Vec::new();
        
        // Check cache with dimension-specific keys
        for (i, text) in texts.iter().enumerate() {
            let sanitized_text = self.sanitize_text_input(text);
            let cache_key = self.generate_cache_key(&sanitized_text, task_type, dimension);
            
            if let Some(cached) = self.get_cached_embedding(&cache_key) {
                debug!("Cache hit for batch index {} with dimension {}", i, dimension);
                embeddings.push((*cached).clone());
            } else {
                embeddings.push(vec![]); // Placeholder
                uncached_texts.push(sanitized_text);
                uncached_indices.push(i);
            }
        }
        
        // Process uncached texts with Matryoshka dimension
        if !uncached_texts.is_empty() {
            debug!("Processing {} uncached texts with Matryoshka dimension {}", uncached_texts.len(), dimension);
            let formatted_texts: Vec<String> = uncached_texts.iter()
                .map(|text| self.format_with_instruction(text, task_type))
                .collect();
            
            let new_embeddings = self.encode_batch_with_matryoshka(formatted_texts, Some(dimension)).await?;
            
            // Store in cache and update results
            for (idx, &batch_index) in uncached_indices.iter().enumerate() {
                let sanitized_text = &uncached_texts[idx];
                let cache_key = self.generate_cache_key(sanitized_text, task_type, dimension);
                let embedding = &new_embeddings[idx];
                
                self.store_cached_embedding(cache_key, embedding)?;
                embeddings[batch_index] = embedding.clone();
            }
        }
        
        debug!("Generated batch embeddings: {} vectors of dimension {}", embeddings.len(), dimension);
        Ok(embeddings)
    }
    
    /// Internal batch encoding with Matryoshka dimension support
    async fn encode_batch_with_matryoshka(&self, texts: Vec<String>, truncate_dim: Option<usize>) -> MemoryResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        debug!("Batch Matryoshka encoding {} texts with truncation: {:?}", texts.len(), truncate_dim);
        
        let texts_clone = texts.clone();
        let truncate_dim_clone = truncate_dim;
        let python_module = self.python_module.clone();
        self.execute_python_task(move || -> MemoryResult<Vec<Vec<f32>>> {
            Python::with_gil(move |py| -> MemoryResult<Vec<Vec<f32>>> {
                let model_guard = python_module.lock()
                    .map_err(|e| MemoryError::Embedding(format!("Failed to lock Python module: {}", e)))?;
                let model = model_guard.bind(py);
                
                let py_texts: Vec<&str> = texts_clone.iter().map(|s| s.as_str()).collect();
                let texts_list = PyList::new_bound(py, &py_texts);
                let kwargs = pyo3::types::PyDict::new_bound(py);
                kwargs.set_item("normalize_embeddings", true)?;
                kwargs.set_item("convert_to_tensor", false)?;
                
                if let Some(dim) = truncate_dim_clone {
                    if ![128, 256, 512, 768].contains(&dim) {
                        return Err(MemoryError::Embedding(format!(
                            "Invalid Matryoshka dimension {}. Supported: 128, 256, 512, 768", dim
                        )));
                    }
                    kwargs.set_item("truncate_dim", dim)?;
                    debug!("Using batch Matryoshka truncation to {} dimensions", dim);
                }
                
                // Call model.encode without releasing GIL (PyO3 0.20 limitation)
                let result = model
                    .call_method("encode", (texts_list,), Some(&kwargs))
                    .map_err(|e| MemoryError::Embedding(format!("Python batch Matryoshka encode failed: {}", e)))?;
                
                debug!("Python batch Matryoshka encode completed successfully");
                Self::extract_2d_embeddings(result, py, texts_clone.len())
            })
        }, "Batch Matryoshka encoding").await
    }
    
    /// Extract 2D embeddings from Python result with comprehensive validation for EmbeddingGemma
    fn extract_2d_embeddings(
        result: Bound<'_, PyAny>, 
        _py: Python<'_>, 
        expected_batch_size: usize
    ) -> MemoryResult<Vec<Vec<f32>>> {
        let array = result.extract::<Bound<'_, PyArray2<f32>>>()
            .map_err(|e| MemoryError::Embedding(format!("Failed to extract 2D numpy array: {}", e)))?;
        
        let shape = array.shape().to_vec();
        
        if shape.len() != 2 {
            return Err(MemoryError::Embedding(
                format!("Expected 2D array, got {}D array with shape {:?}", shape.len(), shape)
            ));
        }
        
        if shape[0] != expected_batch_size {
            return Err(MemoryError::Embedding(
                format!("Batch size mismatch: expected {}, got {}", expected_batch_size, shape[0])
            ));
        }
        
        if shape[1] == 0 {
            return Err(MemoryError::Embedding("Model returned zero-dimensional embeddings".to_string()));
        }
        
        // Validate EmbeddingGemma dimensions
        let expected_dims = [128, 256, 512, 768];
        if !expected_dims.contains(&shape[1]) {
            debug!("Unexpected batch embedding dimension: {}, expected one of {:?}", shape[1], expected_dims);
        }
        
        // Безопасное извлечение данных из 2D numpy массива
        let data = array.to_vec()
            .map_err(|e| MemoryError::Embedding(format!("Failed to read 2D numpy array: {:?}", e)))?;
        
        // Validate all floats are finite
        if data.iter().any(|&f| !f.is_finite()) {
            return Err(MemoryError::Embedding("Model returned invalid float values in batch embeddings".to_string()));
        }
        
        let mut embeddings = Vec::with_capacity(shape[0]);
        let dim = shape[1];
        
        for i in 0..shape[0] {
            let start = i * dim;
            let end = start + dim;
            // Bounds checking for safety
            if end <= data.len() {
                embeddings.push(data[start..end].to_vec());
            } else {
                return Err(MemoryError::Embedding(
                    format!("Array bounds error: tried to access range {}..{} in array of length {}", start, end, data.len())
                ));
            }
        }
        
        debug!("Successfully extracted 2D embeddings: {} vectors of dimension {}", embeddings.len(), dim);
        Ok(embeddings)
    }
    
    /// Generate embeddings for query with EmbeddingGemma-specific formatting
    pub async fn embed_query(&self, query: &str) -> MemoryResult<Vec<f32>> {
        if query.is_empty() {
            return Ok(vec![]);
        }
        
        self.validate_text_input(query)?;
        debug!("Processing query for embedding: length={}", query.len());
        
        let cache_key = self.generate_cache_key(query, TaskType::Query, self.matryoshka_dims[0]);
        if let Some(cached) = self.get_cached_embedding(&cache_key) {
            debug!("Cache hit for query embedding");
            return Ok((*cached).clone());
        }
        
        let sanitized_query = self.sanitize_text_input(query);
        let formatted_query = self.format_with_instruction(&sanitized_query, TaskType::Query);
        let embedding = self.encode_single_text(formatted_query).await?;
        self.validate_embedding_dimension(&embedding, self.matryoshka_dims[0], "Query embedding")?;
        self.store_cached_embedding(cache_key, &embedding)?;
        
        debug!("Generated query embedding with dimension: {}", embedding.len());
        Ok(embedding)
    }
    
    /// Generate embeddings for document with EmbeddingGemma-specific formatting
    pub async fn embed_document(&self, document: &str) -> MemoryResult<Vec<f32>> {
        self.embed_text(document).await // embed_text already uses Document task type
    }
    
    /// Format text with EmbeddingGemma-specific instructions for optimal performance
    pub fn format_with_instruction(&self, text: &str, task_type: TaskType) -> String {
        if !self.use_instructions {
            return text.to_string();
        }
        
        // EmbeddingGemma-300M optimal prompts based on official documentation
        match task_type {
            TaskType::Query => format!("task: search result | query: {}", text),
            TaskType::Document => format!("title: none | text: {}", text),
            TaskType::Classification => format!("task: classification | text: {}", text),
            TaskType::Similarity => format!("task: similarity | text: {}", text),
        }
    }
    
    /// Get model information
    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            name: self.model_name.clone(),
            dimensions: self.matryoshka_dims[0], // Default dimension
            vocab_size: 250000, // Approximate for EmbeddingGemma
            max_sequence_length: 2048,
        }
    }
    
    /// Clear embedding cache
    pub fn clear_cache(&self) {
        self.embedding_cache.clear();
        info!("Embedding cache cleared");
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.embedding_cache.len(), MAX_CACHE_SIZE)
    }
    
    /// Get supported embedding dimensions (Matryoshka)
    pub fn supported_dimensions(&self) -> &[usize] {
        &self.matryoshka_dims
    }
    
    /// Check if a dimension is supported
    pub fn is_dimension_supported(&self, dim: usize) -> bool {
        self.matryoshka_dims.contains(&dim)
    }
    
    /// Validate single text input
    fn validate_text_input(&self, text: &str) -> MemoryResult<()> {
        if text.len() > self.max_text_length {
            return Err(MemoryError::Embedding(
                format!("Text length {} exceeds maximum of {}", text.len(), self.max_text_length)
            ));
        }
        Ok(())
    }
    
    /// Validate batch input
    fn validate_batch_input(&self, texts: &[String]) -> MemoryResult<()> {
        if texts.len() > self.max_batch_size {
            return Err(MemoryError::Embedding(
                format!("Batch size {} exceeds maximum of {}", texts.len(), self.max_batch_size)
            ));
        }
        
        let total_chars: usize = texts.iter().map(|t| t.len()).sum();
        if total_chars > MAX_TOTAL_BATCH_CHARS {
            return Err(MemoryError::Embedding(
                format!("Total batch characters {} exceeds maximum of {}", total_chars, MAX_TOTAL_BATCH_CHARS)
            ));
        }
        
        for (i, text) in texts.iter().enumerate() {
            if text.is_empty() {
                return Err(MemoryError::Embedding(
                    format!("Empty string found at batch index {}", i)
                ));
            }
            
            if text.len() > self.max_text_length {
                return Err(MemoryError::Embedding(
                    format!("Text at index {} length {} exceeds maximum of {}", i, text.len(), self.max_text_length)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Enhanced sanitization for text input to prevent potential code injection and ensure EmbeddingGemma compatibility
    fn sanitize_text_input(&self, text: &str) -> String {
        text.chars()
            .filter(|&c| {
                // Remove null bytes, control characters, and potentially dangerous chars
                c != '\0' && c != '\r' && 
                !c.is_control() || c == '\n' || c == '\t'
            })
            .collect::<String>()
            .trim()
            .replace("\x00", "") // Extra safety for null bytes
            .replace("\x01", "") // Remove SOH
            .replace("\x02", "") // Remove STX
    }
    
    /// Execute Python task with common timeout and error handling
    async fn execute_python_task<F, T>(&self, task_fn: F, operation_name: &str) -> MemoryResult<T>
    where
        F: FnOnce() -> MemoryResult<T> + Send + 'static,
        T: Send + 'static,
    {
        let task = tokio::task::spawn_blocking(task_fn);
        
        tokio::time::timeout(self.embedding_timeout, task)
            .await
            .map_err(|_| MemoryError::Embedding(
                format!("{} timed out after {} seconds", operation_name, self.embedding_timeout.as_secs())
            ))?
            .map_err(|e| MemoryError::Embedding(
                format!("{} task failed: {}", operation_name, e)
            ))?
    }
    
    /// Validate embedding dimensions match expected size
    fn validate_embedding_dimension(&self, embedding: &[f32], expected_dim: usize, context: &str) -> MemoryResult<()> {
        if embedding.len() != expected_dim {
            return Err(MemoryError::Embedding(format!(
                "{}: embedding dimension mismatch. Expected {}, got {}",
                context, expected_dim, embedding.len()
            )));
        }
        Ok(())
    }
    
    /// Enhanced cache retrieval with dimension validation
    fn get_cached_embedding_validated(&self, key: &str, expected_dim: usize) -> Option<Arc<Vec<f32>>> {
        if let Some(cached) = self.get_cached_embedding(key) {
            if cached.len() == expected_dim {
                Some(cached)
            } else {
                debug!("Cache hit but dimension mismatch: expected {}, got {}", expected_dim, cached.len());
                // Remove invalid cache entry
                self.embedding_cache.remove(key);
                None
            }
        } else {
            None
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_embeddinggemma_service_creation() {
        // This test requires Python and sentence-transformers to be installed
        if std::env::var("SKIP_PYTHON_TESTS").is_ok() {
            return;
        }
        
        // Test EmbeddingGemma-300M creation (if available)
        let service_result = EmbeddingService::new(
            "google/embeddinggemma-300m",
            "",
            8,
            2048
        ).await;
        
        if service_result.is_ok() {
            let service = service_result.unwrap();
            assert_eq!(service.model_name, "google/embeddinggemma-300m");
            assert_eq!(service.supported_dimensions(), &[768, 512, 256, 128]);
        } else {
            println!("EmbeddingGemma not available - testing with fallback model");
            
            // Fallback test
            let fallback_service = EmbeddingService::new(
                "all-MiniLM-L6-v2",
                "",
                8,
                512
            ).await;
            
            assert!(fallback_service.is_ok(), "Fallback model should work");
        }
    }
    
    #[test]
    fn test_cosine_similarity_comprehensive() {
        use crate::simd_search::cosine_similarity_simd;
        
        // Test identical vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];
        
        assert_eq!(cosine_similarity_simd(&a, &b), 0.0);
        assert_eq!(cosine_similarity_simd(&a, &c), 1.0);
        
        // Test normalized vectors
        let normalized_a = vec![0.6, 0.8];
        let normalized_b = vec![0.8, 0.6];
        let similarity = cosine_similarity_simd(&normalized_a, &normalized_b);
        assert!((similarity - 0.96).abs() < 0.01, "Normalized similarity should be ~0.96");
        
        // Test different length vectors
        let short = vec![1.0];
        let long = vec![1.0, 0.0];
        assert_eq!(cosine_similarity_simd(&short, &long), 0.0, "Different lengths should return 0.0");
        
        // Test zero vectors
        let zero_a = vec![0.0, 0.0];
        let zero_b = vec![0.0, 0.0];
        assert_eq!(cosine_similarity_simd(&zero_a, &zero_b), 0.0, "Zero vectors should return 0.0");
    }
    
    #[test]
    fn test_cache_key_generation_embeddinggemma() {
        let service = create_test_embedding_service();
        
        let key1 = service.generate_cache_key("test text", TaskType::Query, 768);
        let key2 = service.generate_cache_key("test text", TaskType::Document, 768);
        let key3 = service.generate_cache_key("test text", TaskType::Query, 768);
        
        assert_ne!(key1, key2, "Different task types should generate different keys");
        assert_eq!(key1, key3, "Same inputs should generate same keys");
        
        // Test dimension sensitivity
        let key_768 = service.generate_cache_key("test", TaskType::Query, 768);
        let key_512 = service.generate_cache_key("test", TaskType::Query, 512);
        assert_ne!(key_768, key_512, "Different dimensions should generate different keys");
    }
    
    #[test]
    fn test_embeddinggemma_prompt_formatting() {
        let service = create_test_embedding_service();
        
        // Test EmbeddingGemma-specific prompts
        let query_prompt = service.format_with_instruction("search query", TaskType::Query);
        assert_eq!(query_prompt, "task: search result | query: search query");
        
        let doc_prompt = service.format_with_instruction("document content", TaskType::Document);
        assert_eq!(doc_prompt, "title: none | text: document content");
        
        let classification_prompt = service.format_with_instruction("classify this", TaskType::Classification);
        assert_eq!(classification_prompt, "task: classification | text: classify this");
        
        let similarity_prompt = service.format_with_instruction("similar text", TaskType::Similarity);
        assert_eq!(similarity_prompt, "task: similarity | text: similar text");
    }
    
    #[test]
    fn test_matryoshka_dimensions_validation() {
        let service = create_test_embedding_service();
        
        // Test all EmbeddingGemma-300M supported dimensions
        for &dim in &[768, 512, 256, 128] {
            assert!(service.is_dimension_supported(dim), "Should support dimension {}", dim);
        }
        
        // Test unsupported dimensions
        for &dim in &[64, 100, 384, 1024, 1536] {
            assert!(!service.is_dimension_supported(dim), "Should not support dimension {}", dim);
        }
    }
    
    #[test]
    fn test_input_validation_comprehensive() {
        let service = create_test_embedding_service();
        
        // Test text validation
        assert!(service.validate_text_input("").is_ok(), "Empty text should be valid");
        assert!(service.validate_text_input("normal text").is_ok(), "Normal text should be valid");
        
        let long_text = "a".repeat(MAX_TEXT_LENGTH + 1);
        assert!(service.validate_text_input(&long_text).is_err(), "Oversized text should be rejected");
        
        // Test batch validation
        let valid_batch = vec!["text1".to_string(), "text2".to_string()];
        assert!(service.validate_batch_input(&valid_batch).is_ok(), "Valid batch should pass");
        
        let oversized_batch: Vec<String> = (0..MAX_BATCH_SIZE + 1)
            .map(|i| format!("text{}", i))
            .collect();
        assert!(service.validate_batch_input(&oversized_batch).is_err(), "Oversized batch should be rejected");
        
        // Test batch with empty strings
        let empty_string_batch = vec!["valid".to_string(), "".to_string()];
        assert!(service.validate_batch_input(&empty_string_batch).is_err(), "Batch with empty strings should be rejected");
    }
    
    #[test]
    fn test_text_sanitization_comprehensive() {
        let service = create_test_embedding_service();
        
        // Test null byte removal
        let with_null = "text\0with\0nulls";
        let sanitized = service.sanitize_text_input(with_null);
        assert!(!sanitized.contains('\0'), "Null bytes should be removed");
        
        // Test control character removal
        let with_control = "text\x01with\x02control";
        let sanitized_control = service.sanitize_text_input(with_control);
        assert!(!sanitized_control.contains('\x01'), "Control chars should be removed");
        
        // Test whitespace preservation
        let with_whitespace = "  spaced   text  ";
        let sanitized_ws = service.sanitize_text_input(with_whitespace);
        assert_eq!(sanitized_ws, "spaced   text", "Should trim but preserve internal spaces");
        
        // Test newlines and tabs (allowed)
        let with_newlines = "line1\nline2\tindented";
        let sanitized_nl = service.sanitize_text_input(with_newlines);
        assert!(sanitized_nl.contains('\n'), "Newlines should be preserved");
        assert!(sanitized_nl.contains('\t'), "Tabs should be preserved");
    }
    
    #[test]
    fn test_embedding_dimension_validation() {
        let service = create_test_embedding_service();
        
        // Test valid dimensions
        let valid_768 = vec![0.1; 768];
        assert!(service.validate_embedding_dimension(&valid_768, 768, "test").is_ok());
        
        let valid_512 = vec![0.2; 512];
        assert!(service.validate_embedding_dimension(&valid_512, 512, "test").is_ok());
        
        // Test dimension mismatch
        let wrong_dim = vec![0.3; 256];
        assert!(service.validate_embedding_dimension(&wrong_dim, 512, "test").is_err());
        
        // Test empty embedding
        let empty_emb: Vec<f32> = vec![];
        assert!(service.validate_embedding_dimension(&empty_emb, 0, "test").is_ok());
        assert!(service.validate_embedding_dimension(&empty_emb, 768, "test").is_err());
    }
    
    #[test]
    fn test_model_info_embeddinggemma() {
        let service = create_test_embedding_service();
        let info = service.model_info();
        
        // EmbeddingGemma-300M specifications
        assert_eq!(info.dimensions, 768, "Default dimension should be 768");
        assert_eq!(info.max_sequence_length, 2048, "Context window should be 2048");
        assert_eq!(info.vocab_size, 256000, "Vocab size should match EmbeddingGemma");
    }
    
    fn create_test_embedding_service() -> EmbeddingService {
        let python_obj = pyo3::Python::with_gil(|py| py.None());
        
        EmbeddingService {
            python_module: Arc::new(Mutex::new(python_obj)),
            model_name: "google/embeddinggemma-300m".to_string(),
            config: EmbeddingConfig::default(), // Add missing config field
            embedding_cache: Arc::new(dashmap::DashMap::new()),
            use_instructions: true,
            matryoshka_dims: vec![768, 512, 256, 128], // EmbeddingGemma-300M dimensions
            embedding_timeout: Duration::from_secs(30),
            max_text_length: MAX_TEXT_LENGTH,
            max_batch_size: MAX_BATCH_SIZE,
        }
    }
}

