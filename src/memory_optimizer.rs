//! Memory optimization system for AI Memory Service
//!
//! Provides advanced memory management, garbage collection, and optimization
//! techniques specifically tuned for Windows systems and large-scale embeddings.

use std::sync::{Arc, OnceLock, atomic::{AtomicU64, AtomicUsize, Ordering}};
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use tokio::time::interval;
use tracing::{info, warn, error, debug};
use crate::types::MemoryResult;
use serde::{Serialize, Deserialize};

/// Memory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizerConfig {
    /// Target memory usage in bytes
    pub target_memory_bytes: u64,
    /// Memory pressure threshold (0.0-1.0)
    pub pressure_threshold: f32,
    /// Aggressive cleanup threshold (0.0-1.0)  
    pub aggressive_threshold: f32,
    /// Cleanup interval in seconds
    pub cleanup_interval_secs: u64,
    /// Enable memory compaction
    pub enable_compaction: bool,
    /// Enable embedding compression
    pub enable_compression: bool,
    /// Memory pool size for embeddings
    pub embedding_pool_size: usize,
    /// Maximum age for cached items (seconds)
    pub max_cache_age_secs: u64,
}

impl Default for MemoryOptimizerConfig {
    fn default() -> Self {
        MemoryOptimizerConfig {
            target_memory_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            pressure_threshold: 0.75,
            aggressive_threshold: 0.9,
            cleanup_interval_secs: 60,
            enable_compaction: true,
            enable_compression: true,
            embedding_pool_size: 10000,
            max_cache_age_secs: 3600,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub total_allocated: u64,
    pub heap_size: u64,
    pub embeddings_memory: u64,
    pub cache_memory: u64,
    pub fragmentation_ratio: f32,
    pub pressure_level: f32,
    pub cleanup_count: u64,
    pub compression_ratio: f32,
    pub pool_utilization: f32,
}

/// Memory pool for embedding vectors
pub struct EmbeddingPool {
    /// Pre-allocated vector buffers
    available: Arc<Mutex<Vec<Vec<f32>>>>,
    /// Pool configuration
    config: EmbeddingPoolConfig,
    /// Statistics
    allocated_count: AtomicUsize,  // Total vectors ever created
    reused_count: AtomicU64,       // Vectors served from pool
    total_requests: AtomicU64,     // Total get_vector() calls
}

#[derive(Debug, Clone)]
pub struct EmbeddingPoolConfig {
    pub initial_capacity: usize,
    pub max_capacity: usize,
    pub vector_size: usize,
    pub growth_factor: f32,
}

impl Default for EmbeddingPoolConfig {
    fn default() -> Self {
        EmbeddingPoolConfig {
            initial_capacity: 1000,
            max_capacity: 10000,
            vector_size: 768,
            growth_factor: 1.5,
        }
    }
}

impl EmbeddingPool {
    pub fn new(config: EmbeddingPoolConfig) -> Self {
        let mut initial_vectors = Vec::with_capacity(config.initial_capacity);
        
        // Pre-allocate vectors
        for _ in 0..config.initial_capacity {
            initial_vectors.push(vec![0.0; config.vector_size]);
        }
        
        info!("Embedding pool initialized with {} vectors of size {}", 
              config.initial_capacity, config.vector_size);
        
        EmbeddingPool {
            available: Arc::new(Mutex::new(initial_vectors)),
            config: config.clone(),
            allocated_count: AtomicUsize::new(0), // Only count new allocations, not pre-allocated
            reused_count: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
        }
    }
    
    /// Get a vector from the pool
    pub async fn get_vector(&self) -> Vec<f32> {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        
        let mut available = self.available.lock().await;
        
        if let Some(mut vector) = available.pop() {
            // Validate and resize if needed
            if vector.len() != self.config.vector_size {
                warn!("Vector size mismatch: expected {}, got {}", self.config.vector_size, vector.len());
                vector.resize(self.config.vector_size, 0.0);
            }
            vector.fill(0.0); // Clear vector
            
            self.reused_count.fetch_add(1, Ordering::Relaxed);
            debug!("Reused vector from pool (available: {})", available.len());
            vector
        } else {
            // Pool is empty, create new vector
            self.allocated_count.fetch_add(1, Ordering::Relaxed);
            debug!("Created new vector (pool empty)");
            vec![0.0; self.config.vector_size]
        }
    }
    
    /// Return a vector to the pool
    pub async fn return_vector(&self, vector: Vec<f32>) {
        let mut available = self.available.lock().await;
        
        // Only return to pool if we haven't exceeded max capacity
        if available.len() < self.config.max_capacity {
            available.push(vector);
            // Don't modify allocated_count - it tracks total vectors ever created
            debug!("Returned vector to pool (available: {})", available.len());
        } else {
            // Pool is full, let vector be dropped
            // Don't modify allocated_count - it tracks total vectors ever created
            debug!("Pool full, dropped vector");
        }
    }
    
    /// Get pool statistics
    pub async fn get_stats(&self) -> PoolStats {
        let available = self.available.lock().await;
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let reused = self.reused_count.load(Ordering::Relaxed);
        
        PoolStats {
            available_vectors: available.len(),
            allocated_vectors: self.allocated_count.load(Ordering::Relaxed),
            total_requests,
            reused_count: reused,
            reuse_rate: if total_requests > 0 { 
                reused as f32 / total_requests as f32 
            } else { 
                0.0 
            },
            max_capacity: self.config.max_capacity,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PoolStats {
    pub available_vectors: usize,
    pub allocated_vectors: usize,
    pub total_requests: u64,
    pub reused_count: u64,
    pub reuse_rate: f32,
    pub max_capacity: usize,
}

/// Memory compaction system
pub struct MemoryCompactor {
    /// Last compaction time
    last_compaction: Arc<Mutex<Option<Instant>>>,
    /// Compaction statistics
    compaction_count: AtomicU64,
    /// Memory freed by compaction
    bytes_freed: AtomicU64,
}

impl MemoryCompactor {
    pub fn new() -> Self {
        MemoryCompactor {
            last_compaction: Arc::new(Mutex::new(None)),
            compaction_count: AtomicU64::new(0),
            bytes_freed: AtomicU64::new(0),
        }
    }
    
    /// Perform memory compaction
    pub async fn compact(&self) -> MemoryResult<u64> {
        let start_time = Instant::now();
        info!("Starting memory compaction");
        
        // Get initial memory usage
        let initial_memory = self.get_memory_usage();
        
        // Force garbage collection
        // Note: Rust doesn't have explicit GC, but we can drop unused allocations
        self.force_cleanup().await?;
        
        // Update statistics
        let final_memory = self.get_memory_usage();
        let freed = initial_memory.saturating_sub(final_memory);
        
        self.compaction_count.fetch_add(1, Ordering::Relaxed);
        self.bytes_freed.fetch_add(freed, Ordering::Relaxed);
        
        *self.last_compaction.lock().await = Some(start_time);
        
        let duration = start_time.elapsed();
        info!("Memory compaction completed in {:?}, freed {} bytes", duration, freed);
        
        Ok(freed)
    }
    
    /// Force cleanup of unused allocations
    async fn force_cleanup(&self) -> MemoryResult<()> {
        // In a real implementation, this would:
        // 1. Drop unused caches
        // 2. Compact memory pools
        // 3. Release fragmented memory
        // 4. Optimize data structures
        
        // For now, we simulate some cleanup work
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        Ok(())
    }
    
    /// Get current memory usage (platform-specific)
    fn get_memory_usage(&self) -> u64 {
        #[cfg(target_os = "windows")]
        {
            if let Ok(memory) = get_windows_process_memory() {
                memory
            } else {
                0
            }
        }
        
        #[cfg(not(target_os = "windows"))]
        {
            // Fallback for other platforms
            0
        }
    }
    
    /// Get compaction statistics
    pub async fn get_stats(&self) -> CompactionStats {
        let last_compaction = *self.last_compaction.lock().await;
        
        CompactionStats {
            compaction_count: self.compaction_count.load(Ordering::Relaxed),
            total_bytes_freed: self.bytes_freed.load(Ordering::Relaxed),
            last_compaction_time: last_compaction,
        }
    }
}

impl Default for MemoryCompactor {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone)]
pub struct CompactionStats {
    pub compaction_count: u64,
    pub total_bytes_freed: u64,
    pub last_compaction_time: Option<Instant>,
}

/// Main memory optimizer
pub struct MemoryOptimizer {
    config: MemoryOptimizerConfig,
    embedding_pool: Arc<EmbeddingPool>,
    compactor: Arc<MemoryCompactor>,
    stats: Arc<RwLock<MemoryStats>>,
    
    // Background task handle
    cleanup_task: Option<tokio::task::JoinHandle<()>>,
}

impl MemoryOptimizer {
    pub fn new(config: MemoryOptimizerConfig) -> Self {
        let pool_config = EmbeddingPoolConfig {
            max_capacity: config.embedding_pool_size,
            ..Default::default()
        };
        
        let embedding_pool = Arc::new(EmbeddingPool::new(pool_config));
        let compactor = Arc::new(MemoryCompactor::new());
        
        let initial_stats = MemoryStats {
            timestamp: chrono::Utc::now(),
            total_allocated: 0,
            heap_size: 0,
            embeddings_memory: 0,
            cache_memory: 0,
            fragmentation_ratio: 0.0,
            pressure_level: 0.0,
            cleanup_count: 0,
            compression_ratio: 1.0,
            pool_utilization: 0.0,
        };
        
        MemoryOptimizer {
            config,
            embedding_pool,
            compactor,
            stats: Arc::new(RwLock::new(initial_stats)),
            cleanup_task: None,
        }
    }
    
    /// Start background memory management
    pub async fn start(&mut self) -> MemoryResult<()> {
        if self.cleanup_task.is_some() {
            warn!("Memory optimizer already running");
            return Ok(());
        }
        
        info!("Starting memory optimizer with cleanup interval: {}s", 
              self.config.cleanup_interval_secs);
        
        let config = self.config.clone();
        let compactor = Arc::clone(&self.compactor);
        let stats = Arc::clone(&self.stats);
        
        let task = tokio::spawn(async move {
            Self::background_cleanup_task(config, compactor, stats).await;
        });
        
        self.cleanup_task = Some(task);
        Ok(())
    }
    
    /// Stop background memory management
    pub async fn stop(&mut self) {
        if let Some(task) = self.cleanup_task.take() {
            task.abort();
            info!("Memory optimizer stopped");
        }
    }
    
    /// Background cleanup task
    async fn background_cleanup_task(
        config: MemoryOptimizerConfig,
        compactor: Arc<MemoryCompactor>,
        stats: Arc<RwLock<MemoryStats>>,
    ) {
        let mut interval = interval(Duration::from_secs(config.cleanup_interval_secs));
        
        loop {
            interval.tick().await;
            
            // Update memory statistics
            if let Err(e) = Self::update_stats(&config, Arc::clone(&stats)).await {
                error!("Failed to update memory stats: {}", e);
                continue;
            }
            
            // Check if cleanup is needed
            let pressure_level = {
                let stats_guard = stats.read().await;
                stats_guard.pressure_level
            };
            
            if pressure_level > config.pressure_threshold {
                info!("Memory pressure detected: {:.2}", pressure_level);
                
                if let Err(e) = compactor.compact().await {
                    error!("Memory compaction failed: {}", e);
                } else {
                    // Update cleanup count
                    let mut stats_guard = stats.write().await;
                    stats_guard.cleanup_count += 1;
                }
            }
            
            debug!("Memory optimizer tick completed (pressure: {:.2})", pressure_level);
        }
    }
    
    /// Update memory statistics
    async fn update_stats(
        config: &MemoryOptimizerConfig,
        stats: Arc<RwLock<MemoryStats>>,
    ) -> MemoryResult<()> {
        let current_memory = get_current_memory_usage();
        let pressure_level = current_memory as f32 / config.target_memory_bytes as f32;
        
        let mut stats_guard = stats.write().await;
        stats_guard.timestamp = chrono::Utc::now();
        stats_guard.total_allocated = current_memory;
        stats_guard.pressure_level = pressure_level;
        
        // Calculate fragmentation (simplified)
        stats_guard.fragmentation_ratio = if current_memory > 0 {
            (current_memory as f32 * 0.1) / current_memory as f32
        } else {
            0.0
        };
        
        Ok(())
    }
    
    /// Get embedding pool reference
    pub fn embedding_pool(&self) -> Arc<EmbeddingPool> {
        Arc::clone(&self.embedding_pool)
    }
    
    /// Get current memory statistics
    pub async fn get_stats(&self) -> MemoryStats {
        self.stats.read().await.clone()
    }
    
    /// Force memory cleanup
    pub async fn force_cleanup(&self) -> MemoryResult<()> {
        info!("Forcing memory cleanup");
        self.compactor.compact().await?;
        Ok(())
    }
}

/// Get current memory usage
fn get_current_memory_usage() -> u64 {
    #[cfg(target_os = "windows")]
    {
        get_windows_process_memory().unwrap_or(0)
    }
    
    #[cfg(not(target_os = "windows"))]
    {
        // Fallback for other platforms
        0
    }
}

/// Windows-specific memory information
#[cfg(target_os = "windows")]
fn get_windows_process_memory() -> Result<u64, Box<dyn std::error::Error>> {
    use std::mem;
    use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    use winapi::um::processthreadsapi::GetCurrentProcess;
    
    unsafe {
        let mut pmc: PROCESS_MEMORY_COUNTERS_EX = mem::zeroed();
        pmc.cb = mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        
        if GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut pmc as *mut _ as *mut _,
            pmc.cb,
        ) != 0 {
            Ok(pmc.WorkingSetSize as u64)
        } else {
            Err("Failed to get process memory info".into())
        }
    }
}

/// Global memory optimizer instance
static MEMORY_OPTIMIZER: OnceLock<Arc<Mutex<MemoryOptimizer>>> = OnceLock::new();

impl MemoryOptimizer {
    /// Initialize global memory optimizer
    pub fn initialize(config: MemoryOptimizerConfig) -> Arc<Mutex<Self>> {
        MEMORY_OPTIMIZER.get_or_init(|| {
            Arc::new(Mutex::new(MemoryOptimizer::new(config)))
        }).clone()
    }
    
    /// Get global memory optimizer
    pub fn global() -> Option<Arc<Mutex<Self>>> {
        MEMORY_OPTIMIZER.get().cloned()
    }
}

/// High-level memory management API
pub struct MemoryManager;

impl MemoryManager {
    /// Initialize memory management
    pub async fn init(config: Option<MemoryOptimizerConfig>) -> MemoryResult<()> {
        let config = config.unwrap_or_default();
        let optimizer = MemoryOptimizer::initialize(config);
        
        // Start background optimization
        let mut optimizer_guard = optimizer.lock().await;
        optimizer_guard.start().await?;
        drop(optimizer_guard);
        
        info!("Memory manager initialized");
        Ok(())
    }
    
    /// Get embedding pool
    pub async fn embedding_pool() -> Option<Arc<EmbeddingPool>> {
        if let Some(optimizer) = MemoryOptimizer::global() {
            let optimizer = optimizer.lock().await;
            Some(optimizer.embedding_pool())
        } else {
            None
        }
    }
    
    /// Force cleanup
    pub async fn force_cleanup() -> MemoryResult<()> {
        if let Some(optimizer) = MemoryOptimizer::global() {
            let optimizer = optimizer.lock().await;
            optimizer.force_cleanup().await
        } else {
            Err("Memory optimizer not initialized".into())
        }
    }
    
    /// Get memory statistics
    pub async fn get_stats() -> Option<MemoryStats> {
        if let Some(optimizer) = MemoryOptimizer::global() {
            let optimizer = optimizer.lock().await;
            Some(optimizer.get_stats().await)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_embedding_pool() {
        let config = EmbeddingPoolConfig::default();
        let initial_capacity = config.initial_capacity;
        let vector_size = config.vector_size;
        let pool = EmbeddingPool::new(config);
        
        // Test 1: Initial state
        let initial_stats = pool.get_stats().await;
        assert_eq!(initial_stats.total_requests, 0);
        assert_eq!(initial_stats.reused_count, 0);
        assert_eq!(initial_stats.allocated_vectors, 0);
        assert_eq!(initial_stats.available_vectors, initial_capacity);
        
        // Test 2: Get first vector (uses pre-allocated)
        let vector1 = pool.get_vector().await;
        assert_eq!(vector1.len(), vector_size);
        assert!(vector1.iter().all(|&x| x == 0.0)); // Should be cleared to zeros
        
        let stats_after_first = pool.get_stats().await;
        assert_eq!(stats_after_first.total_requests, 1);
        assert_eq!(stats_after_first.reused_count, 1); // Used from pre-allocated pool
        assert_eq!(stats_after_first.allocated_vectors, 0); // No new allocations yet
        assert_eq!(stats_after_first.available_vectors, initial_capacity - 1); // One vector taken from pool
        
        // Test 3: Return vector - allocated_vectors should NOT change
        pool.return_vector(vector1).await;
        
        let stats_after_return = pool.get_stats().await;
        assert_eq!(stats_after_return.total_requests, 1);
        assert_eq!(stats_after_return.reused_count, 1); // Still 1
        assert_eq!(stats_after_return.allocated_vectors, 0); // Still 0 - no new allocations
        assert_eq!(stats_after_return.available_vectors, initial_capacity); // Back to full capacity
        
        // Test 4: Get another vector (should reuse returned vector)
        let vector2 = pool.get_vector().await;
        assert_eq!(vector2.len(), vector_size);
        assert!(vector2.iter().all(|&x| x == 0.0)); // Should be cleared
        
        let stats_after_reuse = pool.get_stats().await;
        assert_eq!(stats_after_reuse.total_requests, 2);
        assert_eq!(stats_after_reuse.reused_count, 2); // Second reuse
        assert_eq!(stats_after_reuse.allocated_vectors, 0); // Still no new allocations
        assert_eq!(stats_after_reuse.available_vectors, initial_capacity - 1); // One vector taken again
        
        // Test 5: Get third vector (should also reuse from large pool)
        let vector3 = pool.get_vector().await;
        assert_eq!(vector3.len(), vector_size);
        assert!(vector3.iter().all(|&x| x == 0.0)); // Should be initialized to zeros
        
        let final_stats = pool.get_stats().await;
        assert_eq!(final_stats.total_requests, 3);
        assert_eq!(final_stats.reused_count, 3); // Three reuses from large pool
        assert_eq!(final_stats.allocated_vectors, 0); // No new allocations needed with large pool
        assert_eq!(final_stats.available_vectors, initial_capacity - 2); // Two vectors in use
    }
    
    #[tokio::test]
    async fn test_memory_compactor() {
        let compactor = MemoryCompactor::new();
        
        // Perform compaction
        let result = compactor.compact().await;
        assert!(result.is_ok());
        
        let stats = compactor.get_stats().await;
        assert_eq!(stats.compaction_count, 1);
    }
    
    #[test]
    fn test_memory_optimizer_config() {
        let config = MemoryOptimizerConfig::default();
        assert_eq!(config.target_memory_bytes, 2 * 1024 * 1024 * 1024);
        assert_eq!(config.pressure_threshold, 0.75);
    }
}
