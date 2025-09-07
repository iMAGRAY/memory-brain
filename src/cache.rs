//! High-performance hierarchical caching system for AI memory service
//! 
//! Implements multi-level caching with LRU eviction, TTL expiration,
//! and adaptive cache sizing based on access patterns.

use crate::types::{MemoryCell, RecalledMemory};
use dashmap::DashMap;
use moka::future::Cache as MokaCache;
use parking_lot::RwLock;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{Duration, Instant};
use tracing::{debug, info, trace};
use uuid::Uuid;

/// Hierarchical cache system with multiple levels
pub struct CacheSystem {
    /// L1: Hot cache for frequently accessed memories (in-memory)
    l1_cache: Arc<L1Cache>,
    /// L2: Warm cache for recent queries (moka with TTL)
    l2_cache: Arc<L2Cache>,
    /// L3: Cold cache for embeddings (compressed)
    l3_cache: Arc<L3Cache>,
    /// Cache statistics tracker
    stats: Arc<RwLock<CacheStats>>,
}

/// L1 Hot cache - ultra-fast access for most frequent items
struct L1Cache {
    data: DashMap<Uuid, Arc<CachedMemory>>,
    max_size: usize,
    access_counter: DashMap<Uuid, AtomicU64>,
    lru_tracker: Arc<RwLock<BinaryHeap<LruEntry>>>,
}

/// LRU tracking entry for efficient eviction
#[derive(Clone)]
struct LruEntry {
    id: Uuid,
    last_access: Instant,
}

impl PartialEq for LruEntry {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for LruEntry {}

impl Ord for LruEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (oldest first)
        other.last_access.cmp(&self.last_access)
    }
}

impl PartialOrd for LruEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// L2 Warm cache - recent queries with TTL
struct L2Cache {
    query_cache: MokaCache<String, Arc<RecalledMemory>>,
    memory_cache: MokaCache<Uuid, Arc<MemoryCell>>,
}

/// L3 Cold cache - compressed embeddings
struct L3Cache {
    embeddings: DashMap<Uuid, CompressedEmbedding>,
}

/// Cached memory with access metadata
struct CachedMemory {
    memory: Arc<MemoryCell>,
    last_access: RwLock<Instant>,
    access_count: AtomicU64,
}

/// Compressed embedding for cold storage
struct CompressedEmbedding {
    data: Vec<u8>,
    original_dims: usize,
}


/// Cache statistics for monitoring
#[derive(Default)]
pub struct CacheStats {
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub l3_hits: u64,
    pub l3_misses: u64,
    pub total_evictions: u64,
    pub compression_saves: u64,
    pub avg_hit_rate: f32,
}

/// Cache configuration
pub struct CacheConfig {
    pub l1_max_size: usize,
    pub l2_max_size: usize,
    pub ttl_seconds: u64,
    pub importance_threshold_l1: f32,
    pub importance_threshold_l2: f32,
    pub access_frequency_l1: u32,
    pub access_frequency_l2: u32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_max_size: 1000,
            l2_max_size: 10000,
            ttl_seconds: 3600,
            importance_threshold_l1: 0.8,
            importance_threshold_l2: 0.5,
            access_frequency_l1: 10,
            access_frequency_l2: 3,
        }
    }
}

impl CacheSystem {
    /// Create new hierarchical cache system with config
    pub fn new(config: CacheConfig) -> Self {
        let l1_cache = Arc::new(L1Cache::new(config.l1_max_size));
        let l2_cache = Arc::new(L2Cache::new(config.l2_max_size, Duration::from_secs(config.ttl_seconds)));
        let l3_cache = Arc::new(L3Cache::new());

        Self {
            l1_cache,
            l2_cache,
            l3_cache,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Create new hierarchical cache system with defaults

    /// Get memory from cache hierarchy
    pub async fn get_memory(&self, id: &Uuid) -> Option<Arc<MemoryCell>> {
        // Try L1 first (hottest)
        if let Some(cached) = self.l1_cache.get(id) {
            self.stats.write().l1_hits += 1;
            // No need to update_access, get() already does it
            return Some(cached.memory.clone());
        }
        self.stats.write().l1_misses += 1;

        // Try L2 (warm)
        if let Some(memory) = self.l2_cache.get_memory(id).await {
            self.stats.write().l2_hits += 1;
            // Promote to L1 if frequently accessed
            self.maybe_promote_to_l1(id, &memory).await;
            return Some(memory);
        }
        self.stats.write().l2_misses += 1;

        // L3 only has embeddings, not full memories
        None
    }

    /// Cache memory at appropriate level
    pub async fn put_memory(&self, memory: MemoryCell) {
        let id = memory.id;
        let memory_arc = Arc::new(memory.clone());

        // Determine cache level based on importance and access pattern
        let cache_level = self.determine_cache_level(&memory);

        match cache_level {
            1 => {
                self.l1_cache.put(id, memory_arc.clone());
                // Also put in L2 for redundancy
                self.l2_cache.put_memory(id, memory_arc).await;
            }
            2 => {
                self.l2_cache.put_memory(id, memory_arc).await;
            }
            3 => {
                // Only cache embedding in L3
                if !memory.embedding.is_empty() {
                    self.l3_cache.put_embedding(id, &memory.embedding);
                }
            }
            _ => {}
        }
    }

    /// Get cached query result
    pub async fn get_query(&self, query_hash: &str) -> Option<Arc<RecalledMemory>> {
        self.l2_cache.get_query(query_hash).await
    }

    /// Cache query result
    pub async fn put_query(&self, query_hash: String, result: RecalledMemory) {
        self.l2_cache.put_query(query_hash, Arc::new(result)).await;
    }

    /// Get compressed embedding from L3
    pub fn get_embedding(&self, id: &Uuid) -> Option<Vec<f32>> {
        if let Some(compressed) = self.l3_cache.get_embedding(id) {
            self.stats.write().l3_hits += 1;
            Some(compressed)
        } else {
            self.stats.write().l3_misses += 1;
            None
        }
    }

    /// Clear all caches
    pub async fn clear_all(&self) {
        self.l1_cache.clear();
        self.l2_cache.clear().await;
        self.l3_cache.clear();
        *self.stats.write() = CacheStats::default();
        info!("All cache levels cleared");
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let stats = self.stats.read();
        let mut result = stats.clone();
        
        // Calculate overall hit rate
        let total_hits = stats.l1_hits + stats.l2_hits + stats.l3_hits;
        let total_accesses = total_hits + stats.l1_misses + stats.l2_misses + stats.l3_misses;
        
        if total_accesses > 0 {
            result.avg_hit_rate = total_hits as f32 / total_accesses as f32;
        }
        
        result
    }

    /// Determine optimal cache level based on memory characteristics
    fn determine_cache_level(&self, memory: &MemoryCell) -> u8 {
        // L1: High importance, frequently accessed
        if memory.importance > 0.8 || memory.access_frequency > 10 {
            return 1;
        }
        
        // L2: Medium importance, recent access
        if memory.importance > 0.5 || memory.access_frequency > 3 {
            return 2;
        }
        
        // L3: Low importance, just embeddings
        3
    }

    /// Maybe promote memory from L2 to L1
    async fn maybe_promote_to_l1(&self, id: &Uuid, memory: &Arc<MemoryCell>) {
        let access_count = self.l1_cache.get_access_count(id);
        if access_count > 5 {
            debug!("Promoting memory {} to L1 cache", id);
            self.l1_cache.put(*id, memory.clone());
        }
    }
}

impl L1Cache {
    fn new(max_size: usize) -> Self {
        Self {
            data: DashMap::with_capacity(max_size),
            max_size,
            access_counter: DashMap::new(),
            lru_tracker: Arc::new(RwLock::new(BinaryHeap::with_capacity(max_size))),
        }
    }

    fn get(&self, id: &Uuid) -> Option<Arc<CachedMemory>> {
        self.data.get(id).map(|entry| {
            let cached = entry.clone();
            let now = Instant::now();
            *cached.last_access.write() = now;
            cached.access_count.fetch_add(1, AtomicOrdering::Relaxed);
            
            // Update LRU tracker
            self.lru_tracker.write().push(LruEntry {
                id: *id,
                last_access: now,
            });
            
            cached
        })
    }

    fn put(&self, id: Uuid, memory: Arc<MemoryCell>) {
        // Evict LRU if at capacity
        if self.data.len() >= self.max_size {
            self.evict_lru();
        }

        let now = Instant::now();
        let cached = Arc::new(CachedMemory {
            memory,
            last_access: RwLock::new(now),
            access_count: AtomicU64::new(1),
        });

        self.data.insert(id, cached);
        self.lru_tracker.write().push(LruEntry {
            id,
            last_access: now,
        });
        trace!("Added memory {} to L1 cache", id);
    }


    fn get_access_count(&self, id: &Uuid) -> u64 {
        self.access_counter
            .get(id)
            .map(|c| c.load(AtomicOrdering::Relaxed))
            .unwrap_or(0)
    }

    fn evict_lru(&self) {
        let mut lru_tracker = self.lru_tracker.write();
        
        // Clean up stale entries and find valid LRU
        let mut entries_to_check = Vec::new();
        while let Some(entry) = lru_tracker.pop() {
            if self.data.contains_key(&entry.id) {
                // Found valid LRU entry
                self.data.remove(&entry.id);
                self.access_counter.remove(&entry.id);
                trace!("Evicted {} from L1 cache", entry.id);
                
                // Put back other entries
                for e in entries_to_check {
                    lru_tracker.push(e);
                }
                return;
            }
            entries_to_check.push(entry);
            
            // Limit search to avoid excessive processing
            if entries_to_check.len() > 100 {
                break;
            }
        }
        
        // Fallback: if heap is corrupted, rebuild it
        if self.data.len() >= self.max_size {
            self.rebuild_lru_tracker();
            // Try again with rebuilt tracker
            if let Some(entry) = lru_tracker.pop() {
                self.data.remove(&entry.id);
                self.access_counter.remove(&entry.id);
                trace!("Evicted {} from L1 cache after rebuild", entry.id);
            }
        }
    }
    
    fn rebuild_lru_tracker(&self) {
        let mut lru_tracker = self.lru_tracker.write();
        lru_tracker.clear();
        
        for entry in self.data.iter() {
            lru_tracker.push(LruEntry {
                id: *entry.key(),
                last_access: *entry.value().last_access.read(),
            });
        }
    }

    fn clear(&self) {
        self.data.clear();
        self.access_counter.clear();
        self.lru_tracker.write().clear();
    }
}

impl L2Cache {
    fn new(max_size: usize, ttl: Duration) -> Self {
        let query_cache = MokaCache::builder()
            .max_capacity(max_size as u64 / 2)
            .time_to_live(ttl)
            .build();

        let memory_cache = MokaCache::builder()
            .max_capacity(max_size as u64)
            .time_to_live(ttl * 2) // Memories live longer than queries
            .build();

        Self {
            query_cache,
            memory_cache,
        }
    }

    async fn get_memory(&self, id: &Uuid) -> Option<Arc<MemoryCell>> {
        self.memory_cache.get(id).await
    }

    async fn put_memory(&self, id: Uuid, memory: Arc<MemoryCell>) {
        self.memory_cache.insert(id, memory).await;
    }

    async fn get_query(&self, hash: &str) -> Option<Arc<RecalledMemory>> {
        self.query_cache.get(hash).await
    }

    async fn put_query(&self, hash: String, result: Arc<RecalledMemory>) {
        self.query_cache.insert(hash, result).await;
    }

    async fn clear(&self) {
        self.query_cache.invalidate_all();
        self.memory_cache.invalidate_all();
    }
}

impl L3Cache {
    fn new() -> Self {
        Self {
            embeddings: DashMap::new(),
        }
    }

    fn put_embedding(&self, id: Uuid, embedding: &[f32]) {
        let compressed = self.compress_embedding(embedding);
        self.embeddings.insert(id, compressed);
    }

    fn get_embedding(&self, id: &Uuid) -> Option<Vec<f32>> {
        self.embeddings.get(id).map(|compressed| {
            self.decompress_embedding(&compressed)
        })
    }

    fn compress_embedding(&self, embedding: &[f32]) -> CompressedEmbedding {
        // Simple quantization for now (f32 -> u8)
        let quantized: Vec<u8> = embedding
            .iter()
            .map(|&v| ((v + 1.0) * 127.5).clamp(0.0, 255.0) as u8)
            .collect();

        // Further compress with LZ4
        let compressed = lz4_flex::compress_prepend_size(&quantized);

        CompressedEmbedding {
            data: compressed,
            original_dims: embedding.len(),
        }
    }

    fn decompress_embedding(&self, compressed: &CompressedEmbedding) -> Vec<f32> {
        // Decompress LZ4
        let quantized = lz4_flex::decompress_size_prepended(&compressed.data)
            .unwrap_or_else(|_| vec![128; compressed.original_dims]);

        // Dequantize u8 -> f32
        quantized
            .iter()
            .map(|&v| (v as f32 / 127.5) - 1.0)
            .collect()
    }

    fn clear(&self) {
        self.embeddings.clear();
    }
}

impl Clone for CacheStats {
    fn clone(&self) -> Self {
        Self {
            l1_hits: self.l1_hits,
            l1_misses: self.l1_misses,
            l2_hits: self.l2_hits,
            l2_misses: self.l2_misses,
            l3_hits: self.l3_hits,
            l3_misses: self.l3_misses,
            total_evictions: self.total_evictions,
            compression_saves: self.compression_saves,
            avg_hit_rate: self.avg_hit_rate,
        }
    }
}