//! Embedded database implementation as Neo4j fallback
//! Uses SQLite with JSON support for graph-like operations

use crate::types::{MemoryCell, MemoryError, MemoryResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use tracing::{debug, info, warn, error};

/// Embedded database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedDbConfig {
    pub database_path: PathBuf,
    pub max_connections: u32,
    pub cache_size_mb: usize,
    pub enable_wal_mode: bool,
    pub backup_interval_hours: Option<u64>,
}

impl Default for EmbeddedDbConfig {
    fn default() -> Self {
        Self {
            database_path: PathBuf::from("./data/memory.db"),
            max_connections: 10,
            cache_size_mb: 64,
            enable_wal_mode: true,
            backup_interval_hours: Some(24),
        }
    }
}

/// Embedded storage using SQLite for local deployment
pub struct EmbeddedStorage {
    config: EmbeddedDbConfig,
    connection_pool: Arc<RwLock<Option<rusqlite::Connection>>>,
    in_memory_cache: Arc<RwLock<HashMap<Uuid, MemoryCell>>>,
    vector_index: Arc<RwLock<VectorIndex>>,
}

/// Simple vector index for similarity search
#[derive(Debug, Default)]
struct VectorIndex {
    vectors: HashMap<Uuid, Vec<f32>>,
    dimension: Option<usize>,
}

impl VectorIndex {
    fn add_vector(&mut self, id: Uuid, vector: Vec<f32>) -> MemoryResult<()> {
        if let Some(dim) = self.dimension {
            if vector.len() != dim {
                return Err(MemoryError::ValidationError(
                    format!("Vector dimension mismatch: expected {}, got {}", dim, vector.len())
                ));
            }
        } else {
            self.dimension = Some(vector.len());
        }
        
        self.vectors.insert(id, vector);
        Ok(())
    }

    fn find_similar(&self, query_vector: &[f32], limit: usize) -> Vec<(Uuid, f32)> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        let mut similarities: Vec<(Uuid, f32)> = self.vectors
            .iter()
            .map(|(id, vector)| {
                let similarity = crate::simd_search::cosine_similarity_simd(query_vector, vector);
                (*id, similarity)
            })
            .collect();

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(limit);
        
        similarities
    }

    fn remove_vector(&mut self, id: &Uuid) {
        self.vectors.remove(id);
    }
}

impl EmbeddedStorage {
    /// Create new embedded storage instance
    pub async fn new(config: EmbeddedDbConfig) -> MemoryResult<Self> {
        info!("Initializing embedded storage at: {:?}", config.database_path);

        // Ensure database directory exists
        if let Some(parent) = config.database_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                MemoryError::StorageError(format!("Failed to create database directory: {}", e))
            })?;
        }

        let storage = Self {
            config,
            connection_pool: Arc::new(RwLock::new(None)),
            in_memory_cache: Arc::new(RwLock::new(HashMap::new())),
            vector_index: Arc::new(RwLock::new(VectorIndex::default())),
        };

        storage.initialize_database().await?;
        
        info!("Embedded storage initialized successfully");
        Ok(storage)
    }

    /// Initialize SQLite database with required schema
    async fn initialize_database(&self) -> MemoryResult<()> {
        let connection = rusqlite::Connection::open(&self.config.database_path)
            .map_err(|e| MemoryError::StorageError(format!("Failed to open database: {}", e)))?;

        // Enable WAL mode for better concurrency
        if self.config.enable_wal_mode {
            connection.execute("PRAGMA journal_mode = WAL", [])
                .map_err(|e| MemoryError::StorageError(format!("Failed to enable WAL mode: {}", e)))?;
        }

        // Set cache size
        let cache_size_kb = (self.config.cache_size_mb * 1024) as i32;
        connection.execute(&format!("PRAGMA cache_size = -{}", cache_size_kb), [])
            .map_err(|e| MemoryError::StorageError(format!("Failed to set cache size: {}", e)))?;

        // Create schema
        connection.execute_batch(r#"
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                summary TEXT NOT NULL,
                tags TEXT NOT NULL, -- JSON array
                embedding BLOB,     -- Serialized f32 vector
                memory_type TEXT NOT NULL, -- JSON object
                importance REAL NOT NULL,
                access_frequency INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                context_path TEXT NOT NULL,
                metadata TEXT NOT NULL -- JSON object
            );

            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_context_path ON memories(context_path);

            -- Full-text search support
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED,
                content,
                summary,
                tags,
                context_path,
                content='memories',
                content_rowid='rowid'
            );

            -- Triggers to keep FTS table synchronized
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(id, content, summary, tags, context_path)
                VALUES (new.id, new.content, new.summary, new.tags, new.context_path);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                DELETE FROM memories_fts WHERE id = old.id;
            END;

            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                DELETE FROM memories_fts WHERE id = old.id;
                INSERT INTO memories_fts(id, content, summary, tags, context_path)
                VALUES (new.id, new.content, new.summary, new.tags, new.context_path);
            END;
        "#)
        .map_err(|e| MemoryError::StorageError(format!("Failed to create schema: {}", e)))?;

        // Store connection in pool
        let mut pool = self.connection_pool.write().await;
        *pool = Some(connection);

        Ok(())
    }

    /// Store a memory cell
    pub async fn store_memory(&self, memory: &MemoryCell) -> MemoryResult<()> {
        debug!("Storing memory: {}", memory.id);

        // Add to vector index
        {
            let mut index = self.vector_index.write().await;
            index.add_vector(memory.id, memory.embedding.clone())?;
        }

        // Add to cache
        {
            let mut cache = self.in_memory_cache.write().await;
            cache.insert(memory.id, memory.clone());
        }

        // Store in database
        let pool = self.connection_pool.read().await;
        if let Some(ref connection) = *pool {
            let tags_json = serde_json::to_string(&memory.tags)
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
            
            let memory_type_json = serde_json::to_string(&memory.memory_type)
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?;
            
            let metadata_json = serde_json::to_string(&memory.metadata)
                .map_err(|e| MemoryError::SerializationError(e.to_string()))?;

            // Serialize embedding as bytes
            let embedding_bytes: Vec<u8> = memory.embedding
                .iter()
                .flat_map(|&f| f.to_le_bytes().to_vec())
                .collect();

            connection.execute(
                r#"INSERT OR REPLACE INTO memories 
                   (id, content, summary, tags, embedding, memory_type, importance, 
                    access_frequency, created_at, last_accessed, context_path, metadata)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)"#,
                rusqlite::params![
                    memory.id.to_string(),
                    memory.content,
                    memory.summary,
                    tags_json,
                    embedding_bytes,
                    memory_type_json,
                    memory.importance,
                    memory.access_frequency,
                    memory.created_at.to_rfc3339(),
                    memory.last_accessed.to_rfc3339(),
                    memory.context_path,
                    metadata_json
                ],
            ).map_err(|e| MemoryError::StorageError(format!("Failed to insert memory: {}", e)))?;
        }

        debug!("Memory stored successfully: {}", memory.id);
        Ok(())
    }

    /// Retrieve a memory by ID
    pub async fn get_memory(&self, id: &Uuid) -> MemoryResult<MemoryCell> {
        debug!("Retrieving memory: {}", id);

        // Check cache first
        {
            let cache = self.in_memory_cache.read().await;
            if let Some(memory) = cache.get(id) {
                debug!("Memory found in cache: {}", id);
                return Ok(memory.clone());
            }
        }

        // Query database
        let pool = self.connection_pool.read().await;
        if let Some(ref connection) = *pool {
            let mut stmt = connection.prepare(
                "SELECT id, content, summary, tags, embedding, memory_type, importance, 
                        access_frequency, created_at, last_accessed, context_path, metadata 
                 FROM memories WHERE id = ?1"
            ).map_err(|e| MemoryError::StorageError(format!("Failed to prepare statement: {}", e)))?;

            let memory = stmt.query_row(
                rusqlite::params![id.to_string()],
                |row| {
                    let embedding_bytes: Vec<u8> = row.get(4)?;
                    let embedding: Vec<f32> = embedding_bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    let tags_json: String = row.get(3)?;
                    let memory_type_json: String = row.get(5)?;
                    let metadata_json: String = row.get(11)?;

                    Ok(MemoryCell {
                        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                        content: row.get(1)?,
                        summary: row.get(2)?,
                        tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                        embedding,
                        memory_type: serde_json::from_str(&memory_type_json).unwrap(),
                        importance: row.get(6)?,
                        access_frequency: row.get(7)?,
                        created_at: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                            .unwrap().with_timezone(&chrono::Utc),
                        last_accessed: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(9)?)
                            .unwrap().with_timezone(&chrono::Utc),
                        context_path: row.get(10)?,
                        metadata: serde_json::from_str(&metadata_json).unwrap_or_default(),
                    })
                }
            ).map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => MemoryError::NotFound(format!("Memory not found: {}", id)),
                _ => MemoryError::StorageError(format!("Failed to query memory: {}", e))
            })?;

            // Update cache
            {
                let mut cache = self.in_memory_cache.write().await;
                cache.insert(*id, memory.clone());
            }

            debug!("Memory retrieved from database: {}", id);
            Ok(memory)
        } else {
            Err(MemoryError::StorageError("Database connection not available".to_string()))
        }
    }

    /// Find similar memories using vector search
    pub async fn find_similar_memories(
        &self,
        query_embedding: &[f32],
        limit: usize,
        min_similarity: f32,
    ) -> MemoryResult<Vec<(MemoryCell, f32)>> {
        debug!("Finding similar memories, limit: {}, min_similarity: {}", limit, min_similarity);

        let similar_ids = {
            let index = self.vector_index.read().await;
            index.find_similar(query_embedding, limit * 2) // Get more to filter by similarity
        };

        let mut results = Vec::new();
        for (id, similarity) in similar_ids {
            if similarity >= min_similarity {
                match self.get_memory(&id).await {
                    Ok(memory) => results.push((memory, similarity)),
                    Err(e) => warn!("Failed to retrieve similar memory {}: {}", id, e),
                }
            }
        }

        results.truncate(limit);
        debug!("Found {} similar memories", results.len());
        
        Ok(results)
    }

    /// Search memories by text content
    pub async fn search_memories(&self, query: &str, limit: usize) -> MemoryResult<Vec<MemoryCell>> {
        debug!("Searching memories with query: '{}', limit: {}", query, limit);

        let pool = self.connection_pool.read().await;
        if let Some(ref connection) = *pool {
            let mut stmt = connection.prepare(
                "SELECT m.id, m.content, m.summary, m.tags, m.embedding, m.memory_type, 
                        m.importance, m.access_frequency, m.created_at, m.last_accessed, 
                        m.context_path, m.metadata 
                 FROM memories m
                 JOIN memories_fts fts ON m.id = fts.id
                 WHERE memories_fts MATCH ?1
                 ORDER BY rank
                 LIMIT ?2"
            ).map_err(|e| MemoryError::StorageError(format!("Failed to prepare search statement: {}", e)))?;

            let memories: Result<Vec<MemoryCell>, _> = stmt.query_map(
                rusqlite::params![query, limit],
                |row| {
                    let embedding_bytes: Vec<u8> = row.get(4)?;
                    let embedding: Vec<f32> = embedding_bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    let tags_json: String = row.get(3)?;
                    let memory_type_json: String = row.get(5)?;
                    let metadata_json: String = row.get(11)?;

                    Ok(MemoryCell {
                        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                        content: row.get(1)?,
                        summary: row.get(2)?,
                        tags: serde_json::from_str(&tags_json).unwrap_or_default(),
                        embedding,
                        memory_type: serde_json::from_str(&memory_type_json).unwrap(),
                        importance: row.get(6)?,
                        access_frequency: row.get(7)?,
                        created_at: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                            .unwrap().with_timezone(&chrono::Utc),
                        last_accessed: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(9)?)
                            .unwrap().with_timezone(&chrono::Utc),
                        context_path: row.get(10)?,
                        metadata: serde_json::from_str(&metadata_json).unwrap_or_default(),
                    })
                }
            ).and_then(|rows| rows.collect());

            let memories = memories.map_err(|e| MemoryError::StorageError(format!("Failed to search memories: {}", e)))?;
            
            debug!("Found {} memories for search query", memories.len());
            Ok(memories)
        } else {
            Err(MemoryError::StorageError("Database connection not available".to_string()))
        }
    }

    /// Delete a memory
    pub async fn delete_memory(&self, id: &Uuid) -> MemoryResult<()> {
        debug!("Deleting memory: {}", id);

        // Remove from cache
        {
            let mut cache = self.in_memory_cache.write().await;
            cache.remove(id);
        }

        // Remove from vector index
        {
            let mut index = self.vector_index.write().await;
            index.remove_vector(id);
        }

        // Delete from database
        let pool = self.connection_pool.read().await;
        if let Some(ref connection) = *pool {
            let changes = connection.execute(
                "DELETE FROM memories WHERE id = ?1",
                rusqlite::params![id.to_string()]
            ).map_err(|e| MemoryError::StorageError(format!("Failed to delete memory: {}", e)))?;

            if changes == 0 {
                return Err(MemoryError::NotFound(format!("Memory not found: {}", id)));
            }
        }

        debug!("Memory deleted successfully: {}", id);
        Ok(())
    }

    /// Get database statistics
    pub async fn get_stats(&self) -> MemoryResult<HashMap<String, serde_json::Value>> {
        let pool = self.connection_pool.read().await;
        if let Some(ref connection) = *pool {
            let mut stats = HashMap::new();

            // Memory count
            let count: i64 = connection.query_row(
                "SELECT COUNT(*) FROM memories",
                [],
                |row| row.get(0)
            ).unwrap_or(0);
            stats.insert("memory_count".to_string(), serde_json::Value::from(count));

            // Database file size
            if let Ok(metadata) = std::fs::metadata(&self.config.database_path) {
                stats.insert("database_size_bytes".to_string(), serde_json::Value::from(metadata.len()));
            }

            // Cache statistics
            let cache = self.in_memory_cache.read().await;
            stats.insert("cache_count".to_string(), serde_json::Value::from(cache.len()));

            // Vector index statistics
            let index = self.vector_index.read().await;
            stats.insert("vector_index_count".to_string(), serde_json::Value::from(index.vectors.len()));
            if let Some(dim) = index.dimension {
                stats.insert("vector_dimension".to_string(), serde_json::Value::from(dim));
            }

            Ok(stats)
        } else {
            Err(MemoryError::StorageError("Database connection not available".to_string()))
        }
    }
}