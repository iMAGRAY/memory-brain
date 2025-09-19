//! Storage layer for memory persistence and vector operations
//!
//! This module handles Neo4j graph database operations, vector indexing,
//! and memory persistence with efficient retrieval capabilities.

use crate::simd_search::cosine_similarity_simd;
use crate::types::*;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use bincode;
use chrono::{DateTime, Utc};
use neo4rs::{query, Graph, Node};
use parking_lot::RwLock;
use serde_json;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{error, info, warn};
use uuid::Uuid;

// Optional internal ANN index (stub) — enabled under feature `ann_hnsw`.
// This provides a safe compilation path even when the real ANN backend is not wired.
// When the feature is enabled and runtime flag ENABLE_HNSW=1, we switch vector search
// to this backend (currently exact but structured for drop-in HNSW integration).
#[cfg(feature = "ann_hnsw")]
mod ann {
    use super::*;
    #[derive(Debug, Clone)]
    pub struct AnnHnswIndex {
        dim: usize,
        store: HashMap<Uuid, Vec<f32>>, // placeholder storage; replace with real HNSW nodes
    }

    impl AnnHnswIndex {
        pub fn new(dim: usize) -> Self { Self { dim, store: HashMap::new() } }
        pub fn len(&self) -> usize { self.store.len() }
        pub fn insert(&mut self, id: Uuid, v: Vec<f32>) {
            if v.len() == self.dim { self.store.insert(id, v); }
        }
        pub fn remove(&mut self, id: &Uuid) { self.store.remove(id); }
        pub fn search(&self, q: &[f32], k: usize, threshold: f32) -> Vec<(Uuid, f32)> {
            let mut top: Vec<(Uuid, f32)> = Vec::with_capacity(k);
            for (id, v) in self.store.iter() {
                if v.len() != q.len() { continue; }
                let s = crate::simd_search::cosine_similarity_simd(q, v);
                if s < threshold { continue; }
                if top.len() < k { top.push((*id, s)); }
                else {
                    // replace worst
                    let mut mi=0; let mut mv=top[0].1;
                    for (i, &(_, sc)) in top.iter().enumerate().skip(1) { if sc < mv { mv=sc; mi=i; } }
                    if s > mv { top[mi] = (*id, s); }
                }
            }
            top.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            top
        }
    }
}

/// Graph storage backend using Neo4j
pub struct GraphStorage {
    graph: Graph,
    vector_index: Arc<RwLock<VectorIndex>>,
    connection_semaphore: Semaphore,
}

/// In-memory vector index for fast similarity search
///
/// Maintains multiple indices for efficient memory retrieval:
/// - `embeddings`: Main storage for vector embeddings
/// - `context_index`: Groups memories by context path
/// - `type_index`: Groups memories by memory type
/// - `importance_index`: Sorted list by importance for ranking
/// - `memory_context_map`: Reverse lookup for memory ID to context path
#[derive(Debug, Clone)]
pub struct VectorIndex {
    /// Primary storage: memory ID -> embedding vector
    embeddings: HashMap<Uuid, Vec<f32>>,
    /// Context grouping: context path -> list of memory IDs
    context_index: HashMap<String, Vec<Uuid>>,
    /// Type grouping: memory type -> list of memory IDs  
    type_index: HashMap<String, Vec<Uuid>>,
    /// Sorted by importance: (memory ID, importance score)
    importance_index: Vec<(Uuid, f32)>,
    /// Reverse lookup: memory ID -> context path (for fast context retrieval)
    memory_context_map: HashMap<Uuid, String>,
    /// Type mapping: memory ID -> memory type (for efficient removal)
    memory_type_map: HashMap<Uuid, String>,
    /// ANN status flag (runtime toggle), currently always false unless ANN feature is enabled
    ann_enabled: bool,
    #[cfg(feature = "ann_hnsw")]
    ann: Option<ann::AnnHnswIndex>,
}

/// Similarity search result
#[derive(Debug, Clone)]
pub struct SimilarityMatch {
    pub memory_id: Uuid,
    pub similarity: f32,
    pub context_path: String,
}

impl GraphStorage {
    /// Create new graph storage instance
    pub async fn new(uri: &str, user: &str, password: &str) -> MemoryResult<Self> {
        let graph = Graph::new(uri, user, password)
            .await
            .map_err(|e| MemoryError::Storage(format!("Neo4j connection failed: {}", e)))?;

        let storage = Self {
            graph,
            vector_index: Arc::new(RwLock::new(VectorIndex::new())),
            connection_semaphore: Semaphore::new(10), // Default max connections
        };

        storage.initialize_schema().await?;
        storage.rebuild_vector_index().await?;

        Ok(storage)
    }

    /// Get ANN status for diagnostics/health endpoints
    pub fn ann_status(&self) -> (bool, usize) {
        let idx = self.vector_index.read();
        let enabled = idx.ann_enabled;
        let size = idx.embeddings.len();
        (enabled, size)
    }

    /// Persist ANN index to disk (best-effort). Only active when compiled with `ann_hnsw` and enabled at runtime.
    #[cfg(feature = "ann_hnsw")]
    pub fn ann_persist(&self, path: Option<&str>) -> MemoryResult<usize> {
        let idx = self.vector_index.read();
        if !idx.ann_enabled {
            return Ok(0);
        }
        let p = PathBuf::from(path.unwrap_or("target/ann_index.bin"));
        if let Some(dir) = p.parent() { let _ = fs::create_dir_all(dir); }
        // Dump from ANN store if present; fallback to embeddings map
        #[derive(serde::Serialize, serde::Deserialize)]
        struct AnnDump { dim: usize, items: Vec<(Uuid, Vec<f32>)> }
        let mut dim = 0usize;
        let mut items: Vec<(Uuid, Vec<f32>)> = Vec::new();
        if let Some(ann) = idx.ann.as_ref() {
            dim = ann.dim;
            for (id, v) in ann.store.iter() { items.push((*id, v.clone())); }
        } else {
            // Fallback: use main embeddings map
            for (id, v) in idx.embeddings.iter() {
                if dim == 0 { dim = v.len(); }
                items.push((*id, v.clone()));
            }
        }
        let dump = AnnDump { dim, items };
        let data = bincode::serialize(&dump).map_err(|e| MemoryError::Storage(format!("ANN dump serialize failed: {}", e)))?;
        fs::write(&p, data).map_err(|e| MemoryError::Storage(format!("ANN dump write failed: {}", e)))?;
        Ok(dump.items.len())
    }

    /// Stub when ANN feature not compiled
    #[cfg(not(feature = "ann_hnsw"))]
    pub fn ann_persist(&self, _path: Option<&str>) -> MemoryResult<usize> { Ok(0) }

    /// Initialize Neo4j schema and constraints
    async fn initialize_schema(&self) -> MemoryResult<()> {
        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let mut txn = self
            .graph
            .start_txn()
            .await
            .map_err(|e| MemoryError::Storage(format!("Transaction start failed: {}", e)))?;

        let schema_queries = vec![
            "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT context_path IF NOT EXISTS FOR (c:Context) REQUIRE c.path IS UNIQUE",
            "CREATE INDEX memory_importance IF NOT EXISTS FOR (m:Memory) ON (m.importance)",
            "CREATE INDEX memory_created IF NOT EXISTS FOR (m:Memory) ON (m.created_at)",
            "CREATE INDEX memory_accessed IF NOT EXISTS FOR (m:Memory) ON (m.last_accessed)",
            "CREATE INDEX memory_type IF NOT EXISTS FOR (m:Memory) ON (m.memory_type)",
            "CREATE INDEX context_activity IF NOT EXISTS FOR (c:Context) ON (c.activity_level)",
        ];

        for query_str in schema_queries {
            txn.run(query(query_str))
                .await
                .map_err(|e| MemoryError::Storage(format!("Schema creation failed: {}", e)))?;
        }

        txn.commit()
            .await
            .map_err(|e| MemoryError::Storage(format!("Schema commit failed: {}", e)))?;

        Ok(())
    }

    /// Store a memory cell in the graph database
    pub async fn store_memory(&self, memory: &MemoryCell) -> MemoryResult<()> {
        if memory.embedding.is_empty() {
            return Err(MemoryError::Storage(
                "Memory embedding cannot be empty".to_string(),
            ));
        }
        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let mut txn = self
            .graph
            .start_txn()
            .await
            .map_err(|e| MemoryError::Storage(format!("Transaction start failed: {}", e)))?;

        // Serialize embedding as base64 for storage
        let embedding_bytes = bincode::serialize(&memory.embedding)
            .map_err(|e| MemoryError::Storage(format!("Embedding serialization failed: {}", e)))?;
        let embedding_b64 = BASE64.encode(&embedding_bytes);

        // Serialize metadata with proper error handling
        let metadata_json = serde_json::to_string(&memory.metadata)
            .map_err(|e| MemoryError::Storage(format!("Failed to serialize metadata: {}", e)))?;

        // Use new memory type serialization method
        let memory_type_str = memory.memory_type.to_storage_string()?;

        let memory_query = query(
            "MERGE (m:Memory {id: $id})
             SET m.content = $content,
                 m.summary = $summary,
                 m.tags = $tags,
                 m.memory_type = $memory_type,
                 m.importance = $importance,
                 m.access_frequency = $access_frequency,
                 m.created_at = $created_at,
                 m.last_accessed = $last_accessed,
                 m.context_path = $context_path,
                 m.metadata = $metadata,
                 m.embedding = $embedding",
        )
        .param("id", memory.id.to_string())
        .param("content", memory.content.clone())
        .param("summary", memory.summary.clone())
        .param("tags", memory.tags.clone())
        .param("memory_type", memory_type_str)
        .param("importance", memory.importance)
        .param("access_frequency", memory.access_frequency)
        .param("created_at", memory.created_at.timestamp())
        .param("last_accessed", memory.last_accessed.timestamp())
        .param("context_path", memory.context_path.clone())
        .param("metadata", metadata_json)
        .param("embedding", embedding_b64);

        txn.run(memory_query)
            .await
            .map_err(|e| MemoryError::Storage(format!("Memory storage failed: {}", e)))?;

        let context_query = query(
            "MERGE (c:Context {path: $context_path})
             ON CREATE SET c.name = $context_path,
                          c.description = '',
                          c.parent = null,
                          c.children = [],
                          c.activity_level = 0.1,
                          c.memory_count = 0,
                          c.created_at = $now,
                          c.last_activity = $now
             ON MATCH SET c.memory_count = c.memory_count + 1,
                         c.last_activity = $now
             WITH c
             MATCH (m:Memory {id: $memory_id})
             MERGE (m)-[:IN_CONTEXT]->(c)",
        )
        .param("context_path", memory.context_path.as_str())
        .param("memory_id", memory.id.to_string())
        .param("now", Utc::now().timestamp());

        txn.run(context_query)
            .await
            .map_err(|e| MemoryError::Storage(format!("Context storage failed: {}", e)))?;

        for tag in &memory.tags {
            let tag_query = query(
                "MERGE (t:Tag {name: $tag})
                 WITH t
                 MATCH (m:Memory {id: $memory_id})
                 MERGE (m)-[:TAGGED_WITH]->(t)",
            )
            .param("tag", tag.as_str())
            .param("memory_id", memory.id.to_string());

            txn.run(tag_query)
                .await
                .map_err(|e| MemoryError::Storage(format!("Tag relationship failed: {}", e)))?;
        }

        txn.commit()
            .await
            .map_err(|e| MemoryError::Storage(format!("Transaction commit failed: {}", e)))?;

        self.update_vector_index(
            memory.id,
            &memory.embedding,
            &memory.context_path,
            &format!("{:?}", memory.memory_type),
            memory.importance,
        )?;

        Ok(())
    }

    /// Retrieve memories by IDs
    pub async fn get_memories(&self, ids: &[Uuid]) -> MemoryResult<Vec<MemoryCell>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let id_strings: Vec<String> = ids.iter().map(|id| id.to_string()).collect();

        let mut query_result = self
            .graph
            .execute(query("MATCH (m:Memory) WHERE m.id IN $ids RETURN m").param("ids", id_strings))
            .await
            .map_err(|e| MemoryError::Storage(format!("Memory retrieval failed: {}", e)))?;

        let mut memories = Vec::new();
        while let Ok(Some(row)) = query_result.next().await {
            if let Ok(node) = row.get::<Node>("m") {
                if let Ok(memory) = self.node_to_memory(node).await {
                    memories.push(memory);
                }
            }
        }

        Ok(memories)
    }

    /// Find similar memories using vector similarity with optimized search
    pub async fn find_similar_memories(
        &self,
        query_embedding: &[f32],
        limit: usize,
        threshold: f32,
        context_filter: Option<&str>,
        memory_type_filter: Option<&str>,
    ) -> MemoryResult<Vec<SimilarityMatch>> {
        if query_embedding.is_empty() {
            return Err(MemoryError::Storage(
                "Query embedding cannot be empty".to_string(),
            ));
        }

        let index = self.vector_index.read();

        // Get filtered candidate IDs efficiently
        let candidate_ids =
            self.get_filtered_candidates(&index, context_filter, memory_type_filter);

        if candidate_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Pre-allocate with capacity hint for better performance
        let mut similarities = Vec::with_capacity(std::cmp::min(candidate_ids.len(), limit * 2));

        // Use parallel processing for large candidate sets
        if candidate_ids.len() > 1000 {
            self.compute_similarities_parallel(
                query_embedding,
                &candidate_ids,
                threshold,
                &index,
                &mut similarities,
            )?;
        } else {
            self.compute_similarities_sequential(
                query_embedding,
                &candidate_ids,
                threshold,
                &index,
                &mut similarities,
            )?;
        }

        // Sort by similarity descending and return top results
        similarities.sort_unstable_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        similarities.truncate(limit);

        Ok(similarities)
    }

    /// List memories with missing or effectively empty embeddings (id + content for backfill)
    pub async fn list_memories_missing_embeddings(&self, limit: usize) -> MemoryResult<Vec<(Uuid, String)>> {
        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let mut out = Vec::new();
        // Detect both null/empty and near-empty serialized payloads.
        // We cannot decode base64 in Cypher reliably; use a conservative heuristic on string length.
        // Empty Vec<f32> serialized via bincode/base64 will be very short (< 16 bytes as base64 string).
        let cy = query(
            "MATCH (m:Memory) \
             WHERE m.embedding IS NULL OR m.embedding = '' OR size(m.embedding) < 16 \
             RETURN m.id as id, m.content as content \
             LIMIT $limit"
        ).param("limit", limit as i64);

        let mut result = self.graph.execute(cy).await
            .map_err(|e| MemoryError::Storage(format!("List missing embeddings failed: {}", e)))?;
        while let Ok(Some(row)) = result.next().await {
            let id_str: String = row.get::<String>("id").unwrap_or_default();
            if let Ok(id) = Uuid::parse_str(&id_str) {
                let content: String = row.get::<String>("content").unwrap_or_default();
                out.push((id, content));
            }
        }
        Ok(out)
    }

    /// Count memories with missing or effectively empty embeddings
    pub async fn count_missing_embeddings(&self) -> MemoryResult<usize> {
        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let cy = query(
            "MATCH (m:Memory) \
             WHERE m.embedding IS NULL OR m.embedding = '' OR size(m.embedding) < 16 \
             RETURN count(m) as c"
        );
        let mut result = self.graph.execute(cy).await
            .map_err(|e| MemoryError::Storage(format!("Count missing embeddings failed: {}", e)))?;
        if let Ok(Some(row)) = result.next().await {
            let c: i64 = row.get::<i64>("c").unwrap_or(0);
            return Ok(c as usize);
        }
        Ok(0)
    }

    /// Count memories with missing/empty embeddings under contexts with the given prefix
    pub async fn count_missing_embeddings_with_prefix(&self, prefix: &str) -> MemoryResult<usize> {
        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let cy = query(
            "MATCH (m:Memory)-[:IN_CONTEXT]->(c:Context) \
             WHERE c.path STARTS WITH $pref AND (m.embedding IS NULL OR m.embedding = '' OR size(m.embedding) < 16) \
             RETURN count(m) as c"
        ).param("pref", prefix.to_string());
        let mut result = self.graph.execute(cy).await
            .map_err(|e| MemoryError::Storage(format!("Count missing embeddings by prefix failed: {}", e)))?;
        if let Ok(Some(row)) = result.next().await {
            let c: i64 = row.get::<i64>("c").unwrap_or(0);
            return Ok(c as usize);
        }
        Ok(0)
    }

    /// Purge orphaned memories with missing/empty embeddings. Optional context prefix filter.
    pub async fn purge_orphaned_embeddings(&self, prefix: Option<&str>, dry_run: bool) -> MemoryResult<usize> {
        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let cy = if let Some(pref) = prefix {
            if dry_run {
                query(
                    "MATCH (m:Memory)-[:IN_CONTEXT]->(c:Context) \
                     WHERE c.path STARTS WITH $pref AND (m.embedding IS NULL OR m.embedding = '' OR size(m.embedding) < 16) \
                     RETURN count(m) AS removed"
                ).param("pref", pref.to_string())
            } else {
                query(
                    "MATCH (m:Memory)-[:IN_CONTEXT]->(c:Context) \
                     WHERE c.path STARTS WITH $pref AND (m.embedding IS NULL OR m.embedding = '' OR size(m.embedding) < 16) \
                     WITH m \
                     DETACH DELETE m \
                     RETURN 0 AS removed"
                ).param("pref", pref.to_string())
            }
        } else {
            if dry_run {
                query(
                    "MATCH (m:Memory) \
                     WHERE m.embedding IS NULL OR m.embedding = '' OR size(m.embedding) < 16 \
                     RETURN count(m) AS removed"
                )
            } else {
                query(
                    "MATCH (m:Memory) \
                     WHERE m.embedding IS NULL OR m.embedding = '' OR size(m.embedding) < 16 \
                     WITH m \
                     DETACH DELETE m \
                     RETURN 0 AS removed"
                )
            }
        };

        let mut result = self.graph.execute(cy).await
            .map_err(|e| MemoryError::Storage(format!("Purge orphaned embeddings failed: {}", e)))?;
        if let Ok(Some(row)) = result.next().await {
            let r: i64 = row.get::<i64>("removed").unwrap_or(0);
            Ok(r as usize)
        } else {
            Ok(0)
        }
    }

    /// Update a memory embedding field and refresh vector index (idempotent)
    pub async fn update_memory_embedding(&self, memory_id: &Uuid, embedding: &[f32]) -> MemoryResult<()> {
        if embedding.is_empty() {
            return Err(MemoryError::Storage("Refusing to write empty embedding".to_string()));
        }
        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let embedding_bytes = bincode::serialize(embedding)
            .map_err(|e| MemoryError::Storage(format!("Embedding serialization failed: {}", e)))?;
        let embedding_b64 = BASE64.encode(&embedding_bytes);

        let cy = query(
            "MATCH (m:Memory {id: $id}) SET m.embedding = $emb RETURN m.id as id"
        ).param("id", memory_id.to_string()).param("emb", embedding_b64);

        let _ = self.graph.execute(cy).await
            .map_err(|e| MemoryError::Storage(format!("Failed to update embedding: {}", e)))?;

        // Update in-memory vector index
        self.update_vector_index(
            *memory_id,
            embedding,
            &self.get_memory_context(memory_id).await.unwrap_or_else(|_| "unknown".to_string()),
            "semantic",
            0.5,
        )?;
        Ok(())
    }

    /// List a sample of memories with id, content and raw embedding payload for probe
    pub async fn list_memories_for_embedding_probe(&self, limit: usize) -> MemoryResult<Vec<(Uuid, String, String)>> {
        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let mut out = Vec::new();
        let cy = query(
            "MATCH (m:Memory) \
             RETURN m.id as id, m.content as content, coalesce(m.embedding,'') as emb \
             LIMIT $limit"
        ).param("limit", limit as i64);
        let mut result = self.graph.execute(cy).await
            .map_err(|e| MemoryError::Storage(format!("Probe list failed: {}", e)))?;
        while let Ok(Some(row)) = result.next().await {
            let id_str: String = row.get::<String>("id").unwrap_or_default();
            if let Ok(id) = Uuid::parse_str(&id_str) {
                let content: String = row.get::<String>("content").unwrap_or_default();
                let emb_str: String = row.get::<String>("emb").unwrap_or_default();
                out.push((id, content, emb_str));
            }
        }
        Ok(out)
    }

    async fn get_memory_context(&self, memory_id: &Uuid) -> MemoryResult<String> {
        let cy = query(
            "MATCH (m:Memory {id: $id}) RETURN m.context_path as ctx LIMIT 1"
        ).param("id", memory_id.to_string());
        let mut result = self.graph.execute(cy).await
            .map_err(|e| MemoryError::Storage(format!("Failed to get memory context: {}", e)))?;
        if let Ok(Some(row)) = result.next().await {
            Ok(row.get::<String>("ctx").unwrap_or_else(|_| "unknown".to_string()))
        } else {
            Ok("unknown".to_string())
        }
    }

    /// Get filtered candidate IDs based on context and type filters
    /// Returns candidate IDs efficiently without unnecessary cloning
    fn get_filtered_candidates(
        &self,
        index: &VectorIndex,
        context_filter: Option<&str>,
        memory_type_filter: Option<&str>,
    ) -> Vec<Uuid> {
        match (context_filter, memory_type_filter) {
            (Some(context), Some(mem_type)) => {
                // Efficient intersection using smaller set for lookup
                let (lookup_ids, filter_ids) = self.get_filter_sets(index, context, mem_type);

                filter_ids
                    .iter()
                    .filter(|&&id| lookup_ids.contains(&id))
                    .copied()
                    .collect()
            }
            (Some(context), None) => self.get_indexed_ids(&index.context_index, context, "context"),
            (None, Some(mem_type)) => {
                self.get_indexed_ids(&index.type_index, mem_type, "memory type")
            }
            (None, None) => index.embeddings.keys().copied().collect(),
        }
    }

    /// Optimize intersection by choosing smaller set for HashSet creation
    ///
    /// Performance optimization strategy:
    /// 1. Create HashSet from the smaller collection for faster lookups
    /// 2. Iterate through the larger collection to minimize HashSet size
    /// 3. Intersection complexity: O(min(m,n)) space + O(max(m,n)) time
    /// 4. Without optimization: O(m) space + O(n) time where m might be >> n
    ///
    /// Example: If context has 1000 memories and type has 10 memories,
    /// create HashSet(10) and iterate Vec(1000) instead of HashSet(1000)
    fn get_filter_sets<'a>(
        &self,
        index: &'a VectorIndex,
        context: &str,
        mem_type: &str,
    ) -> (std::collections::HashSet<Uuid>, &'a [Uuid]) {
        let context_ids = index.context_index.get(context);
        let type_ids = index.type_index.get(mem_type);

        match (context_ids, type_ids) {
            (Some(ctx_ids), Some(type_ids)) => {
                // Optimization: Use smaller collection for HashSet creation
                // This reduces memory usage and improves lookup performance
                if ctx_ids.len() <= type_ids.len() {
                    (ctx_ids.iter().copied().collect(), type_ids.as_slice())
                } else {
                    (type_ids.iter().copied().collect(), ctx_ids.as_slice())
                }
            }
            (Some(ctx_ids), None) => {
                tracing::debug!("Memory type '{}' not found in index", mem_type);
                (std::collections::HashSet::new(), ctx_ids.as_slice())
            }
            (None, Some(type_ids)) => {
                tracing::debug!("Context '{}' not found in index", context);
                (std::collections::HashSet::new(), type_ids.as_slice())
            }
            (None, None) => {
                tracing::debug!(
                    "Neither context '{}' nor memory type '{}' found in index",
                    context,
                    mem_type
                );
                (std::collections::HashSet::new(), &[])
            }
        }
    }

    /// Get IDs from index with zero-copy optimization for single filters
    ///
    /// Performance note: For single filters, we still need to clone the Vec
    /// because the caller expects ownership. In future optimizations, this
    /// could return a Cow<[Uuid]> to avoid cloning when possible.
    fn get_indexed_ids(
        &self,
        index_map: &HashMap<String, Vec<Uuid>>,
        key: &str,
        index_type: &str,
    ) -> Vec<Uuid> {
        match index_map.get(key) {
            Some(ids) => {
                // TODO: Consider returning Cow<[Uuid]> to avoid cloning in read-only scenarios
                tracing::trace!(
                    "Found {} {} memories for key '{}'",
                    ids.len(),
                    index_type,
                    key
                );
                ids.clone()
            }
            None => {
                tracing::debug!("{} '{}' not found in index", index_type, key);
                Vec::new()
            }
        }
    }

    /// Compute similarities sequentially for smaller datasets
    fn compute_similarities_sequential(
        &self,
        query_embedding: &[f32],
        candidate_ids: &[Uuid],
        threshold: f32,
        index: &VectorIndex,
        similarities: &mut Vec<SimilarityMatch>,
    ) -> MemoryResult<()> {
        for &memory_id in candidate_ids {
            if let Some(similarity_match) =
                self.compute_single_similarity(query_embedding, memory_id, threshold, index)?
            {
                similarities.push(similarity_match);
            }
        }
        Ok(())
    }

    /// Compute similarities in parallel for larger datasets
    fn compute_similarities_parallel(
        &self,
        query_embedding: &[f32],
        candidate_ids: &[Uuid],
        threshold: f32,
        index: &VectorIndex,
        similarities: &mut Vec<SimilarityMatch>,
    ) -> MemoryResult<()> {
        // Note: This would need proper cloning of index data for parallel processing
        // For now, fall back to sequential processing to avoid borrowing issues
        self.compute_similarities_sequential(
            query_embedding,
            candidate_ids,
            threshold,
            index,
            similarities,
        )
    }

    /// Compute similarity for a single memory with proper error handling
    fn compute_single_similarity(
        &self,
        query_embedding: &[f32],
        memory_id: Uuid,
        threshold: f32,
        index: &VectorIndex,
    ) -> MemoryResult<Option<SimilarityMatch>> {
        let embedding = index.embeddings.get(&memory_id).ok_or_else(|| {
            MemoryError::Storage(format!("Embedding not found for memory {}", memory_id))
        })?;

        // Validate embedding dimensions
        if embedding.len() != query_embedding.len() {
            return Err(MemoryError::Storage(format!(
                "Embedding dimension mismatch: query={}, stored={} for memory {}",
                query_embedding.len(),
                embedding.len(),
                memory_id
            )));
        }

        let similarity = cosine_similarity_simd(query_embedding, embedding);

        if similarity >= threshold {
            let context_path = index
                .memory_context_map
                .get(&memory_id)
                .cloned()
                .unwrap_or_else(|| {
                    // Fallback to expensive lookup if not in optimized map
                    index
                        .context_index
                        .iter()
                        .find(|(_, ids)| ids.contains(&memory_id))
                        .map(|(path, _)| path.clone())
                        .unwrap_or_default()
                });

            Ok(Some(SimilarityMatch {
                memory_id,
                similarity,
                context_path,
            }))
        } else {
            Ok(None)
        }
    }

    /// Get memories by context path
    pub async fn get_memories_in_context_with_option(
        &self,
        context_path: &str,
        limit: Option<usize>,
    ) -> MemoryResult<Vec<MemoryCell>> {
        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let query_str = if let Some(limit_val) = limit {
            format!(
                "MATCH (m:Memory)-[:IN_CONTEXT]->(c:Context {{path: $context_path}})
                 RETURN m ORDER BY m.importance DESC, m.last_accessed DESC LIMIT {}",
                limit_val
            )
        } else {
            "MATCH (m:Memory)-[:IN_CONTEXT]->(c:Context {path: $context_path})
             RETURN m ORDER BY m.importance DESC, m.last_accessed DESC"
                .to_string()
        };

        let mut query_result = self
            .graph
            .execute(query(&query_str).param("context_path", context_path))
            .await
            .map_err(|e| MemoryError::Storage(format!("Context query failed: {}", e)))?;

        let mut memories = Vec::new();
        while let Ok(Some(row)) = query_result.next().await {
            if let Ok(node) = row.get::<Node>("m") {
                if let Ok(memory) = self.node_to_memory(node).await {
                    memories.push(memory);
                }
            }
        }

        Ok(memories)
    }

    /// Update memory access statistics
    pub async fn update_access_stats(&self, memory_id: Uuid) -> MemoryResult<()> {
        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let mut query_result = self
            .graph
            .execute(
                query(
                    "MATCH (m:Memory {id: $id})
                 SET m.access_frequency = m.access_frequency + 1,
                     m.last_accessed = $now
                 RETURN m.context_path",
                )
                .param("id", memory_id.to_string())
                .param("now", Utc::now().timestamp()),
            )
            .await
            .map_err(|e| MemoryError::Storage(format!("Access update failed: {}", e)))?;

        if let Ok(Some(row)) = query_result.next().await {
            if let Ok(context_path) = row.get::<String>("m.context_path") {
                let _result = self
                    .graph
                    .execute(
                        query(
                            "MATCH (c:Context {path: $context_path})
                         SET c.activity_level = c.activity_level + 0.1,
                             c.last_activity = $now",
                        )
                        .param("context_path", context_path)
                        .param("now", Utc::now().timestamp()),
                    )
                    .await
                    .map_err(|e| {
                        MemoryError::Storage(format!("Context activity update failed: {}", e))
                    })?;
            }
        }

        Ok(())
    }

    /// Get memory statistics
    pub async fn get_memory_stats(&self) -> MemoryResult<MemoryStats> {
        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let mut total_query = self
            .graph
            .execute(query("MATCH (m:Memory) RETURN count(m) as total"))
            .await
            .map_err(|e| MemoryError::Storage(format!("Stats query failed: {}", e)))?;

        let total_memories = if let Ok(Some(row)) = total_query.next().await {
            row.get::<i64>("total").unwrap_or(0) as usize
        } else {
            0
        };

        let mut contexts_query = self
            .graph
            .execute(query("MATCH (c:Context) RETURN count(c) as total"))
            .await
            .map_err(|e| MemoryError::Storage(format!("Context stats failed: {}", e)))?;

        let total_contexts = if let Ok(Some(row)) = contexts_query.next().await {
            row.get::<i64>("total").unwrap_or(0) as usize
        } else {
            0
        };

        let mut type_query = self
            .graph
            .execute(query(
                "MATCH (m:Memory) RETURN m.memory_type, count(*) as count",
            ))
            .await
            .map_err(|e| MemoryError::Storage(format!("Type stats failed: {}", e)))?;

        let mut memory_by_type = HashMap::new();
        while let Ok(Some(row)) = type_query.next().await {
            if let (Ok(mem_type), Ok(count)) =
                (row.get::<String>("m.memory_type"), row.get::<i64>("count"))
            {
                memory_by_type.insert(mem_type, count as usize);
            }
        }

        let mut top_contexts_query = self.graph.execute(
            query("MATCH (c:Context) RETURN c.path, c.memory_count ORDER BY c.memory_count DESC LIMIT 10")
        ).await
        .map_err(|e| MemoryError::Storage(format!("Top contexts query failed: {}", e)))?;

        let mut top_contexts = Vec::new();
        while let Ok(Some(row)) = top_contexts_query.next().await {
            if let (Ok(path), Ok(count)) = (
                row.get::<String>("c.path"),
                row.get::<i64>("c.memory_count"),
            ) {
                top_contexts.push((path, count as usize));
            }
        }

        Ok(MemoryStats {
            total_memories,
            total_contexts,
            active_memories: total_memories,
            memory_by_type,
            top_contexts,
            recent_queries: 0,
            avg_recall_time_ms: 0.0,
            cache_hit_rate: 0.0,
            storage_size_mb: 0.0,
        })
    }

    /// Rebuild vector index from database
    async fn rebuild_vector_index(&self) -> MemoryResult<()> {
        info!("🔄 Rebuilding vector index from Neo4j...");

        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let mut new_index = VectorIndex::new();
        let mut processed_count = 0;
        let mut error_count = 0;

        let mut query_result = self.graph.execute(
            query("MATCH (m:Memory) RETURN m.id, m.context_path, m.memory_type, m.importance, m.embedding")
        ).await
        .map_err(|e| MemoryError::Storage(format!("Index rebuild failed: {}", e)))?;

        while let Ok(Some(row)) = query_result.next().await {
            if let (
                Ok(id_str),
                Ok(context_path),
                Ok(memory_type),
                Ok(importance),
                Ok(embedding_b64),
            ) = (
                row.get::<String>("m.id"),
                row.get::<String>("m.context_path"),
                row.get::<String>("m.memory_type"),
                row.get::<f64>("m.importance"),
                row.get::<String>("m.embedding"),
            ) {
                if let Ok(memory_id) = Uuid::parse_str(&id_str) {
                    if let Ok(embedding_bytes) = BASE64.decode(&embedding_b64) {
                        if let Ok(embedding) = bincode::deserialize::<Vec<f32>>(&embedding_bytes) {
                            if embedding.is_empty() {
                                tracing::debug!("Skipping empty embedding during index rebuild for {}", memory_id);
                            } else {
                                new_index.add_memory(
                                    memory_id,
                                    embedding,
                                    context_path,
                                    memory_type,
                                    importance as f32,
                                );
                            }
                            processed_count += 1;

                            // Log progress every 100 memories
                            if processed_count % 100 == 0 {
                                info!("📊 Processed {} memories...", processed_count);
                            }
                        } else {
                            error!("❌ Failed to deserialize embedding for memory: {}", id_str);
                            error_count += 1;
                        }
                    } else {
                        error!(
                            "❌ Failed to decode base64 embedding for memory: {}",
                            id_str
                        );
                        error_count += 1;
                    }
                } else {
                    error!("❌ Failed to parse UUID for memory: {}", id_str);
                    error_count += 1;
                }
            } else {
                warn!("⚠️ Skipping memory with incomplete data");
                error_count += 1;
            }
        }

        // Replace the old index with the new one
        *self.vector_index.write() = new_index;

        info!(
            "✅ Vector index rebuilt successfully: {} memories processed, {} errors",
            processed_count, error_count
        );

        if error_count > 0 {
            warn!("⚠️ Encountered {} errors during index rebuild", error_count);
        }

        Ok(())
    }

    fn update_vector_index(
        &self,
        memory_id: Uuid,
        embedding: &[f32],
        context_path: &str,
        memory_type: &str,
        importance: f32,
    ) -> MemoryResult<()> {
        let mut index = self.vector_index.write();
        index.add_memory(
            memory_id,
            embedding.to_vec(),
            context_path.to_string(),
            memory_type.to_string(),
            importance,
        );
        Ok(())
    }

    async fn node_to_memory(&self, node: Node) -> MemoryResult<MemoryCell> {
        let id = Uuid::parse_str(
            &node
                .get::<String>("id")
                .map_err(|e| MemoryError::Storage(format!("Invalid memory ID: {}", e)))?,
        )
        .map_err(|e| MemoryError::Storage(format!("UUID parse error: {}", e)))?;

        let content = node
            .get::<String>("content")
            .map_err(|e| MemoryError::Storage(format!("Missing content: {}", e)))?;

        // Handle optional fields with proper validation
        let summary = node.get::<String>("summary").unwrap_or_else(|_| {
            tracing::debug!("Summary not found for memory {}, using empty string", id);
            String::new()
        });

        let tags = node.get::<Vec<String>>("tags").unwrap_or_else(|_| {
            tracing::debug!("Tags not found for memory {}, using empty vector", id);
            Vec::new()
        });

        let context_path = node.get::<String>("context_path").unwrap_or_else(|_| {
            tracing::warn!("Context path not found for memory {}, using default", id);
            "unknown".to_string()
        });

        // Handle timestamps with validation
        let created_timestamp = node.get::<i64>("created_at").unwrap_or_else(|_| {
            tracing::warn!(
                "Created timestamp not found for memory {}, using current time",
                id
            );
            Utc::now().timestamp()
        });

        let accessed_timestamp = node.get::<i64>("last_accessed").unwrap_or_else(|_| {
            tracing::debug!(
                "Last accessed timestamp not found for memory {}, using created time",
                id
            );
            created_timestamp
        });

        let created_at = self.parse_timestamp_safe(created_timestamp, "created_at", id)?;
        let last_accessed = self.parse_timestamp_safe(accessed_timestamp, "last_accessed", id)?;

        // Handle numeric fields with validation
        let importance = node
            .get::<f64>("importance")
            .unwrap_or_else(|_| {
                tracing::debug!("Importance not found for memory {}, using default 0.5", id);
                0.5
            })
            .clamp(0.0, 1.0) as f32;

        let access_frequency = node
            .get::<i64>("access_frequency")
            .unwrap_or_else(|_| {
                tracing::debug!("Access frequency not found for memory {}, using 0", id);
                0
            })
            .max(0) as u32;

        // Handle metadata with proper JSON validation
        let metadata_str = node.get::<String>("metadata").unwrap_or_else(|_| {
            tracing::debug!("Metadata not found for memory {}, using empty object", id);
            "{}".to_string()
        });

        let metadata = serde_json::from_str(&metadata_str).map_err(|e| {
            MemoryError::Storage(format!("Invalid metadata JSON for memory {}: {}", id, e))
        })?;

        // Parse memory type from stored string with comprehensive validation
        let memory_type_str = node.get::<String>("memory_type").map_err(|_e| {
            MemoryError::Storage(
                "Memory type field missing for memory: database integrity error".to_string(),
            )
        })?;

        // Security: Limit memory type string size to prevent abuse
        if memory_type_str.len() > 1024 {
            return Err(MemoryError::Storage(
                "Memory type string exceeds maximum allowed length".to_string(),
            ));
        }

        let memory_type = MemoryType::from_storage_string(&memory_type_str).map_err(|_| {
            MemoryError::Storage("Invalid memory type format in database".to_string())
        })?;

        // Validate memory type data integrity
        memory_type.validate().map_err(|_| {
            MemoryError::Storage("Memory type failed validation checks".to_string())
        })?;

        // Parse embedding with strict security and size limits
        let embedding = self.parse_embedding_secure(&node, id)?;

        Ok(MemoryCell {
            id,
            content,
            summary,
            tags,
            embedding,
            memory_type,
            importance,
            access_frequency,
            created_at,
            last_accessed,
            context_path,
            metadata,
        })
    }

    /// Securely parse and validate embedding data with comprehensive limits
    fn parse_embedding_secure(&self, node: &Node, memory_id: Uuid) -> MemoryResult<Vec<f32>> {
        // Security limits for embeddings
        const MAX_EMBEDDING_B64_SIZE: usize = 50 * 1024; // 50KB base64 limit
        const MAX_EMBEDDING_DIMENSIONS: usize = 4096; // Reasonable dimension limit
        const MAX_EMBEDDING_BYTES: usize = MAX_EMBEDDING_DIMENSIONS * 4; // f32 = 4 bytes

        let embedding_b64 = node.get::<String>("embedding").map_err(|_| {
            tracing::warn!("No embedding field found for memory {}", memory_id);
            MemoryError::Storage("Embedding field missing from database record".to_string())
        })?;

        // Security: Prevent DoS via oversized base64 strings
        if embedding_b64.len() > MAX_EMBEDDING_B64_SIZE {
            return Err(MemoryError::Storage(format!(
                "Embedding data too large: {} bytes (max: {})",
                embedding_b64.len(),
                MAX_EMBEDDING_B64_SIZE
            )));
        }

        // Decode base64 with size validation
        let embedding_bytes = BASE64.decode(&embedding_b64).map_err(|e| {
            tracing::error!("Base64 decode failed for memory {}: {}", memory_id, e);
            MemoryError::Storage("Invalid base64 encoding in embedding data".to_string())
        })?;

        // Security: Validate decoded size before deserialization
        if embedding_bytes.len() > MAX_EMBEDDING_BYTES {
            return Err(MemoryError::Storage(format!(
                "Decoded embedding too large: {} bytes (max: {})",
                embedding_bytes.len(),
                MAX_EMBEDDING_BYTES
            )));
        }

        // Safe deserialization with bounds checking
        let embedding: Vec<f32> = bincode::deserialize(&embedding_bytes).map_err(|e| {
            tracing::error!(
                "Embedding deserialization failed for memory {}: {}",
                memory_id,
                e
            );
            MemoryError::Storage("Corrupted embedding data in database".to_string())
        })?;

        // Validate embedding dimensions and data quality
        if embedding.len() > MAX_EMBEDDING_DIMENSIONS {
            return Err(MemoryError::Storage(format!(
                "Embedding has too many dimensions: {} (max: {})",
                embedding.len(),
                MAX_EMBEDDING_DIMENSIONS
            )));
        }

        if embedding.is_empty() {
            tracing::warn!("Empty embedding vector for memory {}", memory_id);
            return Ok(embedding); // Allow empty embeddings for some use cases
        }

        // Validate embedding values for sanity
        let invalid_count = embedding
            .iter()
            .filter(|&&val| val.is_nan() || val.is_infinite())
            .count();

        if invalid_count > 0 {
            tracing::warn!(
                "Embedding contains {} invalid values for memory {}",
                invalid_count,
                memory_id
            );
            // Could choose to reject or sanitize - for now, warn but accept
        }

        // Check for suspicious patterns that might indicate attack vectors
        let zero_count = embedding.iter().filter(|&&val| val == 0.0).count();
        if zero_count == embedding.len() && embedding.len() > 10 {
            tracing::warn!(
                "Embedding is all zeros for memory {} - potential data corruption",
                memory_id
            );
        }

        tracing::trace!(
            "Successfully parsed embedding with {} dimensions for memory {}",
            embedding.len(),
            memory_id
        );
        Ok(embedding)
    }

    /// Helper to parse timestamps with validation
    fn parse_timestamp_safe(
        &self,
        timestamp: i64,
        field_name: &str,
        memory_id: Uuid,
    ) -> MemoryResult<DateTime<Utc>> {
        chrono::DateTime::from_timestamp(timestamp, 0).ok_or_else(|| {
            tracing::error!(
                "Invalid {} timestamp {} for memory {}",
                field_name,
                timestamp,
                memory_id
            );
            MemoryError::Storage(format!("Invalid {} timestamp in database", field_name))
        })
    }

    /// Perform vector similarity search
    pub async fn vector_search(
        &self,
        query_embedding: &[f32],
        limit: usize,
        similarity_threshold: Option<f32>,
    ) -> MemoryResult<Vec<MemoryCell>> {
        // ANN path (if feature enabled and runtime flag ENABLE_HNSW=1)
        #[cfg(feature = "ann_hnsw")]
        {
            let idx = self.vector_index.read();
            if idx.ann_enabled {
                let threshold = similarity_threshold.unwrap_or(0.1);
                let hits = idx.ann.as_ref().and_then(|ann| Some(ann.search(query_embedding, limit.max(1), threshold))).unwrap_or_default();
                drop(idx);
                let mut out = Vec::with_capacity(hits.len());
                for (id, _s) in hits { if let Ok(m) = self.get_memory(&id).await { out.push(m); } }
                return Ok(out);
            }
        }
        // Fallback exact SIMD cosine search
        use super::simd_search::cosine_similarity_simd;
        
        let threshold = similarity_threshold.unwrap_or(0.1); // Default to 0.1 if not specified
        tracing::debug!("Vector search with threshold: {} (limit: {})", threshold, limit);

        // Compute top-K using a small min-heap to avoid full sort
        let top = {
            let vector_index = self.vector_index.read();
            let mut topk: Vec<(uuid::Uuid, f32)> = Vec::with_capacity(limit + 1);
            for (id, embedding) in &vector_index.embeddings {
                let sim = cosine_similarity_simd(embedding, query_embedding);
                if sim < threshold { continue; }
                if topk.len() < limit {
                    topk.push((*id, sim));
                } else if limit > 0 {
                    // find current min
                    let mut min_idx = 0usize; let mut min_val = topk[0].1;
                    for (i, &(_, s)) in topk.iter().enumerate().skip(1) {
                        if s < min_val { min_val = s; min_idx = i; }
                    }
                    if sim > min_val { topk[min_idx] = (*id, sim); }
                }
            }
            topk.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            topk
        }; // drop read lock

        let mut results = Vec::with_capacity(top.len());
        for (id, sim) in top.into_iter() {
            tracing::trace!("Returning memory {} with similarity {}", id, sim);
            if let Ok(memory) = self.get_memory(&id).await {
                results.push(memory);
            }
        }
        Ok(results)
    }

    /// Find memories related to a given memory
    pub async fn find_related_memories(
        &self,
        memory_id: &Uuid,
        limit: usize,
    ) -> MemoryResult<Vec<MemoryCell>> {
        let query = query(
            r#"
            MATCH (m:Memory {id: $id})-[:RELATED_TO]-(related:Memory)
            RETURN related
            ORDER BY related.importance DESC
            LIMIT $limit
            "#,
        )
        .param("id", memory_id.to_string())
        .param("limit", limit as i64);

        let mut result =
            self.graph.execute(query).await.map_err(|e| {
                MemoryError::Storage(format!("Failed to find related memories: {}", e))
            })?;

        let mut memories = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            if let Ok(memory) = self.parse_memory_from_row(&row).await {
                memories.push(memory);
            }
        }

        Ok(memories)
    }

    /// Create or strengthen RELATED_TO link between two memories
    pub async fn create_related_link(
        &self,
        a: &Uuid,
        b: &Uuid,
        weight: f32,
    ) -> MemoryResult<()> {
        if a == b {
            return Ok(());
        }

        // Direction normalization: smaller UUID string goes first to avoid duplicates
        let (from_id, to_id) = {
            let sa = a.to_string();
            let sb = b.to_string();
            if sa <= sb { (sa, sb) } else { (sb, sa) }
        };

        let q = query(
            r#"
            MATCH (a:Memory {id: $from}), (b:Memory {id: $to})
            MERGE (a)-[r:RELATED_TO]->(b)
            ON CREATE SET r.weight = $w, r.created_at = timestamp()
            ON MATCH SET r.weight = (coalesce(r.weight, 0.0) + $w) / 2.0, r.updated_at = timestamp()
            "#
        )
        .param("from", from_id)
        .param("to", to_id)
        .param("w", weight as f64);

        self.graph
            .run(q)
            .await
            .map_err(|e| MemoryError::Storage(format!("Failed to create RELATED_TO link: {}", e)))?;

        Ok(())
    }

    /// Apply global importance decay
    pub async fn apply_decay(&self, decay_rate: f32, min_threshold: f32) -> MemoryResult<usize> {
        let q = query(
            r#"
            MATCH (m:Memory)
            WITH m, m.importance AS imp
            SET m.importance = CASE WHEN imp * (1.0 - $rate) < $min THEN $min ELSE imp * (1.0 - $rate) END
            RETURN count(m) AS updated
            "#
        )
        .param("rate", decay_rate as f64)
        .param("min", min_threshold as f64);

        let mut res = self
            .graph
            .execute(q)
            .await
            .map_err(|e| MemoryError::Storage(format!("Failed to apply decay: {}", e)))?;

        let mut updated = 0usize;
        if let Ok(Some(row)) = res.next().await {
            updated = row.get::<i64>("updated").unwrap_or(0) as usize;
        }
        Ok(updated)
    }

    /// Mark one memory as duplicate of another and adjust importance of the duplicate
    pub async fn mark_duplicate_of(
        &self,
        duplicate: &Uuid,
        master: &Uuid,
        reduce_to: Option<f32>,
    ) -> MemoryResult<()> {
        let q = query(
            r#"
            MATCH (d:Memory {id: $dup}), (m:Memory {id: $mas})
            MERGE (d)-[r:DUPLICATE_OF]->(m)
            ON CREATE SET r.created_at = timestamp()
            ON MATCH SET r.updated_at = timestamp()
            "#
        )
        .param("dup", duplicate.to_string())
        .param("mas", master.to_string());

        self.graph
            .run(q)
            .await
            .map_err(|e| MemoryError::Storage(format!("Failed to mark duplicate: {}", e)))?;

        if let Some(new_imp) = reduce_to {
            self.update_memory_importance(duplicate, new_imp).await?;
        }
        Ok(())
    }

    /// Update importance field for a memory
    pub async fn update_memory_importance(&self, id: &Uuid, importance: f32) -> MemoryResult<()> {
        let q = query(
            r#"
            MATCH (m:Memory {id: $id})
            SET m.importance = $imp
            RETURN m.id AS id
            "#
        )
        .param("id", id.to_string())
        .param("imp", importance as f64);

        self
            .graph
            .run(q)
            .await
            .map_err(|e| MemoryError::Storage(format!("Failed to update importance: {}", e)))?;
        Ok(())
    }

    /// Get a specific memory by ID
    pub async fn get_memory(&self, id: &Uuid) -> MemoryResult<MemoryCell> {
        let query = query(
            r#"
            MATCH (m:Memory {id: $id})
            RETURN m
            "#,
        )
        .param("id", id.to_string());

        let mut result = self
            .graph
            .execute(query)
            .await
            .map_err(|e| MemoryError::Storage(format!("Failed to get memory: {}", e)))?;

        if let Ok(Some(row)) = result.next().await {
            self.parse_memory_from_row(&row).await
        } else {
            Err(MemoryError::NotFound(*id))
        }
    }

    /// Delete a memory by ID
    pub async fn delete_memory(&self, id: &Uuid) -> MemoryResult<()> {
        let query = query(
            r#"
            MATCH (m:Memory {id: $id})
            DETACH DELETE m
            "#,
        )
        .param("id", id.to_string());

        self.graph
            .run(query)
            .await
            .map_err(|e| MemoryError::Storage(format!("Failed to delete memory: {}", e)))?;

        // Remove from vector index
        let mut vector_index = self.vector_index.write();
        vector_index.remove_memory(*id);

        Ok(())
    }

    /// List all contexts
    pub async fn list_contexts(&self) -> MemoryResult<Vec<String>> {
        let query = query(
            r#"
            MATCH (c:Context)
            RETURN c.path as path
            ORDER BY c.path
            "#,
        );

        let mut result = self
            .graph
            .execute(query)
            .await
            .map_err(|e| MemoryError::Storage(format!("Failed to list contexts: {}", e)))?;

        let mut contexts = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            if let Ok(path) = row.get::<String>("path") {
                contexts.push(path);
            }
        }

        Ok(contexts)
    }

    /// Get a context by path
    pub async fn get_context(&self, path: &str) -> MemoryResult<MemoryContext> {
        let query = query(
            r#"
            MATCH (c:Context {path: $path})
            OPTIONAL MATCH (c)<-[:IN_CONTEXT]-(m:Memory)
            RETURN c, count(m) as memory_count
            "#,
        )
        .param("path", path);

        let mut result = self
            .graph
            .execute(query)
            .await
            .map_err(|e| MemoryError::Storage(format!("Failed to get context: {}", e)))?;

        if let Ok(Some(row)) = result.next().await {
            let node: neo4rs::Node = row
                .get("c")
                .map_err(|e| MemoryError::Storage(format!("Failed to get context node: {}", e)))?;

            let memory_count: i64 = row.get("memory_count").unwrap_or(0);

            Ok(MemoryContext {
                path: node.get::<String>("path").unwrap_or_default(),
                name: node.get::<String>("name").unwrap_or_default(),
                description: node.get::<String>("description").unwrap_or_default(),
                parent: node.get::<String>("parent").ok(),
                children: Vec::new(),
                embedding: Vec::new(),
                activity_level: node.get::<f64>("activity_level").unwrap_or(0.0) as f32,
                memory_count: memory_count as usize,
                created_at: Utc::now(),
                last_activity: Utc::now(),
            })
        } else {
            Err(MemoryError::Storage(format!("Context not found: {}", path)))
        }
    }

    /// Get memories in a specific context
    pub async fn get_memories_in_context(
        &self,
        context_path: &str,
        limit: usize,
    ) -> MemoryResult<Vec<MemoryCell>> {
        let query = query(
            r#"
            MATCH (c:Context {path: $context_path})<-[:IN_CONTEXT]-(m:Memory)
            RETURN m
            ORDER BY m.importance DESC, m.created_at DESC
            LIMIT $limit
            "#,
        )
        .param("context_path", context_path)
        .param("limit", limit as i64);

        let mut result = self.graph.execute(query).await.map_err(|e| {
            MemoryError::Storage(format!("Failed to get memories in context: {}", e))
        })?;

        let mut memories = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            if let Ok(memory) = self.parse_memory_from_row(&row).await {
                memories.push(memory);
            }
        }

        Ok(memories)
    }

    /// Compute degree and connectivity stats over RELATED_TO edges
    pub async fn graph_degree_stats(&self) -> MemoryResult<(f64, f64, usize)> {
        let q = query(
            r#"
            MATCH (m:Memory)
            OPTIONAL MATCH (m)-[:RELATED_TO]-(:Memory)
            WITH m, count(*) AS deg
            RETURN avg(toFloat(deg)) AS avg_deg,
                   avg(CASE WHEN deg > 0 THEN 1.0 ELSE 0.0 END) AS connected_ratio,
                   count(m) AS total
            "#
        );
        let mut res = self.graph.execute(q).await
            .map_err(|e| MemoryError::Storage(format!("Failed to compute degree stats: {}", e)))?;
        if let Ok(Some(row)) = res.next().await {
            let avg_deg: f64 = row.get::<f64>("avg_deg").unwrap_or(0.0);
            let connected_ratio: f64 = row.get::<f64>("connected_ratio").unwrap_or(0.0);
            let total: i64 = row.get::<i64>("total").unwrap_or(0);
            Ok((avg_deg, connected_ratio, total as usize))
        } else {
            Ok((0.0, 0.0, 0))
        }
    }

    /// Fetch per-context memory counts
    pub async fn context_counts(&self) -> MemoryResult<Vec<usize>> {
        let q = query(
            r#"
            MATCH (c:Context)
            RETURN coalesce(c.memory_count,0) AS cnt
            "#
        );
        let mut res = self.graph.execute(q).await
            .map_err(|e| MemoryError::Storage(format!("Failed to fetch context counts: {}", e)))?;
        let mut out = Vec::new();
        while let Ok(Some(row)) = res.next().await {
            let cnt: i64 = row.get::<i64>("cnt").unwrap_or(0);
            out.push(cnt as usize);
        }
        Ok(out)
    }

    /// Map of context path -> memory_count
    pub async fn context_stats_map(&self) -> MemoryResult<Vec<(String, usize)>> {
        let q = query(
            r#"
            MATCH (c:Context)
            RETURN c.path AS path, coalesce(c.memory_count,0) AS cnt
            ORDER BY path
            "#
        );
        let mut res = self.graph.execute(q).await
            .map_err(|e| MemoryError::Storage(format!("Failed to fetch context stats: {}", e)))?;
        let mut out: Vec<(String, usize)> = Vec::new();
        while let Ok(Some(row)) = res.next().await {
            let path = row.get::<String>("path").unwrap_or_default();
            let cnt: i64 = row.get::<i64>("cnt").unwrap_or(0);
            out.push((path, cnt as usize));
        }
        Ok(out)
    }

    /// Approximate two-hop expansion factor across a sample of memories
    pub async fn two_hop_expansion_factor(&self, sample: usize) -> MemoryResult<f64> {
        let sample = sample.max(1).min(500);
        let q = query(
            r#"
            MATCH (m:Memory)
            WITH m LIMIT $sample
            OPTIONAL MATCH (m)-[:RELATED_TO]-(n)
            WITH m, collect(DISTINCT n) AS one
            OPTIONAL MATCH (m)-[:RELATED_TO]-()-[:RELATED_TO]-(p:Memory)
            WITH one, collect(DISTINCT p) AS two
            WITH size(one) AS one_hop,
                 [x IN two WHERE NOT x IN one] AS addTwo
            RETURN avg( CASE WHEN one_hop>0 THEN toFloat(size(addTwo))/toFloat(one_hop) ELSE 0.0 END ) AS avg_expansion
            "#
        ).param("sample", sample as i64);
        let mut res = self.graph.execute(q).await
            .map_err(|e| MemoryError::Storage(format!("Failed to compute two-hop expansion: {}", e)))?;
        if let Ok(Some(row)) = res.next().await {
            let v: f64 = row.get::<f64>("avg_expansion").unwrap_or(0.0);
            Ok(v)
        } else {
            Ok(0.0)
        }
    }

    /// Approximate average clustering (triad closure) over a sample of nodes
    pub async fn approx_clustering(&self, sample: usize) -> MemoryResult<f64> {
        let sample = sample.max(1).min(200);
        let q = query(
            r#"
            MATCH (a:Memory)
            WITH a LIMIT $sample
            MATCH (a)-[:RELATED_TO]-(b)
            WITH a, collect(DISTINCT b) AS nbrs
            WITH a, nbrs, size(nbrs) AS d WHERE d >= 2
            UNWIND nbrs AS b
            UNWIND nbrs AS c
            WITH a, b, c, d WHERE id(b) < id(c)
            MATCH (b)-[:RELATED_TO]-(c)
            WITH a, d, count(*) AS closed
            RETURN avg( CASE WHEN d*(d-1)/2 > 0 THEN toFloat(closed)/toFloat(d*(d-1)/2) ELSE 0.0 END ) AS avg_closure
            "#
        ).param("sample", sample as i64);
        let mut res = self.graph.execute(q).await
            .map_err(|e| MemoryError::Storage(format!("Failed to compute clustering: {}", e)))?;
        if let Ok(Some(row)) = res.next().await {
            let v: f64 = row.get::<f64>("avg_closure").unwrap_or(0.0);
            Ok(v)
        } else {
            Ok(0.0)
        }
    }

    /// Approximate average shortest path length within up to 5 hops over a small sample
    pub async fn approx_shortest_path_len(&self, sample: usize) -> MemoryResult<f64> {
        let sample = sample.max(1).min(50);
        let q = query(
            r#"
            MATCH (a:Memory)
            WITH a LIMIT $sample
            MATCH (b:Memory)
            WHERE id(b) > id(a)
            WITH a, collect(b)[..5] AS targets
            UNWIND targets AS b
            OPTIONAL MATCH p = shortestPath( (a)-[:RELATED_TO*..5]-(b) )
            WITH CASE WHEN p IS NULL THEN 0.0 ELSE toFloat(length(p)) END AS l
            RETURN avg(l) AS avg_len
            "#
        ).param("sample", sample as i64);
        let mut res = self.graph.execute(q).await
            .map_err(|e| MemoryError::Storage(format!("Failed to compute shortest path len: {}", e)))?;
        if let Ok(Some(row)) = res.next().await {
            let v: f64 = row.get::<f64>("avg_len").unwrap_or(0.0);
            Ok(v)
        } else {
            Ok(0.0)
        }
    }

    /// Parse memory from Neo4j query row
    ///
    /// Extracts a memory node from a query result row and converts it to a MemoryCell
    async fn parse_memory_from_row(&self, row: &neo4rs::Row) -> MemoryResult<MemoryCell> {
        let node: Node = row.get("m").map_err(|e| {
            MemoryError::Storage(format!("Failed to get memory node from row: {}", e))
        })?;

        self.node_to_memory(node).await
    }

    /// Get storage statistics (totals only)
    pub async fn get_stats(&self) -> MemoryResult<StorageStats> {
        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let mut total_query = self
            .graph
            .execute(query("MATCH (m:Memory) RETURN count(m) as total"))
            .await
            .map_err(|e| MemoryError::Storage(format!("Stats query failed: {}", e)))?;

        let total_memories = if let Ok(Some(row)) = total_query.next().await {
            row.get::<i64>("total").unwrap_or(0) as usize
        } else {
            0
        };

        let mut contexts_query = self
            .graph
            .execute(query("MATCH (c:Context) RETURN count(c) as total"))
            .await
            .map_err(|e| MemoryError::Storage(format!("Context stats failed: {}", e)))?;

        let total_contexts = if let Ok(Some(row)) = contexts_query.next().await {
            row.get::<i64>("total").unwrap_or(0) as usize
        } else {
            0
        };

        Ok(StorageStats {
            total_memories,
            total_contexts,
            active_memories: 0,
        })
    }

    /// Count active memories strictly above a given importance threshold
    /// Excludes memories marked as duplicates via DUPLICATE_OF.
    pub async fn get_active_count(&self, threshold: f32) -> MemoryResult<usize> {
        let _permit = self
            .connection_semaphore
            .acquire()
            .await
            .map_err(|e| MemoryError::Storage(format!("Connection pool error: {}", e)))?;

        let mut q = self
            .graph
            .execute(
                query(
                    "MATCH (m:Memory)\n                     WHERE coalesce(m.importance, 0.0) > $thr\n                     AND NOT (m)-[:DUPLICATE_OF]->(:Memory)\n                     RETURN count(m) as active"
                )
                .param("thr", threshold as f64),
            )
            .await
            .map_err(|e| MemoryError::Storage(format!("Active count query failed: {}", e)))?;

        let active = if let Ok(Some(row)) = q.next().await {
            row.get::<i64>("active").unwrap_or(0) as usize
        } else {
            0
        };
        Ok(active)
    }
}

impl VectorIndex {
    /// Create a new empty vector index with all required data structures
    fn new() -> Self {
        // Runtime toggle for ANN (only effective when compiled with ann_hnsw)
        let mut ann_enabled = false;
        #[cfg(feature = "ann_hnsw")]
        {
            let v = env::var("ENABLE_HNSW").unwrap_or_default();
            ann_enabled = matches!(v.as_str(), "1" | "true" | "TRUE" | "True");
        }
        Self {
            embeddings: HashMap::new(),
            context_index: HashMap::new(),
            type_index: HashMap::new(),
            importance_index: Vec::new(),
            memory_context_map: HashMap::new(),
            memory_type_map: HashMap::new(),
            ann_enabled,
            #[cfg(feature = "ann_hnsw")]
            ann: None,
        }
    }

    /// Add a memory to all relevant indices
    ///
    /// Maintains consistency across all index structures:
    /// - Stores embedding vector
    /// - Updates context and type groupings
    /// - Maintains sorted importance index
    /// - Updates reverse lookup maps
    fn add_memory(
        &mut self,
        id: Uuid,
        mut embedding: Vec<f32>,
        context_path: String,
        memory_type: String,
        importance: f32,
    ) {
        // Normalize embedding vector (Matryoshka best-practice) to ensure cosine/dot equivalence
        if !embedding.is_empty() {
            let mut sum = 0.0f32;
            for &x in &embedding { sum += x * x; }
            let norm = sum.sqrt();
            if norm > 0.0 {
                let inv = (1.0f32 / norm).max(1e-12);
                for x in &mut embedding { *x *= inv; }
            }
        }
        // Store embedding vector
        self.embeddings.insert(id, embedding);

        // Add to context index for context-based filtering
        self.context_index
            .entry(context_path.clone())
            .or_default()
            .push(id);

        // Add to type index for type-based filtering
        self.type_index
            .entry(memory_type.clone())
            .or_default()
            .push(id);

        // Add to importance index (maintain descending order for ranking)
        let pos = self
            .importance_index
            .binary_search_by(|(_, existing_importance)| {
                // Sort descending: higher importance comes first
                // Compare importance with existing, then reverse for descending order
                importance
                    .partial_cmp(existing_importance)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .reverse()
            })
            .unwrap_or_else(|pos| pos);
        self.importance_index.insert(pos, (id, importance));

        // Store reverse mappings for efficient lookups during similarity search
        self.memory_context_map.insert(id, context_path);
        self.memory_type_map.insert(id, memory_type);
        #[cfg(feature = "ann_hnsw")]
        if self.ann_enabled {
            let dim = embedding.len();
            if self.ann.is_none() {
                self.ann = Some(ann::AnnHnswIndex::new(dim));
            }
            if let Some(ann) = self.ann.as_mut() { ann.insert(id, embedding.clone()); }
        }
    }

    /// Remove memory from all indices efficiently
    ///
    /// Uses reverse mappings to avoid expensive iterations
    fn remove_memory(&mut self, id: Uuid) {
        // Remove primary embedding
        self.embeddings.remove(&id);

        // Remove from context index using reverse mapping
        if let Some(context_path) = self.memory_context_map.remove(&id) {
            if let Some(context_ids) = self.context_index.get_mut(&context_path) {
                context_ids.retain(|&memory_id| memory_id != id);
                // Clean up empty context entries
                if context_ids.is_empty() {
                    self.context_index.remove(&context_path);
                }
            }
        }

        // Remove from type index using reverse mapping (O(1) instead of O(n))
        if let Some(memory_type) = self.memory_type_map.remove(&id) {
            if let Some(type_ids) = self.type_index.get_mut(&memory_type) {
                type_ids.retain(|&memory_id| memory_id != id);
                // Clean up empty type entries
                if type_ids.is_empty() {
                    self.type_index.remove(&memory_type);
                }
            }
        }

        // Remove from importance index
        self.importance_index
            .retain(|(memory_id, _)| *memory_id != id);
    }

    /// Get total number of memories in the index
    #[allow(dead_code)]
    fn memory_count(&self) -> usize {
        self.embeddings.len()
    }

    /// Comprehensive validation of index consistency
    ///
    /// Verifies that all indices are synchronized and contain consistent data
    #[allow(dead_code)]
    fn validate_consistency(&self) -> Result<(), String> {
        let embedding_count = self.embeddings.len();

        // Check size consistency across all maps
        if self.memory_context_map.len() != embedding_count {
            return Err(format!(
                "Context map size {} doesn't match embeddings size {}",
                self.memory_context_map.len(),
                embedding_count
            ));
        }

        if self.memory_type_map.len() != embedding_count {
            return Err(format!(
                "Type map size {} doesn't match embeddings size {}",
                self.memory_type_map.len(),
                embedding_count
            ));
        }

        // Check that all embeddings have corresponding reverse mappings
        for &id in self.embeddings.keys() {
            if !self.memory_context_map.contains_key(&id) {
                return Err(format!("Memory {} missing from context map", id));
            }
            if !self.memory_type_map.contains_key(&id) {
                return Err(format!("Memory {} missing from type map", id));
            }
        }

        // Check for orphaned entries in reverse mappings
        for &id in self.memory_context_map.keys() {
            if !self.embeddings.contains_key(&id) {
                return Err(format!("Orphaned entry in context map: {}", id));
            }
        }

        for &id in self.memory_type_map.keys() {
            if !self.embeddings.contains_key(&id) {
                return Err(format!("Orphaned entry in type map: {}", id));
            }
        }

        // Validate forward indices contain all memories
        let context_memory_count: usize = self.context_index.values().map(|v| v.len()).sum();
        let type_memory_count: usize = self.type_index.values().map(|v| v.len()).sum();

        if context_memory_count != embedding_count {
            return Err(format!(
                "Context index contains {} memories but should contain {}",
                context_memory_count, embedding_count
            ));
        }

        if type_memory_count != embedding_count {
            return Err(format!(
                "Type index contains {} memories but should contain {}",
                type_memory_count, embedding_count
            ));
        }

        // Validate importance index
        if self.importance_index.len() != embedding_count {
            return Err(format!(
                "Importance index size {} doesn't match embeddings size {}",
                self.importance_index.len(),
                embedding_count
            ));
        }

        // Check importance index is properly sorted (descending)
        for i in 1..self.importance_index.len() {
            if self.importance_index[i - 1].1 < self.importance_index[i].1 {
                return Err(
                    "Importance index is not properly sorted in descending order".to_string(),
                );
            }
        }

        Ok(())
    }

    /// Get memory statistics for monitoring
    #[allow(dead_code)]
    fn get_index_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total_memories".to_string(), self.embeddings.len());
        stats.insert("contexts".to_string(), self.context_index.len());
        stats.insert("types".to_string(), self.type_index.len());
        stats
    }

    /// Compact the index by removing empty entries and optimizing memory usage
    #[allow(dead_code)]
    fn compact(&mut self) {
        // Remove empty context entries
        self.context_index.retain(|_, ids| !ids.is_empty());

        // Remove empty type entries
        self.type_index.retain(|_, ids| !ids.is_empty());

        // Shrink vectors to fit actual content
        self.importance_index.shrink_to_fit();
        for ids in self.context_index.values_mut() {
            ids.shrink_to_fit();
        }
        for ids in self.type_index.values_mut() {
            ids.shrink_to_fit();
        }
    }
}
