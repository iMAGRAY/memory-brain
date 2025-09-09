//! Core memory service implementation
//!
//! This module contains the main memory service that coordinates between
//! storage, embedding generation, AI brain processing, and caching.

use crate::brain::AIBrain;
use crate::cache::{CacheConfig, CacheSystem};
use crate::config::Config;
use crate::embedding::EmbeddingService;
use crate::storage::GraphStorage;
use crate::types::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

/// Main memory service that orchestrates all memory operations
pub struct MemoryService {
    /// Vector embedding generation service
    embedding_service: Arc<EmbeddingService>,
    /// Graph storage backend (Neo4j)
    graph_storage: Arc<GraphStorage>,
    /// AI brain for content analysis
    ai_brain: Arc<AIBrain>,
    /// Multi-layer cache system
    cache: Arc<CacheSystem>,
    /// Configuration settings
    config: Config,
    /// Runtime statistics
    stats: Arc<RwLock<MemoryStats>>,
}

impl MemoryService {
    /// Create a new memory service with the given configuration
    pub async fn new(config: Config) -> MemoryResult<Self> {
        info!("Initializing AI Memory Service");

        // Initialize embedding service
        let embedding_service = Arc::new(
            EmbeddingService::new(
                &config.embedding.model_path,
                &config.embedding.tokenizer_path,
                config.embedding.batch_size,
                config.embedding.max_sequence_length,
            )
            .await
            .map_err(|e| {
                MemoryError::Config(format!("Failed to initialize embedding service: {}", e))
            })?,
        );

        // Initialize graph storage
        let graph_storage = Arc::new(
            GraphStorage::new(
                &config.storage.neo4j_uri,
                &config.storage.neo4j_user,
                &config.storage.neo4j_password,
            )
            .await
            .map_err(|e| {
                MemoryError::Storage(format!("Failed to initialize graph storage: {}", e))
            })?,
        );

        // Initialize AI brain
        let ai_brain = Arc::new(AIBrain::new("gemma-300m".to_string()));

        // Initialize cache system
        let cache_config = CacheConfig {
            l1_max_size: config.cache.l1_size,
            l2_max_size: config.cache.l2_size,
            ttl_seconds: config.cache.ttl_seconds,
            importance_threshold_l1: 0.8,
            importance_threshold_l2: 0.5,
            access_frequency_l1: 10,
            access_frequency_l2: 3,
        };
        let cache = Arc::new(CacheSystem::new(cache_config));

        let stats = Arc::new(RwLock::new(MemoryStats::default()));

        info!("AI Memory Service initialized successfully");

        Ok(Self {
            embedding_service,
            graph_storage,
            ai_brain,
            cache,
            config,
            stats,
        })
    }

    /// Store a new memory in the system
    #[tracing::instrument(skip(self, content, context_hint))]
    pub async fn store_memory(
        &self,
        content: String,
        context_hint: Option<String>,
    ) -> MemoryResult<Uuid> {
        let start_time = Instant::now();
        debug!("Storing new memory with context hint: {:?}", context_hint);

        // Validate inputs
        if content.trim().is_empty() {
            return Err(MemoryError::InvalidQuery(
                "Content cannot be empty".to_string(),
            ));
        }

        // Generate embedding for the content (using Query type for consistency with search)
        let t0 = Instant::now();
        let embedding = self
            .embedding_service
            .embed(&content, crate::embedding::TaskType::Query)
            .await?;
        crate::metrics::record_embedding_latency("gemma-300m", t0.elapsed().as_secs_f64());

        // Analyze content with AI brain
        let analysis = self
            .ai_brain
            .analyze_content(&content, context_hint.as_deref())
            .await?;

        // Create memory cell
        let mut memory_cell = MemoryCell::new(content, analysis.suggested_context.clone());
        memory_cell.summary = analysis.summary;
        memory_cell.tags = analysis.tags;
        memory_cell.embedding = embedding;
        memory_cell.memory_type = analysis.memory_type;
        memory_cell.importance = analysis.importance.clamp(0.0, 1.0);

        // Add extracted metadata
        if let Some(sentiment) = analysis.sentiment {
            memory_cell
                .metadata
                .insert("sentiment_score".to_string(), sentiment.score.to_string());
            memory_cell.metadata.insert(
                "sentiment_confidence".to_string(),
                sentiment.confidence.to_string(),
            );
        }

        for entity in analysis.entities {
            memory_cell.metadata.insert(
                format!("entity_{:?}", entity.entity_type),
                format!("{}:{}", entity.text, entity.confidence),
            );
        }

        // Store in graph database
        self.graph_storage.store_memory(&memory_cell).await?;

        // Cache the memory
        self.cache.put_memory(memory_cell.clone()).await;

        // Enrich context graph with lightweight RELATED_TO links (co-occurrence)
        // Link the new memory with a few recent memories from the same context.
        if !memory_cell.context_path.is_empty() {
            let ctx = memory_cell.context_path.clone();
            // Fetch a small window of memories from the same context
            let mut recent_in_ctx = self
                .graph_storage
                .get_memories_in_context(&ctx, 6)
                .await
                .unwrap_or_default();
            // Keep a few most recent and distinct
            recent_in_ctx.retain(|m| m.id != memory_cell.id);
            recent_in_ctx.sort_by(|a, b| b.last_accessed.cmp(&a.last_accessed));
            recent_in_ctx.truncate(3);

            // Create RELATED_TO links with base weight
            for m in recent_in_ctx {
                let _ = self
                    .graph_storage
                    .create_related_link(&memory_cell.id, &m.id, 1.0)
                    .await;
            }
        }

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.total_memories += 1;
        let memory_type_key = memory_cell.memory_type.type_name().to_string();
        *stats.memory_by_type.entry(memory_type_key).or_insert(0) += 1;

        let elapsed = start_time.elapsed().as_millis();
        info!(
            "Memory stored successfully in {}ms: {}",
            elapsed, memory_cell.id
        );
        crate::metrics::record_memory_op("store", true);

        Ok(memory_cell.id)
    }

    /// Recall memories based on a query
    #[tracing::instrument(skip(self, query))]
    pub async fn recall_memory(&self, query: MemoryQuery) -> MemoryResult<RecalledMemory> {
        let start_time = Instant::now();
        let query_id = Uuid::new_v4();

        debug!("Processing memory recall query: {}", query.text);

        // Check cache first
        let query_hash = self.hash_query(&query);
        if let Some(cached_result) = self.cache.get_query(&query_hash).await {
            debug!("Cache hit for query");
            return Ok((*cached_result).clone());
        }

        // Generate query embedding
        let tq0 = Instant::now();
        let query_embedding = self
            .embedding_service
            .embed(&query.text, crate::embedding::TaskType::Query)
            .await?;
        crate::metrics::record_embedding_latency("gemma-300m", tq0.elapsed().as_secs_f64());

        // Layer 1: Semantic search - fast associative retrieval
        let semantic_results = self
            .graph_storage
            .vector_search(
                &query_embedding,
                query.limit.unwrap_or(20),
                query.similarity_threshold,
            )
            .await?;

        // Layer 2: Contextual search - explore related memories
        let mut contextual_results = Vec::new();
        for memory in &semantic_results[..semantic_results.len().min(5)] {
            let related = self
                .graph_storage
                .find_related_memories(&memory.id, 5)
                .await?;
            contextual_results.extend(related);
        }

        // Layer 3: Detailed search - deep memory exploration
        let detailed_results = if query.include_related {
            self.perform_detailed_search(&contextual_results).await?
        } else {
            Vec::new()
        };

        // Build reasoning chain
        let reasoning_chain = vec![
            format!("Found {} semantic matches", semantic_results.len()),
            format!(
                "Expanded to {} contextual connections",
                contextual_results.len()
            ),
            format!("Retrieved {} detailed memories", detailed_results.len()),
        ];

        // Calculate confidence
        let confidence = self.calculate_confidence(&semantic_results, &contextual_results);

        // Discover additional contexts
        let discovered_contexts: Vec<String> = contextual_results
            .iter()
            .map(|m| m.context_path.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .take(5)
            .collect();

        let recall_time_ms = start_time.elapsed().as_millis() as u64;
        crate::metrics::record_recall_latency(
            if query.include_related { "advanced" } else { "basic" },
            (recall_time_ms as f64) / 1000.0,
        );

        let recalled = RecalledMemory {
            query_id,
            semantic_layer: semantic_results,
            contextual_layer: contextual_results,
            detailed_layer: detailed_results,
            reasoning_chain,
            confidence,
            recall_time_ms,
            discovered_contexts,
        };

        // Cache the result
        self.cache.put_query(query_hash, recalled.clone()).await;

        // Update statistics
        let mut stats = self.stats.write().await;
        stats.recent_queries += 1;
        stats.avg_recall_time_ms = (stats.avg_recall_time_ms * (stats.recent_queries - 1) as f64
            + recall_time_ms as f64)
            / stats.recent_queries as f64;

        info!("Memory recall completed in {}ms", recall_time_ms);

        // Process with AI brain for enhanced results
        self.ai_brain
            .process_recall(recalled)
            .await
            .map(|processed| RecalledMemory {
                query_id,
                semantic_layer: processed.semantic,
                contextual_layer: processed.contextual,
                detailed_layer: processed.detailed,
                reasoning_chain: processed.reasoning,
                confidence: processed.confidence,
                recall_time_ms,
                discovered_contexts: processed.suggestions,
            })
    }

    /// Apply one decay tick to memory importance across storage
    pub async fn apply_decay_tick(&self) -> MemoryResult<usize> {
        let rate = self.config.brain.decay_rate;
        let min_threshold = self.config.brain.importance_threshold;
        let updated = self
            .graph_storage
            .apply_decay(rate, min_threshold)
            .await?;
        Ok(updated)
    }

    /// Consolidate near-duplicates within a context (lightweight)
    /// Uses cosine similarity in memory to mark duplicates and reduce importance of the duplicate.
    pub async fn consolidate_duplicates(
        &self,
        context_path: Option<&str>,
        similarity_threshold: f32,
        max_items: usize,
    ) -> MemoryResult<usize> {
        use crate::simd_search::cosine_similarity_simd;
        let mut total_marked = 0usize;

        // Choose a scope: given context or top few contexts via list_contexts
        let contexts = if let Some(ctx) = context_path {
            vec![ctx.to_string()]
        } else {
            self.graph_storage.list_contexts().await?.into_iter().take(3).collect()
        };

        for ctx in contexts {
            let mut mems = self
                .graph_storage
                .get_memories_in_context(&ctx, max_items)
                .await
                .unwrap_or_default();
            // Pairwise compare (bounded)
            for i in 0..mems.len() {
                for j in (i + 1)..mems.len() {
                    let s = cosine_similarity_simd(&mems[i].embedding, &mems[j].embedding);
                    if s >= similarity_threshold {
                        // choose master by higher importance
                        let (master, duplicate) = if mems[i].importance >= mems[j].importance {
                            (&mems[i], &mems[j])
                        } else {
                            (&mems[j], &mems[i])
                        };
                        let reduced = (duplicate.importance * 0.5).max(self.config.brain.importance_threshold);
                        let _ = self
                            .graph_storage
                            .mark_duplicate_of(&duplicate.id, &master.id, Some(reduced))
                            .await;
                        total_marked += 1;
                        if total_marked >= 50 { // hard cap per call
                            return Ok(total_marked);
                        }
                    }
                }
            }
        }

        Ok(total_marked)
    }

    /// Get a specific memory by ID
    pub async fn get_memory(&self, id: &Uuid) -> Option<MemoryCell> {
        // Check cache first
        if let Some(memory) = self.cache.get_memory(id).await {
            return Some((*memory).clone());
        }

        // Fetch from storage
        self.graph_storage.get_memory(id).await.ok()
    }

    /// Delete a memory by ID
    pub async fn delete_memory(&self, id: &Uuid) -> MemoryResult<()> {
        self.graph_storage.delete_memory(id).await?;
        self.cache.clear_all().await;

        let mut stats = self.stats.write().await;
        if stats.total_memories > 0 {
            stats.total_memories -= 1;
        }

        Ok(())
    }

    /// List all contexts
    pub async fn list_contexts(&self) -> MemoryResult<Vec<String>> {
        self.graph_storage.list_contexts().await
    }

    /// Get context details
    pub async fn get_context(&self, path: &str) -> Option<MemoryContext> {
        self.graph_storage.get_context(path).await.ok()
    }

    /// Get service statistics
    pub async fn get_stats(&self) -> MemoryResult<MemoryStats> {
        let mut stats = self.stats.read().await.clone();

        // Update cache stats
        let cache_stats = self.cache.get_stats();
        stats.cache_hit_rate = cache_stats.avg_hit_rate;

        // Get storage stats
        let storage_stats = self.graph_storage.get_stats().await?;
        stats.total_memories = storage_stats.total_memories;
        stats.total_contexts = storage_stats.total_contexts;

        // Compute active memories strictly above threshold
        let thr = self.config.brain.importance_threshold;
        let active = self.graph_storage.get_active_count(thr).await?;
        stats.active_memories = active;

        Ok(stats)
    }

    /// Simple search method for basic text queries
    pub async fn search(&self, query: &str, limit: usize) -> MemoryResult<Vec<MemoryCell>> {
        // Create a simple memory query
        let memory_query = MemoryQuery {
            text: query.to_string(),
            context_hint: None,
            memory_types: None,
            limit: Some(limit),
            min_importance: Some(0.01),
            time_range: None,
            similarity_threshold: Some(0.20), // Lower threshold for 512D space
            include_related: false,
        };

        // Use recall_memory for the search
        let recalled = self.recall_memory(memory_query).await?;

        // Combine all layers and deduplicate
        let mut results = Vec::new();
        results.extend(recalled.semantic_layer);
        results.extend(recalled.contextual_layer);
        results.extend(recalled.detailed_layer);

        // Deduplicate by ID
        let mut seen = std::collections::HashSet::new();
        results.retain(|m| seen.insert(m.id));

        // Sort by importance and limit (handle NaN safely)
        results.sort_by(|a, b| {
            let a_imp = if a.importance.is_finite() {
                a.importance
            } else {
                0.0
            };
            let b_imp = if b.importance.is_finite() {
                b.importance
            } else {
                0.0
            };
            b_imp
                .partial_cmp(&a_imp)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        Ok(results)
    }

    /// Search memories within a specific context
    pub async fn search_by_context(
        &self,
        context_path: &str,
        query: Option<&str>,
        limit: usize,
    ) -> MemoryResult<Vec<MemoryCell>> {
        // Get memories from the context
        let mut memories = self
            .graph_storage
            .get_memories_in_context(context_path, limit * 2)
            .await?;

        // If query is provided, filter by semantic similarity
        if let Some(query_text) = query {
            // Generate embedding for the query
            let query_embedding = self
                .embedding_service
                .embed(query_text, crate::embedding::TaskType::Query)
                .await
                .map_err(|e| {
                    MemoryError::Embedding(format!("Failed to generate query embedding: {}", e))
                })?;

            // Calculate similarity scores with memories
            let mut scored_memories: Vec<(f32, MemoryCell)> = memories
                .into_iter()
                .map(|memory| {
                    let similarity = crate::simd_search::cosine_similarity_simd(
                        &query_embedding,
                        &memory.embedding,
                    );
                    (similarity, memory)
                })
                .filter(|(score, _)| *score > 0.05) // Filter by minimum relevance
                .collect();

            // Sort by relevance (handle NaN safely)
            scored_memories.sort_by(|a, b| {
                let a_score = if a.0.is_finite() { a.0 } else { 0.0 };
                let b_score = if b.0.is_finite() { b.0 } else { 0.0 };
                b_score
                    .partial_cmp(&a_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Extract memories and limit
            memories = scored_memories
                .into_iter()
                .take(limit)
                .map(|(_, memory)| memory)
                .collect();
        } else {
            // Sort by importance if no query (handle NaN safely)
            memories.sort_by(|a, b| {
                let a_imp = if a.importance.is_finite() {
                    a.importance
                } else {
                    0.0
                };
                let b_imp = if b.importance.is_finite() {
                    b.importance
                } else {
                    0.0
                };
                b_imp
                    .partial_cmp(&a_imp)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Limit results
            memories.truncate(limit);
        }

        Ok(memories)
    }

    /// Get recent memories
    pub async fn get_recent(
        &self,
        limit: usize,
        context: Option<&str>,
    ) -> MemoryResult<Vec<MemoryCell>> {
        // Get recent memories from storage
        let mut memories = if let Some(context_path) = context {
            // Get recent from specific context
            self.graph_storage
                .get_memories_in_context(context_path, limit)
                .await?
        } else {
            // Get recent from all contexts - use search as workaround
            // TODO: Implement proper get_recent_memories in GraphStorage
            let contexts = self.graph_storage.list_contexts().await?;
            let mut all_memories = Vec::new();

            // Get memories from first few contexts (limited for performance)
            for context_path in contexts.iter().take(5) {
                let context_memories = self
                    .graph_storage
                    .get_memories_in_context(context_path, limit / 5)
                    .await
                    .unwrap_or_default();
                all_memories.extend(context_memories);
            }

            all_memories
        };

        // Sort by last_accessed time (most recent first)
        memories.sort_by(|a, b| b.last_accessed.cmp(&a.last_accessed));
        memories.truncate(limit);

        Ok(memories)
    }

    /// Perform detailed search with parallel processing
    /// Retrieves related memories from contexts of top-ranked contextual results
    async fn perform_detailed_search(
        &self,
        contextual_results: &[MemoryCell],
    ) -> MemoryResult<Vec<MemoryCell>> {
        use futures::future::join_all;
        use std::collections::HashSet;
        use tracing::{debug, warn};

        // Early return if no contextual results to process
        if contextual_results.is_empty() {
            debug!("No contextual results provided for detailed search");
            return Ok(Vec::new());
        }

        // Take only top 3 contextual memories to avoid excessive load
        let top_memories = &contextual_results[..contextual_results.len().min(3)];

        // Create parallel tasks for each memory's detailed search
        let search_tasks = top_memories
            .iter()
            .filter_map(|memory| {
                // Skip memories with empty or invalid context paths
                if memory.context_path.is_empty() {
                    debug!("Skipping memory {} with empty context path", memory.id);
                    return None;
                }

                Some(self.get_context_memories(&memory.context_path, memory.id))
            })
            .collect::<Vec<_>>();

        // Execute all searches in parallel
        let results = join_all(search_tasks).await;

        // Collect successful results
        let mut detailed_memories = Vec::new();
        for result in results {
            match result {
                Ok(memories) => detailed_memories.extend(memories),
                Err(e) => {
                    warn!("Failed to retrieve context memories: {}", e);
                    // Continue with other results rather than failing entirely
                }
            }
        }

        // Efficient deduplication using HashSet to avoid O(n²) complexity
        let mut seen_ids = HashSet::new();
        detailed_memories.retain(|memory| seen_ids.insert(memory.id));

        // Sort by importance (descending order)
        // Handle NaN values gracefully - treat as lowest importance (0.0)
        detailed_memories.sort_by(|a, b| {
            let importance_a = if a.importance.is_finite() {
                a.importance
            } else {
                0.0
            };
            let importance_b = if b.importance.is_finite() {
                b.importance
            } else {
                0.0
            };
            importance_b
                .partial_cmp(&importance_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to top 10 detailed memories to prevent information overload
        detailed_memories.truncate(10);

        debug!(
            "Retrieved {} detailed memories from {} contexts",
            detailed_memories.len(),
            top_memories.len()
        );
        Ok(detailed_memories)
    }

    /// Get memories from a specific context (helper method for parallel processing)
    async fn get_context_memories(
        &self,
        context_path: &str,
        source_memory_id: Uuid,
    ) -> MemoryResult<Vec<MemoryCell>> {
        // Get context information
        match self.graph_storage.get_context(context_path).await {
            Ok(context) => {
                // Retrieve related memories from this context
                let mut context_memories = self
                    .graph_storage
                    .get_memories_in_context(&context.path, 5)
                    .await?;

                // Filter out the source memory to avoid duplication
                context_memories.retain(|m| m.id != source_memory_id);

                Ok(context_memories)
            }
            Err(e) => Err(e),
        }
    }

    /// Calculate confidence score for recall results
    fn calculate_confidence(&self, semantic: &[MemoryCell], contextual: &[MemoryCell]) -> f32 {
        if semantic.is_empty() {
            return 0.0;
        }

        let semantic_score = semantic.iter().take(5).map(|m| m.importance).sum::<f32>() / 5.0;

        let contextual_score = if !contextual.is_empty() {
            contextual
                .iter()
                .take(5)
                .map(|m| m.importance * 0.7)
                .sum::<f32>()
                / 5.0
        } else {
            0.0
        };

        (semantic_score * 0.6 + contextual_score * 0.4).min(1.0)
    }

    /// Generate hash for query caching
    fn hash_query(&self, query: &MemoryQuery) -> String {
        let mut hasher = DefaultHasher::new();
        query.text.hash(&mut hasher);
        query.context_hint.hash(&mut hasher);
        query.limit.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    // Compatibility methods for tests and external APIs

    /// Convenience method for storing memory (alias for store_memory)
    pub async fn store(
        &self,
        content: String,
        context_hint: Option<String>,
        _metadata: Option<std::collections::HashMap<String, String>>, // Ignored for now
    ) -> MemoryResult<Uuid> {
        self.store_memory(content, context_hint).await
    }

    /// Convenience method for recalling memory (alias for recall_memory)
    pub async fn recall(&self, query: MemoryQuery) -> MemoryResult<RecalledMemory> {
        self.recall_memory(query).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_storage_and_retrieval() {
        // Test configuration
        let _config = Config::default();

        // This would require mocking the dependencies
        // Full implementation would include proper test setup
    }
}
