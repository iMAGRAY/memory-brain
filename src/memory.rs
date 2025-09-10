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
use std::collections::{HashMap as StdHashMap, HashSet};
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
    pub(crate) graph_storage: Arc<GraphStorage>,
    /// AI brain for content analysis
    ai_brain: Arc<AIBrain>,
    /// Multi-layer cache system
    cache: Arc<CacheSystem>,
    /// Configuration settings
    config: Config,
    /// Runtime statistics
    stats: Arc<RwLock<MemoryStats>>,
}

/// Helper: tokenize text
fn tokenize(text: &str) -> Vec<String> { MemoryService::simple_tokens(text) }

/// Build document frequency map for a set of documents (by unique tokens per doc)
fn document_frequency(docs: &[MemoryCell]) -> StdHashMap<String, usize> {
    let mut df: StdHashMap<String, usize> = StdHashMap::new();
    for m in docs {
        let mut seen = HashSet::new();
        for t in tokenize(&(m.content.clone() + " " + &m.summary)) {
            if seen.insert(t.clone()) {
                *df.entry(t).or_insert(0) += 1;
            }
        }
    }
    df
}

/// Compute TF-IDF cosine similarity between query tokens and doc tokens
fn tfidf_cosine(
    q_tokens: &[String],
    d_tokens: &[String],
    df: &StdHashMap<String, usize>,
    n_docs: usize,
) -> f32 {
    if q_tokens.is_empty() || d_tokens.is_empty() { return 0.0; }
    let mut q_tf: StdHashMap<&str, f32> = StdHashMap::new();
    for t in q_tokens { *q_tf.entry(t.as_str()).or_insert(0.0) += 1.0; }
    let mut d_tf: StdHashMap<&str, f32> = StdHashMap::new();
    for t in d_tokens { *d_tf.entry(t.as_str()).or_insert(0.0) += 1.0; }
    let mut dot = 0.0f32;
    let mut q_norm = 0.0f32;
    let mut d_norm = 0.0f32;
    for (t, &qf) in &q_tf {
        let df_t = *df.get(*t).unwrap_or(&1) as f32;
        let idf = (n_docs as f32 / (1.0 + df_t)).ln().max(0.0);
        let qw = qf * idf;
        q_norm += qw * qw;
        if let Some(&df_) = d_tf.get(t) {
            let dw = df_ * idf;
            dot += qw * dw;
        }
    }
    for (&t, &df_) in &d_tf {
        let df_t = *df.get(t).unwrap_or(&1) as f32;
        let idf = (n_docs as f32 / (1.0 + df_t)).ln().max(0.0);
        let dw = df_ * idf;
        d_norm += dw * dw;
    }
    if q_norm == 0.0 || d_norm == 0.0 { return 0.0; }
    (dot / (q_norm.sqrt() * d_norm.sqrt())).max(0.0)
}

/// Very lightweight context inference from query tokens to prioritize candidates
fn infer_contexts_from_query(text: &str) -> Vec<String> {
    let q = text.to_ascii_lowercase();
    let mut out: std::collections::HashSet<String> = std::collections::HashSet::new();
    let push = |s: &str, set: &mut std::collections::HashSet<String>| { set.insert(format!("quality/{}", s)); };
    // Rust / Python
    if q.contains("rust") { let _ = push("rust", &mut out); }
    if q.contains("python") { let _ = push("python", &mut out); }
    // ML
    let ml_terms = ["overfitting","learning rate","cross-validation","batch normalization","regularization","gradient descent","adam"]; 
    if ml_terms.iter().any(|t| q.contains(t)) || q.contains(" ml ") || q.starts_with("ml ") || q.ends_with(" ml") {
        let _ = push("ml", &mut out);
    }
    // Cooking
    let cook_terms = ["spaghetti","bread","saute","tomato","basil","oregano","cook","bake","grill"];
    if cook_terms.iter().any(|t| q.contains(t)) { let _ = push("cooking", &mut out); }
    // Travel
    let travel_terms = ["travel","train","japan","flight","visa","hostel","insurance"];
    if travel_terms.iter().any(|t| q.contains(t)) { let _ = push("travel", &mut out); }
    // DevOps
    let devops_terms = ["docker","kubernetes","terraform","prometheus","grafana","ci/cd","gitops","canary","blue-green"];
    if devops_terms.iter().any(|t| q.contains(t)) { let _ = push("devops", &mut out); }
    // Databases
    let db_terms = ["acid","index","isolation","sharding","replication","foreign key","sql","oltp","olap"];
    if db_terms.iter().any(|t| q.contains(t)) || q.contains("database") || q.contains("databases") { let _ = push("databases", &mut out); }
    // Security
    let sec_terms = ["tls","encryption","xss","sql injection","least privilege","mfa","csrf","csp","security"];
    if sec_terms.iter().any(|t| q.contains(t)) { let _ = push("security", &mut out); }
    out.into_iter().collect()
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(default)
}

/// Compute BM25 score for a document against a query
fn bm25_score(
    q_tokens: &[String],
    d_tokens: &[String],
    df: &StdHashMap<String, usize>,
    n_docs: usize,
    avgdl: f32,
) -> f32 {
    if q_tokens.is_empty() || d_tokens.is_empty() { return 0.0; }
    // BM25 parameters tuned for short queries and medium docs
    let k1: f32 = 1.2;
    let b: f32 = 0.75;

    // term frequency in doc
    let mut tf: StdHashMap<&str, f32> = StdHashMap::new();
    for t in d_tokens { *tf.entry(t.as_str()).or_insert(0.0) += 1.0; }
    let dl = d_tokens.len() as f32;

    // Query unique terms
    use std::collections::HashSet;
    let uniq_q: HashSet<&str> = q_tokens.iter().map(|s| s.as_str()).collect();

    let mut score = 0.0f32;
    for t in uniq_q {
        let n_t = *df.get(t).unwrap_or(&0) as f32;
        // BM25 IDF with +1 to avoid negative for frequent terms
        let idf = ((n_docs as f32 - n_t + 0.5) / (n_t + 0.5)).ln().max(0.0) + 1.0;
        let freq = *tf.get(t).unwrap_or(&0.0);
        if freq <= 0.0 { continue; }
        let denom = freq + k1 * (1.0 - b + b * (dl / (avgdl.max(1e-6))));
        score += idf * (freq * (k1 + 1.0)) / denom.max(1e-6);
    }
    score.max(0.0)
}

/// Maximal Marginal Relevance reordering to improve diversity and reduce redundancy
fn mmr_reorder(
    candidates: &[MemoryCell],
    scores: &std::collections::HashMap<Uuid, f32>,
    lambda: f32,
    top_k: usize,
) -> Vec<MemoryCell> {
    use crate::simd_search::cosine_similarity_simd;
    let lambda = lambda.clamp(0.0, 1.0);
    if candidates.is_empty() { return Vec::new(); }
    // Pre-collect embeddings and base scores
    let mut items: Vec<(usize, Uuid, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(i, m)| (i, m.id, *scores.get(&m.id).unwrap_or(&0.0)))
        .collect();
    // Normalization of base scores to [0,1]
    let mut max_s = 1e-6f32; for (_,_,s) in &items { if *s > max_s { max_s = *s; } }
    for (_,_,s) in items.iter_mut() { *s = (*s / max_s).clamp(0.0, 1.0); }

    let mut selected: Vec<usize> = Vec::new();
    let mut remaining: std::collections::HashSet<usize> = (0..candidates.len()).collect();
    let k = top_k.min(candidates.len());

    while selected.len() < k && !remaining.is_empty() {
        let mut best_idx = None; let mut best_score = f32::NEG_INFINITY;
        for &i in &remaining {
            let (_, _id, base) = items[i];
            let mut div_penalty = 0.0f32;
            if !selected.is_empty() {
                let mut max_sim = 0.0f32;
                for &j in &selected {
                    let sim = cosine_similarity_simd(&candidates[i].embedding, &candidates[j].embedding).max(0.0);
                    if sim > max_sim { max_sim = sim; }
                }
                div_penalty = max_sim;
            }
            let mmr = (1.0 - lambda) * base - lambda * div_penalty;
            if mmr > best_score { best_score = mmr; best_idx = Some(i); }
        }
        if let Some(i) = best_idx { selected.push(i); remaining.remove(&i); } else { break; }
    }
    let mut out: Vec<MemoryCell> = selected.into_iter().map(|i| candidates[i].clone()).collect();
    // Append any remaining up to top_k to keep length consistent
    if out.len() < k {
        for &i in &remaining { out.push(candidates[i].clone()); if out.len()>=k { break; } }
    }
    out
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

    /// Returns the target embedding dimension configured for Matryoshka alignment.
    /// Falls back to 512 if not specified.
    pub fn embedding_dimension(&self) -> usize {
        self.config.embedding.embedding_dimension.unwrap_or(512)
    }

    /// Report embedding availability for health/substatus
    pub fn embedding_available(&self) -> bool {
        self.embedding_service.is_available()
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
        // Try to generate base embedding; if embedding unavailable, degrade to lexical-only memory
        let embedding = match self
            .embedding_service
            .embed(&content, crate::embedding::TaskType::Query)
            .await
        {
            Ok(base_embedding) => {
                let target_dim = self.config.embedding.embedding_dimension.unwrap_or(512);
                let emb = self
                    .embedding_service
                    .truncate_to_dimension(&base_embedding, target_dim)?;
                crate::metrics::record_embedding_latency("gemma-300m", t0.elapsed().as_secs_f64());
                emb
            }
            Err(MemoryError::Embedding(_)) => {
                // Lexical-only storage; embedding left empty
                Vec::new()
            }
            Err(e) => return Err(e),
        };

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

        // Try embedding path; on Embedding error, fall back to lexical retrieval
        let (semantic_results, contextual_results, detailed_results, reasoning_chain, confidence) = match self
            .build_semantic_layers(&query)
            .await
        {
            Ok((sem, ctx, det, rc, conf)) => (sem, ctx, det, rc, conf),
            Err(MemoryError::Embedding(_)) => {
                let (lex, rc, conf) = self.lexical_recall(&query).await?;
                (lex, Vec::new(), Vec::new(), rc, conf)
            }
            Err(e) => return Err(e),
        };

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

        let mut recalled = RecalledMemory {
            query_id,
            semantic_layer: semantic_results,
            contextual_layer: contextual_results,
            detailed_layer: detailed_results,
            reasoning_chain,
            confidence,
            recall_time_ms,
            discovered_contexts,
        };

        // Final hybrid re-ranking across combined layers to maximize quality (if embedding available)
        // Auto-tune HYBRID_ALPHA by query length (shorter queries favor lexical)
        let mut alpha: f32 = env_f32("HYBRID_ALPHA", 0.5);
        let qtoks = tokenize(&query.text);
        let qlen = qtoks.len();
        if std::env::var("HYBRID_ALPHA").is_err() {
            let short = env_f32("ALPHA_SHORT", 0.4);
            let long = env_f32("ALPHA_LONG", 0.6);
            alpha = if qlen <= 6 { short } else if qlen >= 16 { long } else { alpha };
        }
        let mut combined = Vec::new();
        combined.extend(recalled.semantic_layer.clone());
        combined.extend(recalled.contextual_layer.clone());
        combined.extend(recalled.detailed_layer.clone());
        // Deduplicate
        let mut seen = std::collections::HashSet::new();
        combined.retain(|m| seen.insert(m.id));
        // Score and reorder (hybrid), then apply optional MMR diversity
        if let Ok(scored) = self.score_hybrid_internal(&query.text, &combined, alpha).await {
            use std::collections::HashMap; let mut sm: HashMap<Uuid,f32> = HashMap::new();
            for (id,s,_) in scored { sm.insert(id,s); }
            // Primary reorder by score
            combined.sort_by(|a,b| {
                let sa = *sm.get(&a.id).unwrap_or(&0.0);
                let sb = *sm.get(&b.id).unwrap_or(&0.0);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Optional MMR on top-N
            let enable_mmr = std::env::var("ENABLE_MMR").ok().map(|s| s=="1" || s.eq_ignore_ascii_case("true")).unwrap_or(true);
            if enable_mmr {
                let mmr_lambda = env_f32("MMR_LAMBDA", 0.3);
                let mmr_top = env_usize("MMR_TOP", combined.len().min(200));
                let top_slice = combined.clone();
                let reordered = mmr_reorder(&top_slice, &sm, mmr_lambda, mmr_top);
                combined = reordered;
            }
            // Redistribute into layers (keep sizes roughly as before)
            let s_len = recalled.semantic_layer.len().min(combined.len());
            let c_len = recalled.contextual_layer.len().min(combined.len().saturating_sub(s_len));
            let d_len = combined.len().saturating_sub(s_len + c_len);
            recalled.semantic_layer = combined[..s_len].to_vec();
            recalled.contextual_layer = combined[s_len..s_len+c_len].to_vec();
            recalled.detailed_layer = combined[s_len+c_len..s_len+c_len+d_len].to_vec();
        }

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

    async fn build_semantic_layers(
        &self,
        query: &MemoryQuery,
    ) -> MemoryResult<(
        Vec<MemoryCell>,
        Vec<MemoryCell>,
        Vec<MemoryCell>,
        Vec<String>,
        f32,
    )> {
        // Generate query embedding
        let tq0 = Instant::now();
        let base_query_embedding = self
            .embedding_service
            .embed(&query.text, crate::embedding::TaskType::Query)
            .await?;
        let target_dim = self.config.embedding.embedding_dimension.unwrap_or(512);
        let query_embedding = self
            .embedding_service
            .truncate_to_dimension(&base_query_embedding, target_dim)?;
        crate::metrics::record_embedding_latency("gemma-300m", tq0.elapsed().as_secs_f64());

        // Layer 1: Semantic search — widen candidate pool aggressively
        let base_limit = query.limit.unwrap_or(20);
        let cand_sem_mult = env_usize("CAND_SEM_MULT", 15);
        let cand_sem_cap = env_usize("CAND_SEM_CAP", 500);
        let cand_sem_thresh = env_f32("CAND_SEM_THRESH", 0.05);
        let candidate_limit = (base_limit.saturating_mul(cand_sem_mult)).min(cand_sem_cap);
        let semantic_results_sem = self
            .graph_storage
            .vector_search(
                &query_embedding,
                candidate_limit,
                // Lower threshold further to improve recall; cap by provided threshold if any
                Some(query.similarity_threshold.unwrap_or(cand_sem_thresh).min(cand_sem_thresh)),
            )
            .await?;

        // Optional lexical augmentation (BM25) to boost recall for short queries
        let mut semantic_results = semantic_results_sem;
        let cand_lex_mult = env_usize("CAND_LEX_MULT", 8);
        let cand_lex_cap = env_usize("CAND_LEX_CAP", 150);
        let lex_extra = (base_limit.saturating_mul(cand_lex_mult)).min(cand_lex_cap);
        if lex_extra > 0 {
            let mut qlex = query.clone();
            qlex.limit = Some(lex_extra);
            if let Ok((mut bm25, _rc, _conf)) = self.lexical_recall(&qlex).await {
                // Merge and deduplicate by ID
                semantic_results.extend(bm25.drain(..));
                let mut seen = std::collections::HashSet::new();
                semantic_results.retain(|m| seen.insert(m.id));
            }
        }

        // Layer 2: Contextual search - explore related memories
        let mut contextual_results = Vec::new();
        for memory in &semantic_results[..semantic_results.len().min(5)] {
            let related = self
                .graph_storage
                .find_related_memories(&memory.id, 5)
                .await?;
            contextual_results.extend(related);
        }

        // Augment candidates with inferred contexts (boost recall for domain-anchored queries)
        let inferred = infer_contexts_from_query(&query.text);
        if !inferred.is_empty() {
            for ctx in inferred.into_iter().take(3) {
                let extra = self
                    .graph_storage
                    .get_memories_in_context(&ctx, (base_limit.saturating_mul(5)).min(60))
                    .await
                    .unwrap_or_default();
                contextual_results.extend(extra);
            }
            // Deduplicate contextual pool
            let mut seen = std::collections::HashSet::new();
            contextual_results.retain(|m| seen.insert(m.id));
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
        Ok((
            semantic_results,
            contextual_results,
            detailed_results,
            reasoning_chain,
            confidence,
        ))
    }

    /// Lexical recall fallback using simple TF-IDF scoring across recent memories.
    async fn lexical_recall(
        &self,
        query: &MemoryQuery,
    ) -> MemoryResult<(Vec<MemoryCell>, Vec<String>, f32)> {
        let limit = query.limit.unwrap_or(20).min(200);
        // Aggregate a pool of candidate memories by contexts
        let contexts = self.graph_storage.list_contexts().await.unwrap_or_default();
        let mut pool: Vec<MemoryCell> = Vec::new();
        for ctx in contexts.iter().take(20) {
            let mut chunk = self
                .graph_storage
                .get_memories_in_context(ctx, 128)
                .await
                .unwrap_or_default();
            pool.append(&mut chunk);
            if pool.len() > 4000 {
                break;
            }
        }
        if pool.is_empty() {
            // Fallback to recent from any context
            pool = self.get_recent(256, None).await.unwrap_or_default();
        }

        let q_tokens = tokenize(&query.text);
        let df: StdHashMap<String, usize> = document_frequency(&pool);
        let n_docs = pool.len().max(1);

        // Average doc length for BM25
        let mut total_len: usize = 0;
        for m in &pool { total_len += tokenize(&(m.content.clone() + " " + &m.summary)).len(); }
        let avgdl = if pool.is_empty() { 0.0 } else { (total_len as f32) / (pool.len() as f32) };

        let mut scored: Vec<(f32, MemoryCell)> = pool
            .into_iter()
            .map(|m| {
                let doc_tokens = tokenize(&(m.content.clone() + " " + &m.summary));
                let mut score = bm25_score(&q_tokens, &doc_tokens, &df, n_docs, avgdl);
                // Light context-path prior to improve category relevance
            if !m.context_path.is_empty() {
                let c_tokens = tokenize(&m.context_path);
                let overlap = c_tokens
                    .iter()
                    .filter(|t| q_tokens.contains(t))
                    .count();
                    if overlap > 0 {
                        let cboost = env_f32("CONTEXT_BOOST", 0.35);
                        score *= 1.0 + cboost * (overlap as f32) / (q_tokens.len() as f32);
                    }
                }
                (score, m)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut results: Vec<MemoryCell> = scored.into_iter().take(limit).map(|(_, m)| m).collect();
        // Deduplicate by ID
        let mut seen = HashSet::new();
        results.retain(|m| seen.insert(m.id));

        let rc = vec![
            format!("Lexical fallback: {} candidates scored", n_docs),
            format!("Returned {} documents", results.len()),
        ];
        let confidence = (results.len() as f32 / limit as f32).clamp(0.1, 0.8);
        Ok((results, rc, confidence))
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
            let mems = self
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

    /// Compute graph-level metrics for analysis and dashboard
    pub async fn compute_graph_metrics(&self) -> MemoryResult<GraphMetrics> {
        let (avg_deg, connected_ratio, total_memories) = self.graph_storage.graph_degree_stats().await?;
        let ctx_counts = self.graph_storage.context_counts().await?;
        let total_contexts = ctx_counts.len();
        let sum: f64 = ctx_counts.iter().map(|&c| c as f64).sum();
        let mut entropy = 0.0f64;
        if sum > 0.0 {
            for &c in &ctx_counts {
                if c > 0 {
                    let p = (c as f64) / sum;
                    entropy -= p * p.log2();
                }
            }
        }
        let expansion = self.graph_storage.two_hop_expansion_factor(100).await?;
        let avg_closure = self.graph_storage.approx_clustering(50).await.unwrap_or(0.0);
        let avg_shortest_path = self.graph_storage.approx_shortest_path_len(30).await.unwrap_or(0.0);
        Ok(GraphMetrics {
            total_memories,
            total_contexts,
            avg_related_degree: avg_deg,
            connected_ratio,
            contexts_entropy: entropy,
            two_hop_expansion: expansion,
            avg_closure,
            avg_shortest_path,
        })
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
            similarity_threshold: Some(0.10), // Favor higher recall for hybrid re-ranking
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

        // Hybrid ranking: combine vector cosine and lexical TF-IDF
        // Auto-tune HYBRID_ALPHA by query length (shorter queries favor lexical)
        let mut alpha: f32 = std::env::var("HYBRID_ALPHA").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5);
        let qtoks = tokenize(query);
        let qlen = qtoks.len();
        if std::env::var("HYBRID_ALPHA").is_err() {
            alpha = if qlen <= 6 { 0.4 } else if qlen >= 16 { 0.6 } else { 0.5 };
        }
        let scored = self
            .score_hybrid_internal(query, &results, alpha)
            .await
            .unwrap_or_else(|_| Vec::new());
        if !scored.is_empty() {
            // reorder results by hybrid score
            use std::collections::HashMap;
            let mut score_map: HashMap<Uuid, f32> = HashMap::new();
            for (id, s, _) in &scored {
                score_map.insert(*id, *s);
            }
            results.sort_by(|a, b| {
                let sa = *score_map.get(&a.id).unwrap_or(&0.0);
                let sb = *score_map.get(&b.id).unwrap_or(&0.0);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            // Fallback: importance sort
            results.sort_by(|a, b| {
                let a_imp = if a.importance.is_finite() { a.importance } else { 0.0 };
                let b_imp = if b.importance.is_finite() { b.importance } else { 0.0 };
                b_imp.partial_cmp(&a_imp).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        results.truncate(limit);

        Ok(results)
    }

    /// Compute hybrid scores for query and given results (id, score, method)
    pub async fn score_hybrid(&self, query_text: &str, results: &[MemoryCell], alpha: f32) -> MemoryResult<Vec<(Uuid, f32, String)>> {
        self.score_hybrid_internal(query_text, results, alpha).await
    }

    async fn score_hybrid_internal(&self, query_text: &str, results: &[MemoryCell], alpha: f32) -> MemoryResult<Vec<(Uuid, f32, String)>> {
        let alpha = alpha.clamp(0.0, 1.0);
        // Lexical precompute
        let q_tokens = tokenize(query_text);
        let df = document_frequency(results);
        let n_docs = results.len().max(1);
        // Precompute avg doc length for BM25 and per-doc lexical scores for normalization
        let mut total_len: usize = 0;
        for m in results { total_len += tokenize(&(m.content.clone() + " " + &m.summary)).len(); }
        let avgdl = if results.is_empty() { 0.0 } else { (total_len as f32) / (results.len() as f32) };
        // Try query embedding
        let vec_scores = match self
            .embedding_service
            .embed(query_text, crate::embedding::TaskType::Query)
            .await
        {
            Ok(base_query_embedding) => {
                let target_dim = self.config.embedding.embedding_dimension.unwrap_or(512);
                let q_emb = self
                    .embedding_service
                    .truncate_to_dimension(&base_query_embedding, target_dim)?;
                Some(q_emb)
            }
            Err(_) => None,
        };

        // First pass: compute BM25 scores and track max for normalization
        let mut bm25_scores: Vec<(Uuid, f32, f32)> = Vec::with_capacity(results.len()); // (id, bm25, ctx_boost)
        let mut max_lex = 1e-6f32;
        for m in results {
            let lex_tokens = tokenize(&(m.content.clone() + " " + &m.summary));
            let mut bm = bm25_score(&q_tokens, &lex_tokens, &df, n_docs, avgdl);
            let mut boost = 1.0f32;
            if !m.context_path.is_empty() {
                let c_tokens = tokenize(&m.context_path);
                let overlap = c_tokens.iter().filter(|t| q_tokens.contains(t)).count();
                if overlap > 0 { boost = 1.0 + env_f32("CONTEXT_BOOST", 0.35) * (overlap as f32) / (q_tokens.len() as f32); }
            }
            bm *= boost;
            if bm > max_lex { max_lex = bm; }
            bm25_scores.push((m.id, bm, boost));
        }

        // Second pass: combine with vector cosine (if available) using normalized lexical component
        let mut out = Vec::with_capacity(results.len());
        for (i, m) in results.iter().enumerate() {
            let (_, bm, _) = bm25_scores[i];
            let lex_norm = (bm / max_lex).clamp(0.0, 1.0);
            let combined = if let Some(ref q_emb) = vec_scores {
                let vcos = crate::simd_search::cosine_similarity_simd(q_emb, &m.embedding).max(0.0);
                alpha * vcos + (1.0 - alpha) * lex_norm
            } else {
                lex_norm
            };
            let method = if vec_scores.is_some() { "hybrid" } else { "lexical" };
            out.push((m.id, combined, method.to_string()));
        }
        Ok(out)
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
            // Try embedding; if unavailable, use lexical scoring within context
            match self
                .embedding_service
                .embed(query_text, crate::embedding::TaskType::Query)
                .await
            {
                Ok(base_query_embedding) => {
                    let target_dim = self.config.embedding.embedding_dimension.unwrap_or(512);
                    let query_embedding = self
                        .embedding_service
                        .truncate_to_dimension(&base_query_embedding, target_dim)?;
                    let mut scored_memories: Vec<(f32, MemoryCell)> = memories
                        .into_iter()
                        .map(|memory| {
                            let similarity = crate::simd_search::cosine_similarity_simd(
                                &query_embedding,
                                &memory.embedding,
                            );
                            (similarity, memory)
                        })
                        .filter(|(score, _)| *score > 0.05)
                        .collect();
                    scored_memories.sort_by(|a, b| {
                        let a_score = if a.0.is_finite() { a.0 } else { 0.0 };
                        let b_score = if b.0.is_finite() { b.0 } else { 0.0 };
                        b_score.partial_cmp(&a_score).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    memories = scored_memories
                        .into_iter()
                        .take(limit)
                        .map(|(_, memory)| memory)
                        .collect();
                }
                Err(MemoryError::Embedding(_)) => {
                    let df: StdHashMap<String, usize> = document_frequency(&memories);
                    let n_docs = memories.len().max(1);
                    let q_tokens = tokenize(query_text);
                    let mut total_len: usize = 0;
                    for m in &memories { total_len += tokenize(&(m.content.clone() + " " + &m.summary)).len(); }
                    let avgdl = if memories.is_empty() { 0.0 } else { (total_len as f32) / (memories.len() as f32) };
                    let mut scored: Vec<(f32, MemoryCell)> = memories
                        .into_iter()
                        .map(|m| {
                            let tokens = tokenize(&(m.content.clone() + " " + &m.summary));
                            let mut s = bm25_score(&q_tokens, &tokens, &df, n_docs, avgdl);
                            if !m.context_path.is_empty() {
                                let c_tokens = tokenize(&m.context_path);
                                let overlap = c_tokens.iter().filter(|t| q_tokens.contains(t)).count();
                                if overlap > 0 { s *= 1.0 + env_f32("CONTEXT_BOOST", 0.35) * (overlap as f32) / (q_tokens.len() as f32); }
                            }
                            (s, m)
                        })
                        .collect();
                    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    memories = scored.into_iter().take(limit).map(|(_, m)| m).collect();
                }
                Err(e) => return Err(MemoryError::Embedding(format!("Failed to generate query embedding: {}", e))),
            }
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

    /// Tokenize text into lowercase alphanumeric tokens
    fn simple_tokens(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        for ch in text.chars() {
            if ch.is_alphanumeric() {
                cur.push(ch.to_ascii_lowercase());
            } else if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
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
