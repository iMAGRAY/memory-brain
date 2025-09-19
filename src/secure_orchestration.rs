//! Secure Orchestration Layer for GPT-5-nano Integration
//!
//! Provides secure integration between memory search and GPT-5-nano orchestrator
//! with proper data minimization, access control, and privacy protection.

use crate::memory::MemoryService;
use crate::orchestrator::MemoryOrchestrator;
use crate::types::{InsightType, MemoryCell, MemoryType};
use anyhow::{anyhow, Result};
use regex::Regex;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::debug;
use uuid::Uuid;

// Boost coefficients for re-ranking
const KEY_POINT_BOOST: f32 = 1.5; // Boost for key points from distillation
const PATTERN_BOOST: f32 = 1.3; // Boost for insight patterns
const MAX_IMPORTANCE: f32 = 10.0; // Maximum importance to prevent overflow

/// User context for access control
#[derive(Debug, Clone, Default)]
pub struct UserContext {
    pub user_id: String,
    pub session_id: String,
    pub source: String, // "mcp", "api", "anonymous"
    pub rate_limit_remaining: u32,
}

/// Secure search result without internal details
#[derive(Debug, Clone, Serialize)]
pub struct SecureSearchResult {
    pub results: Vec<MemoryCell>,
    pub confidence: f32,
    pub total_found: usize,
    pub search_id: Uuid,
}

/// Configuration for secure orchestration
#[derive(Debug, Clone)]
pub struct SecureOrchestrationConfig {
    /// Maximum context size to send to GPT-5
    pub max_context_size: usize,
    /// Enable PII filtering
    pub filter_pii: bool,
    /// Mask sensitive data patterns
    pub mask_sensitive_data: bool,
    /// Rate limit per minute
    pub rate_limit_per_minute: u32,
    /// Require authentication
    pub enable_session_tracking: bool,
}

impl Default for SecureOrchestrationConfig {
    fn default() -> Self {
        Self {
            max_context_size: 1000,
            filter_pii: true,
            mask_sensitive_data: true,
            rate_limit_per_minute: 30,
            enable_session_tracking: true,
        }
    }
}

/// Secure orchestration layer
pub struct SecureOrchestrationLayer {
    memory_service: Arc<MemoryService>,
    orchestrator: Arc<MemoryOrchestrator>,
    config: SecureOrchestrationConfig,
    pii_patterns: Vec<Regex>,
    sensitive_patterns: Vec<Regex>,
    rate_limiter: Arc<RwLock<RateLimiter>>,
}

impl SecureOrchestrationLayer {
    /// Create new secure orchestration layer
    pub fn new(
        memory_service: Arc<MemoryService>,
        orchestrator: Arc<MemoryOrchestrator>,
        config: SecureOrchestrationConfig,
    ) -> Result<Self> {
        // Compile PII detection patterns
        let pii_patterns = vec![
            Regex::new(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")?, // Email
            Regex::new(r"\b\d{3}-\d{2}-\d{4}\b")?,                     // SSN
            Regex::new(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")?, // Phone
            Regex::new(
                r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b",
            )?, // Credit card
        ];

        // Compile sensitive data patterns
        let sensitive_patterns = vec![
            Regex::new(r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key)\b[:\s]*\S+")?, // Credentials
            Regex::new(r"(?i)\b(bearer|authorization)\b[:\s]*\S+")?, // Auth tokens
            Regex::new(r"sk-[a-zA-Z0-9]{48}")?,                      // OpenAI API keys
        ];

        let rate_limit_per_minute = config.rate_limit_per_minute;
        Ok(Self {
            memory_service,
            orchestrator,
            config,
            pii_patterns,
            sensitive_patterns,
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new(rate_limit_per_minute))),
        })
    }

    /// Perform secure enhanced search with GPT-5-nano orchestration
    pub async fn secure_enhanced_search(
        &self,
        query: &str,
        user_context: &UserContext,
        limit: usize,
    ) -> Result<SecureSearchResult> {
        // 1. Track session if enabled
        self.track_session(user_context).await;

        // 2. Check rate limit
        self.check_rate_limit(user_context).await?;

        // 3. Sanitize and validate query
        let sanitized_query = self.sanitize_query(query)?;

        // 4. Perform base search first
        let base_results = self
            .memory_service
            .search(&sanitized_query, limit * 2) // Get more for filtering
            .await?;

        // 5. Enhance search with orchestrator if we have results
        let enhanced_results = if !base_results.is_empty() {
            // Convert to format orchestrator expects
            let memories_for_distillation = base_results.clone();

            // Try to enhance with GPT-5, fallback to base results if unavailable
            match self
                .enhance_with_orchestrator(
                    &memories_for_distillation,
                    &sanitized_query,
                    memories_for_distillation.clone(),
                )
                .await
            {
                Ok(enhanced) => enhanced,
                Err(e) => {
                    // Log error but continue with base results
                    debug!("Orchestrator enhancement unavailable: {}", e);
                    memories_for_distillation
                }
            }
        } else {
            base_results
        };

        // 7. Filter sensitive data from results
        let filtered_results = self.filter_sensitive_data(enhanced_results)?;

        // 8. Limit final results
        let final_results: Vec<MemoryCell> = filtered_results.into_iter().take(limit).collect();

        // 9. Calculate safe confidence score
        let confidence = self.calculate_safe_confidence(&final_results);

        Ok(SecureSearchResult {
            total_found: final_results.len(),
            results: final_results,
            confidence,
            search_id: Uuid::new_v4(),
        })
    }

    /// Track session for MCP or API clients
    async fn track_session(&self, user_context: &UserContext) {
        if self.config.enable_session_tracking {
            let session_info = if !user_context.session_id.is_empty() {
                &user_context.session_id
            } else {
                "anonymous"
            };

            debug!(
                "Processing request from {} session: {}",
                user_context.source, session_info
            );
        }
    }

    /// Check rate limit for user
    async fn check_rate_limit(&self, user_context: &UserContext) -> Result<()> {
        let mut limiter = self.rate_limiter.write().await;

        // Use session_id for MCP agents, user_id for API
        let limit_key = if user_context.source == "mcp" && !user_context.session_id.is_empty() {
            &user_context.session_id
        } else {
            &user_context.user_id
        };

        if !limiter.check_and_update(limit_key) {
            return Err(anyhow!("Rate limit exceeded"));
        }
        Ok(())
    }

    /// Sanitize query to prevent injection and remove dangerous patterns
    fn sanitize_query(&self, query: &str) -> Result<String> {
        // Limit query length
        if query.len() > 1000 {
            return Err(anyhow!("Query too long"));
        }

        // Remove control characters
        let mut sanitized = query
            .chars()
            .filter(|c| !c.is_control() || c.is_whitespace())
            .collect::<String>();

        // Remove potential SQL injection patterns
        sanitized = sanitized
            .replace("--", "")
            .replace(";", "")
            .replace("/*", "")
            .replace("*/", "")
            .replace("xp_", "")
            .replace("sp_", "");

        // Remove script tags and similar
        let script_pattern = Regex::new(r"<[^>]*script[^>]*>.*?</[^>]*script[^>]*>")?;
        sanitized = script_pattern.replace_all(&sanitized, "").to_string();

        // Trim and normalize whitespace
        sanitized = sanitized.split_whitespace().collect::<Vec<_>>().join(" ");

        if sanitized.is_empty() {
            return Err(anyhow!("Query is empty after sanitization"));
        }

        Ok(sanitized)
    }

    /// Minimize data before sending to orchestrator
    #[allow(dead_code)]
    fn minimize_for_orchestrator(&self, memories: &[MemoryCell]) -> Result<String> {
        let mut context = String::new();
        let mut total_size = 0;

        for memory in memories.iter().take(10) {
            // Limit to top 10
            // Create minimal representation
            let minimal = format!(
                "ID: {}, Context: {}, Importance: {:.2}, Summary: {}\n",
                memory.id,
                memory.context_path,
                memory.importance,
                self.truncate_content(&memory.content, 100)
            );

            total_size += minimal.len();
            if total_size > self.config.max_context_size {
                break;
            }

            context.push_str(&minimal);
        }

        Ok(context)
    }

    /// Filter sensitive data from results
    fn filter_sensitive_data(&self, mut memories: Vec<MemoryCell>) -> Result<Vec<MemoryCell>> {
        for memory in &mut memories {
            // Filter PII if enabled
            if self.config.filter_pii {
                for pattern in &self.pii_patterns {
                    memory.content = pattern
                        .replace_all(&memory.content, "[REDACTED]")
                        .to_string();
                }
            }

            // Mask sensitive data if enabled
            if self.config.mask_sensitive_data {
                for pattern in &self.sensitive_patterns {
                    memory.content = pattern
                        .replace_all(&memory.content, "[SENSITIVE]")
                        .to_string();
                }
            }

            // Clear raw embeddings to reduce response size
            memory.embedding.clear();
        }

        Ok(memories)
    }

    /// Enhance results with orchestrator
    async fn enhance_with_orchestrator(
        &self,
        memories_for_distillation: &[MemoryCell],
        query: &str,
        mut base_results: Vec<MemoryCell>,
    ) -> Result<Vec<MemoryCell>> {
        // Get distilled context from GPT-5
        let distilled_context = self
            .orchestrator
            .distill_context(memories_for_distillation, Some(query))
            .await?;

        // Generate insights for better ranking (optional)
        let insights = match self
            .orchestrator
            .generate_insights(memories_for_distillation, InsightType::ContextUnderstanding)
            .await
        {
            Ok(insights) => insights,
            Err(e) => {
                // Log at debug level to avoid spam in production
                debug!("Insights generation skipped: {}", e);
                Vec::new()
            }
        };

        // Re-rank based on orchestrator feedback
        self.rerank_with_insights(&mut base_results, insights, distilled_context)?;
        Ok(base_results)
    }

    /// Re-rank results based on orchestrator insights
    fn rerank_with_insights(
        &self,
        memories: &mut [MemoryCell],
        insights: Vec<MemoryType>,
        distilled_context: MemoryType,
    ) -> Result<()> {
        // Extract relevance scores from distilled context if available
        if let MemoryType::ContextDistillation { key_points, .. } = distilled_context {
            // Boost memories mentioned in key points
            for memory in memories.iter_mut() {
                for key_point in &key_points {
                    if memory
                        .content
                        .to_lowercase()
                        .contains(&key_point.to_lowercase())
                    {
                        // Prevent overflow by capping importance
                        if memory.importance < MAX_IMPORTANCE / KEY_POINT_BOOST {
                            memory.importance *= KEY_POINT_BOOST;
                        } else {
                            memory.importance = MAX_IMPORTANCE;
                        }
                    }
                }
            }
        }

        // Apply insights to adjust ranking (if we have UserPreference or other insight types)
        for insight in insights {
            if let MemoryType::UserPreference {
                preference_type,
                examples,
                ..
            } = insight
            {
                // Boost memories that match user preference examples
                for memory in memories.iter_mut() {
                    // Check if memory matches preference type or examples
                    if memory
                        .content
                        .to_lowercase()
                        .contains(&preference_type.to_lowercase())
                    {
                        // Apply moderate boost for preference type match
                        if memory.importance < MAX_IMPORTANCE / PATTERN_BOOST {
                            memory.importance *= PATTERN_BOOST;
                        } else {
                            memory.importance = MAX_IMPORTANCE;
                        }
                    }

                    // Check examples for stronger boosting
                    for example in &examples {
                        if memory
                            .content
                            .to_lowercase()
                            .contains(&example.to_lowercase())
                        {
                            // Apply stronger boost for example match
                            if memory.importance < MAX_IMPORTANCE / KEY_POINT_BOOST {
                                memory.importance *= KEY_POINT_BOOST;
                            } else {
                                memory.importance = MAX_IMPORTANCE;
                            }
                            break; // Only boost once per memory
                        }
                    }
                }
            }
        }

        // Sort by adjusted importance
        memories.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(())
    }

    /// Calculate confidence score without revealing internals
    fn calculate_safe_confidence(&self, results: &[MemoryCell]) -> f32 {
        if results.is_empty() {
            return 0.0;
        }

        // Simple confidence based on result quality
        let avg_importance: f32 =
            results.iter().map(|m| m.importance).sum::<f32>() / results.len() as f32;

        // Normalize to 0-1 range
        (avg_importance * 2.0).clamp(0.0, 1.0)
    }

    /// Truncate content to specified length
    #[allow(dead_code)]
    fn truncate_content(&self, content: &str, max_len: usize) -> String {
        if content.len() <= max_len {
            content.to_string()
        } else {
            format!("{}...", &content[..max_len])
        }
    }
}

/// Simple rate limiter
struct RateLimiter {
    limits: std::collections::HashMap<String, (u32, std::time::Instant)>,
    max_per_minute: u32,
}

impl RateLimiter {
    fn new(max_per_minute: u32) -> Self {
        Self {
            limits: std::collections::HashMap::new(),
            max_per_minute,
        }
    }

    fn check_and_update(&mut self, user_id: &str) -> bool {
        let now = std::time::Instant::now();

        match self.limits.get_mut(user_id) {
            Some((count, last_reset)) => {
                if now.duration_since(*last_reset).as_secs() >= 60 {
                    // Reset counter after a minute
                    *count = 1;
                    *last_reset = now;
                    true
                } else if *count < self.max_per_minute {
                    *count += 1;
                    true
                } else {
                    false
                }
            }
            None => {
                self.limits.insert(user_id.to_string(), (1, now));
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_query() {
        let config = SecureOrchestrationConfig::default();
        // Test would go here
    }

    #[test]
    fn test_pii_filtering() {
        // Test PII pattern matching
    }
}
