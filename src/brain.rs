//! AI brain module for content analysis and reasoning
//!
//! Provides intelligent analysis of content to extract meaning,
//! determine memory types, and generate reasoning chains.

use crate::types::{
    ContentAnalysis, Entity, EntityType, MemoryCell, MemoryResult, MemoryType, Priority,
    ProcessedRecall, RecalledMemory, Sentiment,
};
use tracing::debug;

/// AI brain for intelligent memory processing
pub struct AIBrain {
    /// Model name or configuration
    model_name: String,
    /// Analysis parameters
    params: AnalysisParams,
}

/// Parameters for AI analysis
#[derive(Debug, Clone)]
struct AnalysisParams {
    /// Enable sentiment analysis
    enable_sentiment: bool,
}

impl Default for AnalysisParams {
    fn default() -> Self {
        Self {
            enable_sentiment: true,
        }
    }
}

impl AIBrain {
    /// Create new AI brain instance
    pub fn new(model_name: String) -> Self {
        Self {
            model_name,
            params: AnalysisParams::default(),
        }
    }

    /// Analyze content to extract semantic meaning
    pub async fn analyze_content(
        &self,
        content: &str,
        context_hint: Option<&str>,
    ) -> MemoryResult<ContentAnalysis> {
        debug!("Analyzing content with {} model", self.model_name);

        // Extract summary (simplified for now - in production would use LLM)
        let summary = self.generate_summary(content);

        // Extract tags
        let tags = self.extract_tags(content);

        // Determine memory type
        let memory_type = self.determine_memory_type(content, &tags);

        // Calculate importance
        let importance = self.calculate_importance(content, &tags);

        // Suggest context
        let suggested_context = context_hint
            .map(|h| h.to_string())
            .unwrap_or_else(|| self.suggest_context(&tags));

        // Extract entities
        let entities = self.extract_entities(content);

        // Extract concepts
        let concepts = self.extract_concepts(content, &tags);

        // Analyze sentiment if enabled
        let sentiment = if self.params.enable_sentiment {
            Some(self.analyze_sentiment(content))
        } else {
            None
        };

        Ok(ContentAnalysis {
            summary,
            tags,
            memory_type,
            importance,
            suggested_context,
            entities,
            concepts,
            sentiment,
        })
    }

    /// Process recall results with reasoning
    pub async fn process_recall(
        &self,
        raw_recall: RecalledMemory,
    ) -> MemoryResult<ProcessedRecall> {
        debug!(
            "Processing recall with {} results",
            raw_recall.semantic_layer.len()
                + raw_recall.contextual_layer.len()
                + raw_recall.detailed_layer.len()
        );

        // Filter and rank semantic memories
        let semantic = self.filter_and_rank(raw_recall.semantic_layer, 0.3);

        // Filter contextual memories
        let contextual = self.filter_and_rank(raw_recall.contextual_layer, 0.2);

        // Filter detailed memories
        let detailed = self.filter_and_rank(raw_recall.detailed_layer, 0.1);

        // Generate reasoning steps
        let reasoning = vec![
            format!("Found {} highly relevant semantic memories", semantic.len()),
            format!("Identified {} contextual connections", contextual.len()),
            format!("Retrieved {} detailed memories for depth", detailed.len()),
            "Applied importance-based ranking and filtering".to_string(),
        ];

        // Calculate overall confidence
        let confidence = self.calculate_recall_confidence(&semantic, &contextual, &detailed);

        // Generate suggestions
        let suggestions = self.generate_suggestions(&semantic, &contextual);

        Ok(ProcessedRecall {
            semantic,
            contextual,
            detailed,
            reasoning,
            confidence,
            suggestions,
        })
    }

    /// Generate summary from content
    fn generate_summary(&self, content: &str) -> String {
        // Simple implementation - in production would use LLM
        let words: Vec<&str> = content.split_whitespace().collect();
        let summary_len = (words.len() / 10).max(10).min(50);
        words[..summary_len.min(words.len())].join(" ") + "..."
    }

    /// Extract tags from content
    fn extract_tags(&self, content: &str) -> Vec<String> {
        let mut tags = Vec::new();

        // Extract hashtag-like patterns
        for word in content.split_whitespace() {
            if word.starts_with('#') && word.len() > 1 {
                tags.push(word[1..].to_lowercase());
            }
        }

        // Add common keywords (simplified)
        let keywords = [
            "rust", "memory", "ai", "agent", "system", "database", "cache",
        ];
        for keyword in keywords {
            if content.to_lowercase().contains(keyword) {
                tags.push(keyword.to_string());
            }
        }

        tags.dedup();
        tags
    }

    /// Determine memory type from content
    fn determine_memory_type(&self, content: &str, tags: &[String]) -> MemoryType {
        let lower = content.to_lowercase();

        // Check for procedural indicators
        if lower.contains("how to") || lower.contains("steps") || lower.contains("procedure") {
            return MemoryType::Procedural {
                steps: vec!["Step 1".to_string()],
                tools: vec![],
                prerequisites: vec![],
            };
        }

        // Check for episodic indicators
        if lower.contains("happened") || lower.contains("event") || lower.contains("when") {
            return MemoryType::Episodic {
                event: "Event description".to_string(),
                location: None,
                participants: vec![],
                timeframe: None,
            };
        }

        // Check for working memory indicators
        if lower.contains("task") || lower.contains("todo") || lower.contains("deadline") {
            return MemoryType::Working {
                task: "Task description".to_string(),
                deadline: None,
                priority: Priority::Medium,
            };
        }

        // Default to semantic
        MemoryType::Semantic {
            facts: vec![],
            concepts: tags.to_vec(),
        }
    }

    /// Calculate importance score
    fn calculate_importance(&self, content: &str, tags: &[String]) -> f32 {
        let mut score = 0.6; // Base score (increased for better filtering)

        // Increase for length (more content = potentially more important)
        let word_count = content.split_whitespace().count();
        if word_count > 100 {
            score += 0.1;
        }
        if word_count > 500 {
            score += 0.1;
        }

        // Increase for tags
        score += (tags.len() as f32 * 0.05).min(0.2);

        // Check for importance indicators
        let important_words = ["critical", "important", "urgent", "essential", "key"];
        for word in important_words {
            if content.to_lowercase().contains(word) {
                score += 0.1;
                break;
            }
        }

        score.min(1.0)
    }

    /// Suggest context based on tags
    fn suggest_context(&self, tags: &[String]) -> String {
        if tags.is_empty() {
            return "general".to_string();
        }

        // Use first tag as base context
        let base = &tags[0];

        // Add sub-context if multiple tags
        if tags.len() > 1 {
            format!("{}/{}", base, tags[1])
        } else {
            base.to_string()
        }
    }

    /// Extract entities from content
    fn extract_entities(&self, content: &str) -> Vec<Entity> {
        let mut entities = Vec::new();

        // Simple pattern matching (in production would use NER model)
        for word in content.split_whitespace() {
            // Check for capitalized words (potential names/places)
            if word.chars().next().map_or(false, |c| c.is_uppercase()) {
                let entity_type = if word.ends_with("Inc.") || word.ends_with("Corp.") {
                    EntityType::Organization
                } else if word.parse::<f64>().is_ok() {
                    EntityType::Number
                } else {
                    EntityType::Other(word.to_string())
                };

                entities.push(Entity {
                    text: word.to_string(),
                    entity_type,
                    confidence: 0.7,
                });
            }
        }

        entities
    }

    /// Extract concepts from content
    fn extract_concepts(&self, content: &str, tags: &[String]) -> Vec<String> {
        let mut concepts = tags.to_vec();

        // Add technical concepts if mentioned
        let tech_concepts = [
            "machine learning",
            "neural network",
            "database",
            "api",
            "algorithm",
        ];
        for concept in tech_concepts {
            if content.to_lowercase().contains(concept) {
                concepts.push(concept.to_string());
            }
        }

        concepts.dedup();
        concepts
    }

    /// Analyze sentiment
    fn analyze_sentiment(&self, content: &str) -> Sentiment {
        let lower = content.to_lowercase();
        let mut score: f32 = 0.0;

        // Positive indicators
        let positive = [
            "good",
            "great",
            "excellent",
            "happy",
            "success",
            "wonderful",
        ];
        for word in positive {
            if lower.contains(word) {
                score += 0.2;
            }
        }

        // Negative indicators
        let negative = ["bad", "terrible", "fail", "error", "problem", "issue"];
        for word in negative {
            if lower.contains(word) {
                score -= 0.2;
            }
        }

        // Clamp score
        score = score.max(-1.0).min(1.0);

        // Detect emotions
        let mut emotions = Vec::new();
        if score > 0.3 {
            emotions.push("positive".to_string());
        } else if score < -0.3 {
            emotions.push("negative".to_string());
        } else {
            emotions.push("neutral".to_string());
        }

        Sentiment {
            score,
            confidence: 0.6,
            emotions,
        }
    }

    /// Filter and rank memories by importance
    fn filter_and_rank(&self, memories: Vec<MemoryCell>, threshold: f32) -> Vec<MemoryCell> {
        debug!(
            "Filtering {} memories with importance threshold {}",
            memories.len(),
            threshold
        );

        let mut filtered: Vec<MemoryCell> = memories
            .into_iter()
            .filter(|m| {
                debug!(
                    "Memory {} importance: {} (threshold: {})",
                    m.id, m.importance, threshold
                );
                m.importance >= threshold
            })
            .collect();

        debug!("After filtering: {} memories remain", filtered.len());

        // Sort by dynamic importance
        filtered.sort_by(|a, b| {
            b.dynamic_importance()
                .partial_cmp(&a.dynamic_importance())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to top results
        filtered.truncate(10);
        filtered
    }

    /// Calculate confidence in recall results
    fn calculate_recall_confidence(
        &self,
        semantic: &[MemoryCell],
        contextual: &[MemoryCell],
        detailed: &[MemoryCell],
    ) -> f32 {
        let total_count = semantic.len() + contextual.len() + detailed.len();

        if total_count == 0 {
            return 0.0;
        }

        // Calculate average importance
        let total_importance: f32 = semantic.iter().map(|m| m.importance).sum::<f32>()
            + contextual.iter().map(|m| m.importance).sum::<f32>()
            + detailed.iter().map(|m| m.importance).sum::<f32>();

        let avg_importance = total_importance / total_count as f32;

        // Factor in distribution across layers
        let distribution_score = if !semantic.is_empty() && !contextual.is_empty() {
            0.8
        } else if !semantic.is_empty() {
            0.6
        } else {
            0.4
        };

        (avg_importance * 0.7 + distribution_score * 0.3).min(1.0)
    }

    /// Generate follow-up suggestions
    fn generate_suggestions(
        &self,
        semantic: &[MemoryCell],
        contextual: &[MemoryCell],
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Suggest exploring related contexts
        let contexts: Vec<String> = contextual.iter().map(|m| m.context_path.clone()).collect();

        for context in contexts.iter().take(3) {
            suggestions.push(format!("Explore more memories in context: {}", context));
        }

        // Suggest related tags
        let mut all_tags = Vec::new();
        for memory in semantic.iter().chain(contextual.iter()) {
            all_tags.extend(memory.tags.clone());
        }
        all_tags.dedup();

        if !all_tags.is_empty() {
            suggestions.push(format!(
                "Search for related tags: {}",
                all_tags[..3.min(all_tags.len())].join(", ")
            ));
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_content_analysis() {
        let brain = AIBrain::new("test_model".to_string());
        let content = "This is a #test content about Rust programming with important information.";

        let analysis = brain.analyze_content(content, None).await.unwrap();

        assert!(!analysis.summary.is_empty());
        assert!(analysis.tags.contains(&"test".to_string()));
        assert!(analysis.tags.contains(&"rust".to_string()));
        assert!(analysis.importance > 0.5);
    }

    #[tokio::test]
    async fn test_memory_type_detection() {
        let brain = AIBrain::new("test_model".to_string());

        // Test procedural
        let content = "How to set up a database: first install, then configure.";
        let analysis = brain.analyze_content(content, None).await.unwrap();
        assert!(matches!(
            analysis.memory_type,
            MemoryType::Procedural { .. }
        ));

        // Test episodic
        let content = "The event happened yesterday when we deployed the system.";
        let analysis = brain.analyze_content(content, None).await.unwrap();
        assert!(matches!(analysis.memory_type, MemoryType::Episodic { .. }));

        // Test working
        let content = "Task: Complete the TODO list before the deadline.";
        let analysis = brain.analyze_content(content, None).await.unwrap();
        assert!(matches!(analysis.memory_type, MemoryType::Working { .. }));
    }
}
