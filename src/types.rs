//! Core data types for the AI memory system
//! 
//! This module defines the fundamental data structures that represent
//! memory cells, contexts, queries and responses in a human-brain-like
//! memory architecture for AI agents.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A single memory cell - the fundamental unit of storage
/// 
/// Represents a piece of information stored in the AI's memory,
/// similar to how human brain stores memories with context,
/// importance and access patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCell {
    /// Unique identifier for this memory
    pub id: Uuid,
    /// Original content of the memory
    pub content: String,
    /// AI-generated summary for quick access
    pub summary: String,
    /// Tags for semantic categorization
    pub tags: Vec<String>,
    /// Vector embedding for similarity search
    pub embedding: Vec<f32>,
    /// Type of memory (semantic, episodic, procedural, working)
    pub memory_type: MemoryType,
    /// Importance score from 0.0 to 1.0
    pub importance: f32,
    /// How often this memory has been accessed
    pub access_frequency: u32,
    /// When this memory was created
    pub created_at: DateTime<Utc>,
    /// Last time this memory was accessed
    pub last_accessed: DateTime<Utc>,
    /// Hierarchical context path like "work/projects/rust_memory"
    pub context_path: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Types of memory similar to human cognitive science
/// 
/// Based on human memory research, different types of memories
/// are processed and retrieved differently.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryType {
    /// Semantic memory - facts, concepts, general knowledge
    Semantic {
        /// Key facts extracted from content
        facts: Vec<String>,
        /// Related concepts and categories
        concepts: Vec<String>,
    },
    /// Episodic memory - personal experiences, events
    Episodic {
        /// Description of the event
        event: String,
        /// Where it happened (optional)
        location: Option<String>,
        /// Who was involved
        participants: Vec<String>,
        /// Temporal context
        timeframe: Option<String>,
    },
    /// Procedural memory - how to do things, skills
    Procedural {
        /// Steps of the procedure
        steps: Vec<String>,
        /// Required tools or resources
        tools: Vec<String>,
        /// Prerequisites or conditions
        prerequisites: Vec<String>,
    },
    /// Working memory - temporary, task-focused information
    Working {
        /// Current task being worked on
        task: String,
        /// Deadline if applicable
        deadline: Option<DateTime<Utc>>,
        /// Priority level
        priority: Priority,
    },
}

/// Priority levels for working memory
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Priority {
    Low,
    Medium,
    High,
    Urgent,
}

/// Context represents hierarchical organization of memories
/// 
/// Like folders in a file system, contexts help organize
/// memories by topic, project, or domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    /// Hierarchical path like "work/projects/rust_memory"
    pub path: String,
    /// Human-readable name
    pub name: String,
    /// Description of this context
    pub description: String,
    /// Parent context path (optional)
    pub parent: Option<String>,
    /// Child context paths
    pub children: Vec<String>,
    /// Context embedding for semantic similarity
    pub embedding: Vec<f32>,
    /// How active/frequently used this context is
    pub activity_level: f32,
    /// Number of memories in this context
    pub memory_count: usize,
    /// When this context was created
    pub created_at: DateTime<Utc>,
    /// Last activity in this context
    pub last_activity: DateTime<Utc>,
}

/// Query structure for memory retrieval
/// 
/// Represents a request to recall memories, with various
/// filters and hints to guide the search process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQuery {
    /// The query text to search for
    pub text: String,
    /// Optional context hint to focus search
    pub context_hint: Option<String>,
    /// Filter by specific memory types
    pub memory_types: Option<Vec<MemoryType>>,
    /// Maximum number of results to return
    pub limit: Option<usize>,
    /// Minimum importance threshold
    pub min_importance: Option<f32>,
    /// Time range filter
    pub time_range: Option<TimeRange>,
    /// Include related memories
    pub include_related: bool,
    /// Similarity threshold for vector search
    pub similarity_threshold: Option<f32>,
}

/// Time range for filtering memories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
}

/// Result of memory recall - structured like human memory retrieval
/// 
/// Human memory recall happens in layers: first quick associations,
/// then contextual connections, finally detailed information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecalledMemory {
    /// Unique ID for this recall session
    pub query_id: Uuid,
    /// Fast semantic associations (tags, facts)
    pub semantic_layer: Vec<MemoryCell>,
    /// Related contextual memories
    pub contextual_layer: Vec<MemoryCell>,
    /// Detailed information layer
    pub detailed_layer: Vec<MemoryCell>,
    /// AI reasoning chain for how memories were connected
    pub reasoning_chain: Vec<String>,
    /// Confidence in the recall results (0.0-1.0)
    pub confidence: f32,
    /// How long the recall took in milliseconds
    pub recall_time_ms: u64,
    /// Additional context discovered during recall
    pub discovered_contexts: Vec<String>,
}

/// Analysis result from AI brain processing content
/// 
/// When storing new memories, the AI brain analyzes content
/// to extract semantic meaning, importance, and categorization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentAnalysis {
    /// AI-generated summary
    pub summary: String,
    /// Extracted tags for categorization
    pub tags: Vec<String>,
    /// Determined memory type
    pub memory_type: MemoryType,
    /// Calculated importance (0.0-1.0)
    pub importance: f32,
    /// Suggested context path
    pub suggested_context: String,
    /// Extracted entities (people, places, things)
    pub entities: Vec<Entity>,
    /// Key concepts identified
    pub concepts: Vec<String>,
    /// Emotional tone if applicable
    pub sentiment: Option<Sentiment>,
}

/// Named entity extracted from content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// The entity text
    pub text: String,
    /// Type of entity (person, place, organization, etc.)
    pub entity_type: EntityType,
    /// Confidence in the extraction
    pub confidence: f32,
}

/// Types of named entities
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EntityType {
    Person,
    Place,
    Organization,
    Technology,
    Concept,
    Date,
    Number,
    Other(String),
}

/// Sentiment analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sentiment {
    /// Overall sentiment (-1.0 to 1.0, negative to positive)
    pub score: f32,
    /// Confidence in sentiment analysis
    pub confidence: f32,
    /// Detected emotions
    pub emotions: Vec<String>,
}

/// Processed recall result from AI brain
/// 
/// After retrieving raw memories, the AI brain processes and
/// structures them for optimal presentation and reasoning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedRecall {
    /// Filtered and ranked semantic memories
    pub semantic: Vec<MemoryCell>,
    /// Most relevant contextual memories
    pub contextual: Vec<MemoryCell>,
    /// Important detailed memories
    pub detailed: Vec<MemoryCell>,
    /// Reasoning steps taken
    pub reasoning: Vec<String>,
    /// Overall confidence in results
    pub confidence: f32,
    /// Suggested follow-up queries
    pub suggestions: Vec<String>,
}

/// Statistics about memory usage and performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total number of memories stored
    pub total_memories: usize,
    /// Number of contexts
    pub total_contexts: usize,
    /// Memory usage by type
    pub memory_by_type: HashMap<String, usize>,
    /// Most active contexts
    pub top_contexts: Vec<(String, usize)>,
    /// Recent activity metrics
    pub recent_queries: usize,
    /// Average recall time
    pub avg_recall_time_ms: f64,
    /// Cache hit rate
    pub cache_hit_rate: f32,
    /// Database size metrics
    pub storage_size_mb: f64,
}

/// Storage statistics from database
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StorageStats {
    /// Total number of memories in storage
    pub total_memories: usize,
    /// Total number of contexts in storage  
    pub total_contexts: usize,
}

/// Error types for memory operations
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Storage error: {0}")]
    Storage(String),
    
    #[error("Embedding error: {0}")]
    Embedding(String),
    
    #[error("AI brain error: {0}")]
    Brain(String),
    
    #[error("Cache error: {0}")]
    Cache(String),
    
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
    
    #[error("Memory not found: {0}")]
    NotFound(Uuid),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Network error: {0}")]
    Network(String),
}

pub type MemoryResult<T> = Result<T, MemoryError>;

// Конверсии для различных типов ошибок
impl From<String> for MemoryError {
    fn from(err: String) -> Self {
        MemoryError::Config(err)
    }
}

impl From<&str> for MemoryError {
    fn from(err: &str) -> Self {
        MemoryError::Config(err.to_string())
    }
}

impl From<std::io::Error> for MemoryError {
    fn from(err: std::io::Error) -> Self {
        MemoryError::Storage(err.to_string())
    }
}

impl From<serde_json::Error> for MemoryError {
    fn from(err: serde_json::Error) -> Self {
        MemoryError::Config(err.to_string())
    }
}

// Added PyO3 error conversion for Python integration
impl From<pyo3::PyErr> for MemoryError {
    fn from(err: pyo3::PyErr) -> Self {
        MemoryError::Embedding(format!("Python error: {}", err))
    }
}

impl From<prometheus::Error> for MemoryError {
    fn from(err: prometheus::Error) -> Self {
        MemoryError::Config(err.to_string())
    }
}

impl Default for MemoryQuery {
    fn default() -> Self {
        Self {
            text: String::new(),
            context_hint: None,
            memory_types: None,
            limit: Some(10),
            min_importance: Some(0.1),
            time_range: None,
            include_related: true,
            similarity_threshold: Some(0.7),
        }
    }
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Medium
    }
}

impl MemoryCell {
    /// Create a new memory cell with default values
    pub fn new(content: String, context_path: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            content,
            summary: String::new(),
            tags: Vec::new(),
            embedding: Vec::new(),
            memory_type: MemoryType::Semantic {
                facts: Vec::new(),
                concepts: Vec::new(),
            },
            importance: 0.5,
            access_frequency: 0,
            created_at: now,
            last_accessed: now,
            context_path,
            metadata: HashMap::new(),
        }
    }

    /// Update access statistics
    pub fn update_access(&mut self) {
        self.last_accessed = Utc::now();
        self.access_frequency += 1;
    }

    /// Calculate memory age in days
    pub fn age_days(&self) -> f64 {
        let now = Utc::now();
        (now - self.created_at).num_seconds() as f64 / 86400.0
    }

    /// Calculate dynamic importance based on age and access frequency
    pub fn dynamic_importance(&self) -> f32 {
        let age_factor = 1.0 / (1.0 + self.age_days() as f32 / 30.0);
        let access_factor = (self.access_frequency as f32).log2().max(0.0) / 10.0;
        (self.importance + access_factor) * age_factor
    }
}

impl MemoryContext {
    /// Create a new context
    pub fn new(path: String, name: String, description: String) -> Self {
        let now = Utc::now();
        Self {
            path,
            name,
            description,
            parent: None,
            children: Vec::new(),
            embedding: Vec::new(),
            activity_level: 0.0,
            memory_count: 0,
            created_at: now,
            last_activity: now,
        }
    }

    /// Update context activity
    pub fn update_activity(&mut self, delta: f32) {
        self.activity_level = (self.activity_level + delta).max(0.0).min(1.0);
        self.last_activity = Utc::now();
    }
}

impl MemoryType {
    /// Serialize memory type using JSON for reliable storage
    pub fn to_storage_string(&self) -> MemoryResult<String> {
        serde_json::to_string(self)
            .map_err(|e| MemoryError::Storage(format!("Memory type serialization failed: {}", e)))
    }

    /// Parse memory type from storage format with proper error handling
    pub fn from_storage_string(s: &str) -> MemoryResult<Self> {
        if s.trim().is_empty() {
            return Err(MemoryError::Storage("Empty memory type string".to_string()));
        }

        // Parse as JSON (only supported format)
        serde_json::from_str::<MemoryType>(s)
            .map_err(|e| MemoryError::Storage(format!(
                "Failed to parse memory type: {}. Input: '{}'",
                e, s.chars().take(100).collect::<String>()
            )))
    }


    /// Get simple type name for indexing and statistics
    pub fn type_name(&self) -> &'static str {
        match self {
            MemoryType::Semantic { .. } => "Semantic",
            MemoryType::Episodic { .. } => "Episodic", 
            MemoryType::Procedural { .. } => "Procedural",
            MemoryType::Working { .. } => "Working",
        }
    }

    /// Validate memory type data integrity
    pub fn validate(&self) -> MemoryResult<()> {
        match self {
            MemoryType::Semantic { facts, concepts } => {
                if facts.len() > 1000 {
                    return Err(MemoryError::Storage("Too many facts in semantic memory".to_string()));
                }
                if concepts.len() > 1000 {
                    return Err(MemoryError::Storage("Too many concepts in semantic memory".to_string()));
                }
            }
            MemoryType::Episodic { event, participants, .. } => {
                if event.is_empty() {
                    return Err(MemoryError::Storage("Episodic memory must have an event description".to_string()));
                }
                if participants.len() > 100 {
                    return Err(MemoryError::Storage("Too many participants in episodic memory".to_string()));
                }
            }
            MemoryType::Procedural { steps, tools, prerequisites } => {
                if steps.is_empty() {
                    return Err(MemoryError::Storage("Procedural memory must have at least one step".to_string()));
                }
                if steps.len() > 1000 {
                    return Err(MemoryError::Storage("Too many steps in procedural memory".to_string()));
                }
                if tools.len() > 100 || prerequisites.len() > 100 {
                    return Err(MemoryError::Storage("Too many tools or prerequisites in procedural memory".to_string()));
                }
            }
            MemoryType::Working { task, .. } => {
                if task.is_empty() {
                    return Err(MemoryError::Storage("Working memory must have a task description".to_string()));
                }
            }
        }
        Ok(())
    }
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            total_memories: 0,
            total_contexts: 0,
            memory_by_type: HashMap::new(),
            top_contexts: Vec::new(),
            recent_queries: 0,
            avg_recall_time_ms: 0.0,
            cache_hit_rate: 0.0,
            storage_size_mb: 0.0,
        }
    }
}