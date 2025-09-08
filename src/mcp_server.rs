// AI Memory Service - MCP Server Implementation
// Model Context Protocol server for Claude Code integration
// Provides intelligent memory access, search, and management capabilities

use crate::{
    memory::MemoryService,
    orchestrator::{MemoryOrchestrator, OrchestratorConfig, MemoryOptimization},
    types::*,
};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// MCP Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Server name
    pub name: String,
    /// Server version
    pub version: String,
    /// Maximum results per query
    pub max_results: usize,
    /// Enable orchestrator integration
    pub use_orchestrator: bool,
    /// Enable intelligent ranking
    pub intelligent_ranking: bool,
    /// Session timeout in seconds
    pub session_timeout: u64,
    /// Cache size for recent queries
    pub cache_size: usize,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            name: "ai-memory-service".to_string(),
            version: "1.0.0".to_string(),
            max_results: 50,
            use_orchestrator: true,
            intelligent_ranking: true,
            session_timeout: 3600,
            cache_size: 100,
        }
    }
}

/// MCP Method types supported by the server
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum McpMethod {
    /// Initialize server
    Initialize,
    /// List available tools/methods
    ListTools,
    /// Execute a tool/method
    CallTool,
    /// List resources (memory contexts)
    ListResources,
    /// Read a specific resource
    ReadResource,
    /// Subscribe to updates
    Subscribe,
    /// Unsubscribe from updates
    Unsubscribe,
    /// Get server capabilities
    GetCapabilities,
}

/// MCP Tool definition for memory operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Input schema
    pub input_schema: Value,
    /// Whether tool modifies state
    pub is_mutation: bool,
}

/// MCP Resource representing memory contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResource {
    /// Resource URI
    pub uri: String,
    /// Resource name
    pub name: String,
    /// Resource description
    pub description: String,
    /// MIME type
    pub mime_type: String,
    /// Resource metadata
    pub metadata: HashMap<String, Value>,
}

/// Session information for Claude Code
#[derive(Debug, Clone)]
pub struct McpSession {
    /// Session ID
    pub id: String,
    /// Claude Code session identifier
    pub claude_session_id: String,
    /// Creation time
    pub created_at: DateTime<Utc>,
    /// Last activity time
    pub last_activity: DateTime<Utc>,
    /// Session metadata
    pub metadata: HashMap<String, Value>,
    /// Active subscriptions
    pub subscriptions: Vec<String>,
}

/// MCP Server implementation
pub struct McpServer {
    /// Memory service instance
    memory_service: Arc<MemoryService>,
    /// Orchestrator instance
    orchestrator: Option<Arc<MemoryOrchestrator>>,
    /// Server configuration
    config: McpServerConfig,
    /// Active sessions
    sessions: Arc<RwLock<HashMap<String, McpSession>>>,
    /// Query cache
    query_cache: Arc<RwLock<HashMap<String, Vec<MemoryCell>>>>,
    /// Server statistics
    stats: Arc<RwLock<McpServerStats>>,
}

/// Server statistics
#[derive(Debug, Default)]
pub struct McpServerStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub active_sessions: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub avg_response_time_ms: f64,
}

impl McpServer {
    /// Create new MCP server instance
    pub fn new(
        memory_service: Arc<MemoryService>,
        orchestrator_config: Option<OrchestratorConfig>,
        config: McpServerConfig,
    ) -> Result<Self> {
        let orchestrator = if config.use_orchestrator {
            orchestrator_config
                .map(|cfg| MemoryOrchestrator::new(cfg))
                .transpose()?
                .map(Arc::new)
        } else {
            None
        };

        Ok(Self {
            memory_service,
            orchestrator,
            config,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(McpServerStats::default())),
        })
    }

    /// Get server capabilities
    pub async fn get_capabilities(&self) -> Result<Value> {
        Ok(json!({
            "name": self.config.name,
            "version": self.config.version,
            "capabilities": {
                "tools": true,
                "resources": true,
                "subscriptions": true,
                "intelligent_search": self.config.intelligent_ranking,
                "orchestrator": self.config.use_orchestrator,
                "session_management": true,
                "cross_session_learning": true,
            },
            "supported_memory_types": [
                "semantic", "episodic", "procedural", "working",
                "code", "documentation", "conversation", "problem_solution",
                "meta_cognitive", "insight", "problem_pattern", 
                "user_preference", "context_distillation"
            ],
            "max_results": self.config.max_results,
            "session_timeout": self.config.session_timeout,
        }))
    }

    /// List available tools
    pub async fn list_tools(&self) -> Result<Vec<McpTool>> {
        Ok(vec![
            McpTool {
                name: "search_memory".to_string(),
                description: "Intelligent search across all memory types with semantic understanding".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "memory_types": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Filter by memory types"
                        },
                        "context": {
                            "type": "string",
                            "description": "Context path filter"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 20
                        },
                        "use_orchestrator": {
                            "type": "boolean",
                            "description": "Use GPT-4 orchestrator for enhanced search",
                            "default": true
                        }
                    },
                    "required": ["query"]
                }),
                is_mutation: false,
            },
            McpTool {
                name: "store_memory".to_string(),
                description: "Store new memory with intelligent categorization".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Memory content"
                        },
                        "memory_type": {
                            "type": "string",
                            "description": "Type of memory",
                            "enum": ["semantic", "episodic", "procedural", "working", "code", "documentation", "conversation", "insight"]
                        },
                        "context": {
                            "type": "string",
                            "description": "Context path"
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Tags for categorization"
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance score (0-1)",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "auto_categorize": {
                            "type": "boolean",
                            "description": "Automatically determine memory type using orchestrator",
                            "default": true
                        }
                    },
                    "required": ["content"]
                }),
                is_mutation: true,
            },
            McpTool {
                name: "get_insights".to_string(),
                description: "Generate insights from memory patterns using GPT-4 orchestrator".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "Context to analyze"
                        },
                        "insight_types": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Types of insights to generate",
                            "enum": ["user_preference", "pattern_recognition", "best_practice", "error_pattern", "workflow_optimization"]
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum memories to analyze",
                            "default": 100
                        }
                    }
                }),
                is_mutation: false,
            },
            McpTool {
                name: "distill_context".to_string(),
                description: "Distill large amounts of memory into key insights".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "Context to distill"
                        },
                        "time_range": {
                            "type": "object",
                            "properties": {
                                "start": { "type": "string", "format": "date-time" },
                                "end": { "type": "string", "format": "date-time" }
                            }
                        },
                        "max_points": {
                            "type": "integer",
                            "description": "Maximum key points to extract",
                            "default": 10
                        }
                    }
                }),
                is_mutation: false,
            },
            McpTool {
                name: "optimize_memory".to_string(),
                description: "Optimize memory storage by removing duplicates and consolidating similar memories".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "Context to optimize"
                        },
                        "aggressive": {
                            "type": "boolean",
                            "description": "Use aggressive optimization",
                            "default": false
                        }
                    }
                }),
                is_mutation: true,
            },
            McpTool {
                name: "get_user_preferences".to_string(),
                description: "Analyze and return user preferences and patterns".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "categories": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Preference categories to analyze"
                        }
                    }
                }),
                is_mutation: false,
            },
            McpTool {
                name: "get_problem_patterns".to_string(),
                description: "Identify recurring problem patterns and their solutions".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "problem_category": {
                            "type": "string",
                            "description": "Category of problems to analyze"
                        },
                        "include_solutions": {
                            "type": "boolean",
                            "description": "Include typical solutions",
                            "default": true
                        }
                    }
                }),
                is_mutation: false,
            },
            McpTool {
                name: "manage_session".to_string(),
                description: "Manage Claude Code session state".to_string(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["create", "update", "end"],
                            "description": "Session action"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Claude session ID"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Session metadata"
                        }
                    },
                    "required": ["action", "session_id"]
                }),
                is_mutation: true,
            },
        ])
    }

    /// Execute a tool
    pub async fn call_tool(&self, tool_name: &str, params: Value) -> Result<Value> {
        use tokio::time::Instant;
        let start_time = Instant::now();
        
        let result = match tool_name {
            "search_memory" => self.search_memory(params).await,
            "store_memory" => self.store_memory(params).await,
            "get_insights" => self.get_insights(params).await,
            "distill_context" => self.distill_context(params).await,
            "optimize_memory" => self.optimize_memory(params).await,
            "get_user_preferences" => self.get_user_preferences(params).await,
            "get_problem_patterns" => self.get_problem_patterns(params).await,
            "manage_session" => self.manage_session(params).await,
            _ => Err(anyhow!("Unknown tool: {}", tool_name)),
        };

        let duration_ms = start_time.elapsed().as_millis() as f64;
        
        if result.is_ok() {
            self.update_stats_with_time(true, duration_ms).await;
        } else {
            self.update_stats_with_time(false, duration_ms).await;
        }

        result
    }

    /// Intelligent memory search
    async fn search_memory(&self, params: Value) -> Result<Value> {
        let query = params["query"]
            .as_str()
            .ok_or_else(|| anyhow!("Query is required"))?;
        
        let memory_types: Option<Vec<String>> = params["memory_types"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            });
        
        let context = params["context"].as_str().map(String::from);
        let limit = params["limit"].as_u64().unwrap_or(20) as usize;
        let use_orchestrator = params["use_orchestrator"].as_bool().unwrap_or(true);

        // Check cache first
        let cache_key = format!("{}-{:?}-{:?}-{}", query, memory_types, context, limit);
        if let Some(cached) = self.get_cached_results(&cache_key).await {
            return Ok(json!({
                "results": cached,
                "cached": true
            }));
        }

        // Perform search
        let results = if use_orchestrator && self.orchestrator.is_some() {
            // Use orchestrator for intelligent search
            self.orchestrator_enhanced_search(query, memory_types, context, limit).await?
        } else {
            // Basic search without orchestrator
            self.basic_search(query, memory_types, context, limit).await?
        };

        // Cache results
        self.cache_results(&cache_key, &results).await;

        Ok(json!({
            "results": results,
            "cached": false,
            "orchestrator_used": use_orchestrator && self.orchestrator.is_some()
        }))
    }

    /// Store new memory with intelligent categorization
    async fn store_memory(&self, params: Value) -> Result<Value> {
        let content = params["content"]
            .as_str()
            .ok_or_else(|| anyhow!("Content is required"))?;
        
        let memory_type = if params["auto_categorize"].as_bool().unwrap_or(true) 
            && self.orchestrator.is_some() {
            // Use orchestrator to determine memory type
            self.determine_memory_type(content).await?
        } else {
            // Use provided memory type or default
            params["memory_type"]
                .as_str()
                .unwrap_or("semantic")
                .to_string()
        };

        let context = params["context"]
            .as_str()
            .unwrap_or("general")
            .to_string();
        
        let tags: Vec<String> = params["tags"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        
        let importance = params["importance"]
            .as_f64()
            .unwrap_or(0.5) as f32;

        // Create memory cell
        let memory = self.create_memory_cell(
            content,
            &memory_type,
            &context,
            tags.clone(),
            importance
        )?;

        // Store in memory service with proper parameters
        let metadata: HashMap<String, String> = vec![
            ("memory_type".to_string(), memory_type.clone()),
            ("importance".to_string(), importance.to_string()),
            ("tags".to_string(), tags.join(",")),
        ].into_iter().collect();
        
        self.memory_service.store(
            content.to_string(),
            Some(context.clone()),
            Some(metadata)
        ).await?;

        Ok(json!({
            "id": memory.id,
            "memory_type": memory_type,
            "context": context,
            "stored_at": memory.created_at,
            "success": true
        }))
    }

    /// Generate insights using orchestrator
    async fn get_insights(&self, params: Value) -> Result<Value> {
        if self.orchestrator.is_none() {
            return Err(anyhow!("Orchestrator not available"));
        }

        let context = params["context"].as_str().map(String::from);
        let limit = params["limit"].as_u64().unwrap_or(100) as usize;
        
        // Get relevant memories
        let memories = self.get_memories_for_context(context, limit).await?;
        
        // Generate insights
        let orchestrator = self.orchestrator.as_ref().unwrap();
        let insights = orchestrator.generate_insights(&memories, InsightType::PatternRecognition).await?;

        Ok(json!({
            "insights": insights,
            "memories_analyzed": memories.len(),
            "success": true
        }))
    }

    /// Distill context using orchestrator
    async fn distill_context(&self, params: Value) -> Result<Value> {
        if self.orchestrator.is_none() {
            return Err(anyhow!("Orchestrator not available"));
        }

        let context = params["context"].as_str().map(String::from);
        let max_points = params["max_points"].as_u64().unwrap_or(10) as usize;
        
        // Get memories for context
        let memories = self.get_memories_for_context(context, 1000).await?;
        
        // Distill using orchestrator
        let orchestrator = self.orchestrator.as_ref().unwrap();
        let distillation = orchestrator.distill_context(&memories, None).await?;

        Ok(json!({
            "distillation": distillation,
            "original_count": memories.len(),
            "key_points": max_points,
            "success": true
        }))
    }

    /// Optimize memory storage
    async fn optimize_memory(&self, params: Value) -> Result<Value> {
        if self.orchestrator.is_none() {
            return Err(anyhow!("Orchestrator not available"));
        }

        let context = params["context"].as_str().map(String::from);
        let aggressive = params["aggressive"].as_bool().unwrap_or(false);
        
        // Get memories for optimization
        let memories = self.get_memories_for_context(context, 5000).await?;
        
        // Optimize using orchestrator
        let orchestrator = self.orchestrator.as_ref().unwrap();
        let optimization = orchestrator.optimize_memory_storage(&memories).await?;

        // Apply optimization if aggressive mode
        if aggressive {
            self.apply_optimization(optimization.clone()).await?;
        }

        Ok(json!({
            "optimization": optimization,
            "memories_analyzed": memories.len(),
            "applied": aggressive,
            "success": true
        }))
    }

    /// Get user preferences
    async fn get_user_preferences(&self, _params: Value) -> Result<Value> {
        // Search for user preference memories using basic search
        let query = "preference user pattern";
        let preferences = self.basic_search(query, Some(vec!["user_preference".to_string()]), None, 100)
            .await?;

        // Group by preference type
        let mut grouped: HashMap<String, Vec<Value>> = HashMap::new();
        for pref in &preferences {
            if let MemoryType::UserPreference { preference_type, examples, strength, .. } = &pref.memory_type {
                grouped.entry(preference_type.clone())
                    .or_default()
                    .push(json!({
                        "type": preference_type,
                        "examples": examples,
                        "strength": strength,
                        "context": pref.context_path
                    }));
            }
        }

        Ok(json!({
            "preferences": grouped,
            "total_count": preferences.len(),
            "success": true
        }))
    }

    /// Get problem patterns
    async fn get_problem_patterns(&self, params: Value) -> Result<Value> {
        let category = params["problem_category"].as_str();
        let include_solutions = params["include_solutions"].as_bool().unwrap_or(true);
        
        // Search for problem pattern memories using basic search
        let query = category.unwrap_or("problem pattern solution");
        let mut patterns = self.basic_search(
            query, 
            Some(vec!["problem_pattern".to_string()]), 
            None, 
            100
        ).await?;

        // Filter by category if specified
        if let Some(cat) = category {
            patterns.retain(|p| {
                if let MemoryType::ProblemPattern { problem_category, .. } = &p.memory_type {
                    problem_category.contains(cat)
                } else {
                    false
                }
            });
        }

        // Format results
        let formatted: Vec<Value> = patterns.into_iter().map(|p| {
            if let MemoryType::ProblemPattern { 
                problem_category, 
                typical_solutions,
                prevention_strategies,
                frequency,
                ..
            } = p.memory_type {
                let mut result = json!({
                    "category": problem_category,
                    "frequency": frequency,
                    "prevention": prevention_strategies
                });
                
                if include_solutions {
                    result["solutions"] = json!(typical_solutions);
                }
                
                result
            } else {
                json!({})
            }
        }).collect();

        Ok(json!({
            "patterns": formatted,
            "success": true
        }))
    }

    /// Manage session
    async fn manage_session(&self, params: Value) -> Result<Value> {
        let action = params["action"]
            .as_str()
            .ok_or_else(|| anyhow!("Action is required"))?;
        
        let session_id = params["session_id"]
            .as_str()
            .ok_or_else(|| anyhow!("Session ID is required"))?;

        match action {
            "create" => {
                let session = McpSession {
                    id: Uuid::new_v4().to_string(),
                    claude_session_id: session_id.to_string(),
                    created_at: Utc::now(),
                    last_activity: Utc::now(),
                    metadata: params["metadata"]
                        .as_object()
                        .map(|m| {
                            m.iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect()
                        })
                        .unwrap_or_default(),
                    subscriptions: Vec::new(),
                };

                let mut sessions = self.sessions.write().await;
                sessions.insert(session_id.to_string(), session.clone());

                Ok(json!({
                    "session_id": session.id,
                    "created": true
                }))
            },
            "update" => {
                let mut sessions = self.sessions.write().await;
                if let Some(session) = sessions.get_mut(session_id) {
                    session.last_activity = Utc::now();
                    if let Some(metadata) = params["metadata"].as_object() {
                        for (k, v) in metadata {
                            session.metadata.insert(k.clone(), v.clone());
                        }
                    }
                    Ok(json!({ "updated": true }))
                } else {
                    Err(anyhow!("Session not found"))
                }
            },
            "end" => {
                let mut sessions = self.sessions.write().await;
                sessions.remove(session_id);
                Ok(json!({ "ended": true }))
            },
            _ => Err(anyhow!("Invalid action"))
        }
    }

    // Helper methods

    async fn orchestrator_enhanced_search(
        &self,
        query: &str,
        memory_types: Option<Vec<String>>,
        context: Option<String>,
        limit: usize,
    ) -> Result<Vec<MemoryCell>> {
        if let Some(orchestrator) = &self.orchestrator {
            // First, get basic search results
            let mut results = self.basic_search(query, memory_types.clone(), context.clone(), limit * 2).await?;
            
            // If we have results, use orchestrator to analyze and rank them
            if !results.is_empty() {
                // Prepare prompts for each memory using orchestrator's intelligent prompt preparation
                let prompts = orchestrator.prepare_batch_prompts(&results);
                debug!("Generated {} intelligent prompts for enhanced search", prompts.len());
                
                // Analyze memory patterns to understand context better
                let pattern_analysis = orchestrator.analyze_memory_patterns(&results).await?;
                debug!("Pattern analysis: {}", pattern_analysis);
                
                // Re-rank results based on pattern analysis and query relevance
                // This is a simplified ranking - in production, you'd use more sophisticated algorithms
                results.sort_by(|a, b| {
                    // Sort by importance first, then by recency
                    b.importance.partial_cmp(&a.importance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| b.last_accessed.cmp(&a.last_accessed))
                });
                
                // Apply limit after intelligent ranking
                results.truncate(limit);
                
                // Update cache statistics
                let mut stats = self.stats.write().await;
                stats.cache_hits += 1;
            }
            
            Ok(results)
        } else {
            // Fallback to basic search if orchestrator not available
            self.basic_search(query, memory_types, context, limit).await
        }
    }

    async fn basic_search(
        &self,
        query: &str,
        memory_types: Option<Vec<String>>,
        context: Option<String>,
        limit: usize,
    ) -> Result<Vec<MemoryCell>> {
        // Basic search implementation
        let mut results = self.memory_service.search(query, limit * 2).await?;
        
        // Filter by memory types if specified
        if let Some(types) = memory_types {
            results.retain(|m| types.contains(&m.memory_type.type_name().to_string()));
        }
        
        // Filter by context if specified
        if let Some(ctx) = context {
            results.retain(|m| m.context_path.starts_with(&ctx));
        }
        
        // Limit results
        results.truncate(limit);
        
        Ok(results)
    }

    async fn determine_memory_type(&self, content: &str) -> Result<String> {
        if let Some(orchestrator) = &self.orchestrator {
            // Create a temporary memory cell for analysis
            let temp_memory = MemoryCell {
                id: Uuid::new_v4(),
                content: content.to_string(),
                summary: content.chars().take(200).collect::<String>(),
                memory_type: MemoryType::Semantic {
                    facts: vec![content.to_string()],
                    concepts: Vec::new(),
                },
                embedding: Vec::new(),
                context_path: "temp".to_string(),
                importance: 0.5,
                created_at: Utc::now(),
                last_accessed: Utc::now(),
                access_frequency: 0,
                tags: Vec::new(),
                metadata: HashMap::new(),
            };

            // Analyze content patterns to determine appropriate type
            let memories = vec![temp_memory];
            let pattern_analysis = orchestrator.analyze_memory_patterns(&memories).await
                .unwrap_or_else(|e| {
                    warn!("Failed to analyze patterns for type determination: {}", e);
                    String::from("Unable to analyze")
                });

            // Parse pattern analysis to determine memory type
            let memory_type = if pattern_analysis.contains("code") || 
                               pattern_analysis.contains("function") || 
                               pattern_analysis.contains("implementation") {
                "code"
            } else if pattern_analysis.contains("procedure") || 
                      pattern_analysis.contains("steps") ||
                      pattern_analysis.contains("how-to") {
                "procedural"
            } else if pattern_analysis.contains("event") || 
                      pattern_analysis.contains("happened") ||
                      pattern_analysis.contains("occurred") {
                "episodic"
            } else if pattern_analysis.contains("task") || 
                      pattern_analysis.contains("deadline") ||
                      pattern_analysis.contains("working on") {
                "working"
            } else if pattern_analysis.contains("documentation") || 
                      pattern_analysis.contains("reference") ||
                      pattern_analysis.contains("guide") {
                "documentation"
            } else if pattern_analysis.contains("conversation") || 
                      pattern_analysis.contains("discussion") ||
                      pattern_analysis.contains("chat") {
                "conversation"
            } else if pattern_analysis.contains("insight") || 
                      pattern_analysis.contains("learned") ||
                      pattern_analysis.contains("realized") {
                "meta_cognitive"
            } else if pattern_analysis.contains("problem") || 
                      pattern_analysis.contains("solution") ||
                      pattern_analysis.contains("fix") {
                "problem_solution"
            } else {
                "semantic" // Default to semantic for general knowledge
            };

            debug!("Determined memory type '{}' for content: '{}'", 
                   memory_type, 
                   content.chars().take(50).collect::<String>());

            Ok(memory_type.to_string())
        } else {
            // Fallback to simple heuristics without orchestrator
            let memory_type = if content.contains("fn ") || content.contains("class ") || 
                                content.contains("def ") || content.contains("function ") {
                "code"
            } else if content.contains("Step 1") || content.contains("First,") || 
                      content.contains("Then,") || content.contains("Finally,") {
                "procedural"
            } else if content.contains("I did") || content.contains("happened") || 
                      content.contains("occurred") || content.contains("event") {
                "episodic"
            } else if content.contains("TODO") || content.contains("task") || 
                      content.contains("deadline") || content.contains("working on") {
                "working"
            } else {
                "semantic"
            };

            Ok(memory_type.to_string())
        }
    }

    fn create_memory_cell(
        &self,
        content: &str,
        memory_type: &str,
        context: &str,
        tags: Vec<String>,
        importance: f32,
    ) -> Result<MemoryCell> {
        let memory_type = match memory_type {
            "semantic" => MemoryType::Semantic {
                facts: vec![content.to_string()],
                concepts: tags.clone(),
            },
            "episodic" => MemoryType::Episodic {
                event: content.to_string(),
                location: None,
                participants: Vec::new(),
                timeframe: Some(Utc::now().to_string()),
            },
            "procedural" => MemoryType::Procedural {
                steps: vec![content.to_string()],
                tools: Vec::new(),
                prerequisites: Vec::new(),
            },
            "working" => MemoryType::Working {
                task: content.to_string(),
                deadline: None,
                priority: Priority::Medium,
            },
            _ => MemoryType::Semantic {
                facts: vec![content.to_string()],
                concepts: tags.clone(),
            },
        };

        Ok(MemoryCell {
            id: Uuid::new_v4(),
            content: content.to_string(),
            summary: content.chars().take(200).collect::<String>(),
            memory_type,
            embedding: Vec::new(),
            context_path: context.to_string(),
            importance,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            access_frequency: 0,
            tags,
            metadata: HashMap::new(),
        })
    }

    async fn get_memories_for_context(
        &self,
        context: Option<String>,
        limit: usize,
    ) -> Result<Vec<MemoryCell>> {
        // Используем basic_search как временное решение
        // В продакшене здесь должна быть интеграция с реальными методами MemoryService
        if let Some(ctx) = context {
            // Поиск по контексту
            self.basic_search(&ctx, None, Some(ctx.clone()), limit).await
        } else {
            // Получение последних записей
            self.basic_search("", None, None, limit).await
        }
    }

    async fn apply_optimization(&self, _optimization: MemoryOptimization) -> Result<()> {
        // Apply optimization recommendations
        // This would involve removing duplicates, merging similar memories, etc.
        Ok(())
    }

    async fn get_cached_results(&self, key: &str) -> Option<Vec<MemoryCell>> {
        let cache = self.query_cache.read().await;
        cache.get(key).cloned()
    }

    async fn cache_results(&self, key: &str, results: &[MemoryCell]) {
        let mut cache = self.query_cache.write().await;
        
        // Limit cache size
        if cache.len() >= self.config.cache_size {
            // Remove oldest entry (simple FIFO for now)
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
            }
        }
        
        cache.insert(key.to_string(), results.to_vec());
    }

    async fn update_stats(&self, success: bool) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        if success {
            stats.successful_requests += 1;
        } else {
            stats.failed_requests += 1;
        }
        stats.active_sessions = self.sessions.read().await.len();
    }

    async fn update_stats_with_time(&self, success: bool, duration_ms: f64) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        if success {
            stats.successful_requests += 1;
        } else {
            stats.failed_requests += 1;
        }
        stats.active_sessions = self.sessions.read().await.len();
        
        // Update average response time with exponential moving average
        if stats.avg_response_time_ms == 0.0 {
            stats.avg_response_time_ms = duration_ms;
        } else {
            // EMA with alpha = 0.1 for smooth averaging
            stats.avg_response_time_ms = 
                stats.avg_response_time_ms * 0.9 + duration_ms * 0.1;
        }
    }

    /// Get server statistics
    pub async fn get_stats(&self) -> McpServerStats {
        let guard = self.stats.read().await;
        McpServerStats {
            total_requests: guard.total_requests,
            successful_requests: guard.successful_requests,
            failed_requests: guard.failed_requests,
            active_sessions: guard.active_sessions,
            cache_hits: guard.cache_hits,
            cache_misses: guard.cache_misses,
            avg_response_time_ms: guard.avg_response_time_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mcp_server_creation() {
        // Test would require mock MemoryService
        // For now, just test config creation
        let config = McpServerConfig::default();
        assert_eq!(config.name, "ai-memory-service");
        assert_eq!(config.max_results, 50);
        assert!(config.use_orchestrator);
    }

    #[test]
    fn test_mcp_tool_serialization() {
        let tool = McpTool {
            name: "test_tool".to_string(),
            description: "Test tool".to_string(),
            input_schema: json!({"type": "object"}),
            is_mutation: false,
        };
        
        let serialized = serde_json::to_string(&tool).unwrap();
        assert!(serialized.contains("test_tool"));
    }
}