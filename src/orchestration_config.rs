//! Configuration module for secure GPT-5-nano orchestration
//!
//! Provides comprehensive configuration management for orchestrator integration
//! with proper validation, environment variable support, and security settings.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::{debug, info, warn};

/// Main orchestration configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OrchestrationConfig {
    /// API configuration for GPT-5-nano
    pub api: ApiConfig,
    /// Security settings
    pub security: SecurityConfig,
    /// Performance tuning
    pub performance: PerformanceConfig,
    /// Data handling policies
    pub data_policy: DataPolicyConfig,
    /// Monitoring and audit settings
    pub monitoring: MonitoringConfig,
}

/// API configuration for GPT-5-nano
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ApiConfig {
    /// OpenAI API endpoint
    pub endpoint: String,
    /// API key (loaded from environment)
    #[serde(skip_serializing)]
    pub api_key: String,
    /// Model name
    pub model: String,
    /// Maximum input tokens
    pub max_input_tokens: usize,
    /// Maximum completion tokens
    pub max_completion_tokens: usize,
    /// Reasoning effort level for GPT-5
    pub reasoning_effort: String,
    /// Request timeout in seconds
    pub timeout_seconds: u64,
    /// Retry configuration
    pub retry: RetryConfig,
}

/// Retry configuration for API calls
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Initial retry delay in milliseconds
    pub initial_delay_ms: u64,
    /// Maximum retry delay in milliseconds
    pub max_delay_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f32,
}

/// Security configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SecurityConfig {
    /// Enable authentication requirement
    pub require_authentication: bool,
    /// Enable PII filtering
    pub filter_pii: bool,
    /// Enable sensitive data masking
    pub mask_sensitive_data: bool,
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
    /// Encryption settings
    pub encryption: EncryptionConfig,
    /// Allowed user roles for orchestration
    pub allowed_roles: Vec<String>,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RateLimitConfig {
    /// Requests per minute per user
    pub requests_per_minute: u32,
    /// Burst size for rate limiting
    pub burst_size: u32,
    /// Enable IP-based rate limiting
    pub ip_based: bool,
    /// Whitelist IPs (bypass rate limiting)
    pub whitelist_ips: Vec<String>,
}

/// Encryption configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EncryptionConfig {
    /// Enable encryption for data at rest
    pub encrypt_at_rest: bool,
    /// Enable encryption for data in transit
    pub encrypt_in_transit: bool,
    /// Encryption algorithm
    pub algorithm: String,
    /// Key rotation interval in days
    pub key_rotation_days: u32,
}

/// Performance configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PerformanceConfig {
    /// Maximum context size to send to GPT-5
    pub max_context_size: usize,
    /// Maximum memories to process at once
    pub max_memories_batch: usize,
    /// Cache configuration
    pub cache: CacheConfig,
    /// Parallel processing settings
    pub parallel: ParallelConfig,
}

/// Cache configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CacheConfig {
    /// Enable response caching
    pub enabled: bool,
    /// Cache TTL in seconds
    pub ttl_seconds: u64,
    /// Maximum cache size in MB
    pub max_size_mb: usize,
    /// Cache eviction policy
    pub eviction_policy: String,
}

/// Parallel processing configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ParallelConfig {
    /// Maximum concurrent orchestrator requests
    pub max_concurrent_requests: usize,
    /// Thread pool size
    pub thread_pool_size: usize,
    /// Queue size for pending requests
    pub queue_size: usize,
}

/// Data policy configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DataPolicyConfig {
    /// Data minimization settings
    pub minimization: DataMinimizationConfig,
    /// Data retention policies
    pub retention: DataRetentionConfig,
    /// Compliance settings
    pub compliance: ComplianceConfig,
}

/// Data minimization configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DataMinimizationConfig {
    /// Enable data minimization
    pub enabled: bool,
    /// Fields to exclude from external API calls
    pub exclude_fields: Vec<String>,
    /// Maximum text length to send
    pub max_text_length: usize,
    /// Truncation strategy
    pub truncation_strategy: String,
}

/// Data retention configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DataRetentionConfig {
    /// Audit log retention days
    pub audit_retention_days: u32,
    /// Session data retention hours
    pub session_retention_hours: u32,
    /// Cache data retention hours
    pub cache_retention_hours: u32,
}

/// Compliance configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ComplianceConfig {
    /// Enable GDPR compliance features
    pub gdpr_compliant: bool,
    /// Enable CCPA compliance features
    pub ccpa_compliant: bool,
    /// Enable HIPAA compliance features
    pub hipaa_compliant: bool,
    /// Data residency requirements
    pub data_residency: Vec<String>,
}

/// Monitoring configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MonitoringConfig {
    /// Enable audit logging
    pub audit_enabled: bool,
    /// Audit log level
    pub audit_level: String,
    /// Metrics collection
    pub metrics: MetricsConfig,
    /// Alerting configuration
    pub alerting: AlertingConfig,
}

/// Metrics configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    /// Metrics export interval in seconds
    pub export_interval_seconds: u64,
    /// Metrics endpoints
    pub endpoints: Vec<String>,
    /// Custom metrics tags
    pub tags: HashMap<String, String>,
}

/// Alerting configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AlertingConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Alert thresholds
    pub thresholds: AlertThresholds,
    /// Alert channels
    pub channels: Vec<String>,
}

/// Alert thresholds
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AlertThresholds {
    /// Error rate threshold (percentage)
    pub error_rate_percent: f32,
    /// Response time threshold (ms)
    pub response_time_ms: u64,
    /// Rate limit threshold (percentage of limit)
    pub rate_limit_percent: f32,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            api: ApiConfig::default(),
            security: SecurityConfig::default(),
            performance: PerformanceConfig::default(),
            data_policy: DataPolicyConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://api.openai.com/v1/chat/completions".to_string(),
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            model: "gpt-5-nano".to_string(),
            max_input_tokens: 400000,
            max_completion_tokens: 12000,
            reasoning_effort: "medium".to_string(),
            timeout_seconds: 120,
            retry: RetryConfig::default(),
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            require_authentication: true,
            filter_pii: true,
            mask_sensitive_data: true,
            rate_limit: RateLimitConfig::default(),
            encryption: EncryptionConfig::default(),
            allowed_roles: vec!["user".to_string(), "admin".to_string()],
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 30,
            burst_size: 10,
            ip_based: true,
            whitelist_ips: vec![],
        }
    }
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            encrypt_at_rest: true,
            encrypt_in_transit: true,
            algorithm: "AES-256-GCM".to_string(),
            key_rotation_days: 90,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_context_size: 4000,
            max_memories_batch: 50,
            cache: CacheConfig::default(),
            parallel: ParallelConfig::default(),
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ttl_seconds: 3600,
            max_size_mb: 100,
            eviction_policy: "LRU".to_string(),
        }
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 10,
            thread_pool_size: 4,
            queue_size: 100,
        }
    }
}

impl Default for DataPolicyConfig {
    fn default() -> Self {
        Self {
            minimization: DataMinimizationConfig::default(),
            retention: DataRetentionConfig::default(),
            compliance: ComplianceConfig::default(),
        }
    }
}

impl Default for DataMinimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            exclude_fields: vec!["password".to_string(), "api_key".to_string()],
            max_text_length: 10000,
            truncation_strategy: "middle".to_string(),
        }
    }
}

impl Default for DataRetentionConfig {
    fn default() -> Self {
        Self {
            audit_retention_days: 90,
            session_retention_hours: 24,
            cache_retention_hours: 12,
        }
    }
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            gdpr_compliant: true,
            ccpa_compliant: true,
            hipaa_compliant: false,
            data_residency: vec!["US".to_string(), "EU".to_string()],
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            audit_enabled: true,
            audit_level: "info".to_string(),
            metrics: MetricsConfig::default(),
            alerting: AlertingConfig::default(),
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            export_interval_seconds: 60,
            endpoints: vec![],
            tags: HashMap::new(),
        }
    }
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            thresholds: AlertThresholds::default(),
            channels: vec!["log".to_string()],
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            error_rate_percent: 5.0,
            response_time_ms: 5000,
            rate_limit_percent: 80.0,
        }
    }
}

impl OrchestrationConfig {
    /// Load configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from environment and file
    pub fn from_env_and_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut config = if path.as_ref().exists() {
            Self::from_file(path)?
        } else {
            warn!("Configuration file not found, using defaults");
            Self::default()
        };

        // Override with environment variables
        config.override_from_env();
        config.validate()?;

        info!("Orchestration configuration loaded successfully");
        Ok(config)
    }

    /// Override configuration from environment variables
    fn override_from_env(&mut self) {
        // API key from environment
        if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
            self.api.api_key = api_key;
        }

        // Model from environment
        if let Ok(model) = std::env::var("ORCHESTRATOR_MODEL") {
            self.api.model = model;
        }

        // Authentication requirement
        if let Ok(auth) = std::env::var("REQUIRE_AUTH") {
            self.security.require_authentication = auth.parse().unwrap_or(true);
        }

        // Rate limit
        if let Ok(rate) = std::env::var("RATE_LIMIT_PER_MINUTE") {
            if let Ok(limit) = rate.parse() {
                self.security.rate_limit.requests_per_minute = limit;
            }
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate API key
        if self.api.api_key.is_empty() {
            return Err(anyhow!("OpenAI API key is required"));
        }

        // Validate model
        if !self.api.model.starts_with("gpt-5") && !self.api.model.starts_with("gpt-4") {
            warn!("Using non-GPT-5 model: {}", self.api.model);
        }

        // Validate token limits
        if self.api.max_completion_tokens > 128000 {
            return Err(anyhow!(
                "max_completion_tokens exceeds GPT-5-nano limit of 128K"
            ));
        }

        // Validate rate limits
        if self.security.rate_limit.requests_per_minute == 0 {
            return Err(anyhow!("Rate limit must be greater than 0"));
        }

        // Validate cache settings
        if self.performance.cache.enabled && self.performance.cache.ttl_seconds == 0 {
            return Err(anyhow!(
                "Cache TTL must be greater than 0 when caching is enabled"
            ));
        }

        debug!("Configuration validation successful");
        Ok(())
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}
