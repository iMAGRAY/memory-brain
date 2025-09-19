use crate::types::{MemoryError, MemoryResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, warn};

/// Security validator for input sanitization and validation
pub struct SecurityValidator {
    blocked_patterns: Vec<String>,
    max_content_length: usize,
    max_query_length: usize,
}

impl SecurityValidator {
    pub fn new() -> Self {
        Self {
            blocked_patterns: vec![
                "DROP".to_string(),
                "DELETE".to_string(), 
                "UPDATE".to_string(),
                "CREATE".to_string(),
                "ALTER".to_string(),
                "TRUNCATE".to_string(),
                "MATCH".to_string(),
                "MERGE".to_string(),
                "<script".to_string(),
                "javascript:".to_string(),
                "eval(".to_string(),
                "exec(".to_string(),
            ],
            max_content_length: 100_000, // 100KB max
            max_query_length: 10_000,    // 10KB max query
        }
    }

    /// Validate and sanitize content before storage
    pub fn validate_content(&self, content: &str) -> MemoryResult<()> {
        if content.is_empty() {
            return Err(MemoryError::Validation("Content cannot be empty".to_string()));
        }

        if content.len() > self.max_content_length {
            return Err(MemoryError::Validation(
                format!("Content exceeds maximum length of {} bytes", self.max_content_length)
            ));
        }

        // Check for potentially malicious patterns
        let content_upper = content.to_uppercase();
        for pattern in &self.blocked_patterns {
            if content_upper.contains(pattern) {
                warn!("Blocked suspicious content containing pattern: {}", pattern);
                return Err(MemoryError::Validation(
                    "Content contains potentially malicious patterns".to_string()
                ));
            }
        }

        Ok(())
    }

    /// Validate query input
    pub fn validate_query(&self, query: &str) -> MemoryResult<()> {
        if query.is_empty() {
            return Err(MemoryError::Validation("Query cannot be empty".to_string()));
        }

        if query.len() > self.max_query_length {
            return Err(MemoryError::Validation(
                format!("Query exceeds maximum length of {} bytes", self.max_query_length)
            ));
        }

        // Basic SQL/Cypher injection prevention
        let query_upper = query.to_uppercase();
        for pattern in &self.blocked_patterns {
            if query_upper.contains(pattern) {
                warn!("Blocked suspicious query containing pattern: {}", pattern);
                return Err(MemoryError::Validation(
                    "Query contains potentially malicious patterns".to_string()
                ));
            }
        }

        Ok(())
    }

    /// Sanitize context path
    pub fn sanitize_context(&self, context: &str) -> String {
        // Remove potentially dangerous characters
        context
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '/' || *c == '-' || *c == '_')
            .take(1000) // Limit context path length
            .collect()
    }
}

/// Rate limiter for API endpoints
pub struct RateLimiter {
    requests: Arc<RwLock<HashMap<String, Vec<u64>>>>,
    max_requests_per_minute: usize,
}

impl RateLimiter {
    pub fn new(max_requests_per_minute: usize) -> Self {
        Self {
            requests: Arc::new(RwLock::new(HashMap::new())),
            max_requests_per_minute,
        }
    }

    /// Check if request is within rate limits
    pub async fn check_rate_limit(&self, client_id: &str) -> MemoryResult<()> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut requests = self.requests.write().await;
        let client_requests = requests.entry(client_id.to_string()).or_insert_with(Vec::new);

        // Remove requests older than 1 minute
        client_requests.retain(|&timestamp| now - timestamp < 60);

        if client_requests.len() >= self.max_requests_per_minute {
            error!("Rate limit exceeded for client: {}", client_id);
            return Err(MemoryError::RateLimit(
                "Too many requests. Please slow down.".to_string()
            ));
        }

        client_requests.push(now);
        Ok(())
    }

    /// Clean up old entries periodically
    pub async fn cleanup(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut requests = self.requests.write().await;
        requests.retain(|_, timestamps| {
            timestamps.retain(|&timestamp| now - timestamp < 60);
            !timestamps.is_empty()
        });
    }
}

/// Audit logger for security events
pub struct AuditLogger;

impl AuditLogger {
    pub fn log_access(&self, client_id: &str, operation: &str, success: bool) {
        if success {
            tracing::info!(
                client_id = %client_id,
                operation = %operation,
                success = %success,
                "Memory operation completed"
            );
        } else {
            tracing::warn!(
                client_id = %client_id,
                operation = %operation,
                success = %success,
                "Memory operation failed"
            );
        }
    }

    pub fn log_security_event(&self, client_id: &str, event: &str, severity: &str) {
        tracing::warn!(
            client_id = %client_id,
            event = %event,
            severity = %severity,
            "Security event detected"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_validation() {
        let validator = SecurityValidator::new();

        // Valid content
        assert!(validator.validate_content("This is normal content").is_ok());

        // Empty content
        assert!(validator.validate_content("").is_err());

        // Suspicious content
        assert!(validator.validate_content("DROP TABLE users").is_err());
        assert!(validator.validate_content("<script>alert('xss')</script>").is_err());
    }

    #[test]
    fn test_query_validation() {
        let validator = SecurityValidator::new();

        // Valid query
        assert!(validator.validate_query("Find memories about Rust").is_ok());

        // Suspicious query
        assert!(validator.validate_query("DROP DATABASE").is_err());
        assert!(validator.validate_query("MATCH (n) DELETE n").is_err());
    }

    #[test]
    fn test_context_sanitization() {
        let validator = SecurityValidator::new();

        let result = validator.sanitize_context("programming/rust-lang");
        assert_eq!(result, "programming/rust-lang");

        let result = validator.sanitize_context("evil<script>injection");
        assert_eq!(result, "evilscriptinjection");
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let limiter = RateLimiter::new(2); // 2 requests per minute

        // First two requests should succeed
        assert!(limiter.check_rate_limit("client1").await.is_ok());
        assert!(limiter.check_rate_limit("client1").await.is_ok());

        // Third request should fail
        assert!(limiter.check_rate_limit("client1").await.is_err());
    }
}