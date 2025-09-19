//! Advanced monitoring and observability system for AI Memory Service
//!
//! Provides comprehensive metrics collection, health checks, and system monitoring
//! with Windows-specific optimizations and performance tracking.

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, atomic::{AtomicU64, AtomicUsize, Ordering}};
use std::sync::RwLock;
use serde::{Serialize, Deserialize};
use tokio::sync::Mutex;
use tracing::info;
use prometheus::{
    Counter, Histogram, Gauge, IntGauge, Registry, Opts, HistogramOpts,
    register_counter_with_registry, register_histogram_with_registry, 
    register_gauge_with_registry, register_int_gauge_with_registry,
    Encoder, TextEncoder
};
use axum::{
    http::StatusCode,
    response::{Json, Response},
    routing::get,
    Router,
};
use crate::types::MemoryResult;
// use crate::model_manager::ModelRegistryStats;  // Not used currently

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Component health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub message: String,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub check_duration_ms: u64,
    pub details: Option<serde_json::Value>,
}

/// System performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub uptime_seconds: u64,
    pub memory_usage: MemoryMetrics,
    pub performance: PerformanceMetrics,
    pub storage: StorageMetrics,
    pub requests: RequestMetrics,
    pub models: ModelMetrics,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub total_memories: u64,
    pub cache_hit_rate_l1: f64,
    pub cache_hit_rate_l2: f64,
    pub cache_size_bytes: u64,
    pub embedding_cache_size: u64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub avg_store_latency_ms: f64,
    pub avg_recall_latency_ms: f64,
    pub avg_embedding_latency_ms: f64,
    pub requests_per_second: f64,
    pub errors_per_minute: f64,
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
}

/// Storage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetrics {
    pub neo4j_connection_pool_active: u32,
    pub neo4j_query_success_rate: f64,
    pub avg_query_time_ms: f64,
    pub storage_size_bytes: u64,
}

/// Request metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub active_connections: u32,
}

/// Model metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub loaded_models: usize,
    pub model_cache_size_bytes: u64,
    pub model_load_time_ms: f64,
    pub inference_time_ms: f64,
}

/// Prometheus metrics collector
pub struct MetricsCollector {
    pub registry: Registry,
    
    // Counters
    pub requests_total: Counter,
    pub errors_total: Counter,
    pub memories_stored_total: Counter,
    pub memories_recalled_total: Counter,
    
    // Histograms
    pub request_duration: Histogram,
    pub store_duration: Histogram,
    pub recall_duration: Histogram,
    pub embedding_duration: Histogram,
    pub neo4j_query_duration: Histogram,
    
    // Gauges
    pub active_connections: IntGauge,
    pub memory_count: IntGauge,
    pub cache_hit_rate_l1: Gauge,
    pub cache_hit_rate_l2: Gauge,
    pub system_memory_bytes: Gauge,
    pub model_count: IntGauge,
}

impl MetricsCollector {
    pub fn new() -> MemoryResult<Self> {
        let registry = Registry::new();
        
        // Create counters
        let requests_total = register_counter_with_registry!(
            Opts::new("memory_requests_total", "Total number of memory requests"),
            registry
        )?;
        
        let errors_total = register_counter_with_registry!(
            Opts::new("memory_errors_total", "Total number of errors"),
            registry
        )?;
        
        let memories_stored_total = register_counter_with_registry!(
            Opts::new("memories_stored_total", "Total memories stored"),
            registry
        )?;
        
        let memories_recalled_total = register_counter_with_registry!(
            Opts::new("memories_recalled_total", "Total memories recalled"),
            registry
        )?;
        
        // Create histograms
        let request_duration = register_histogram_with_registry!(
            HistogramOpts::new("request_duration_seconds", "Request duration in seconds"),
            registry
        )?;
        
        let store_duration = register_histogram_with_registry!(
            HistogramOpts::new("memory_store_duration_seconds", "Memory store duration"),
            registry
        )?;
        
        let recall_duration = register_histogram_with_registry!(
            HistogramOpts::new("memory_recall_duration_seconds", "Memory recall duration"),
            registry
        )?;
        
        let embedding_duration = register_histogram_with_registry!(
            HistogramOpts::new("embedding_duration_seconds", "Embedding generation duration"),
            registry
        )?;
        
        let neo4j_query_duration = register_histogram_with_registry!(
            HistogramOpts::new("neo4j_query_duration_seconds", "Neo4j query duration"),
            registry
        )?;
        
        // Create gauges
        let active_connections = register_int_gauge_with_registry!(
            Opts::new("active_connections", "Number of active connections"),
            registry
        )?;
        
        let memory_count = register_int_gauge_with_registry!(
            Opts::new("memory_count", "Total number of stored memories"),
            registry
        )?;
        
        let cache_hit_rate_l1 = register_gauge_with_registry!(
            Opts::new("cache_hit_rate_l1", "L1 cache hit rate"),
            registry
        )?;
        
        let cache_hit_rate_l2 = register_gauge_with_registry!(
            Opts::new("cache_hit_rate_l2", "L2 cache hit rate"),
            registry
        )?;
        
        let system_memory_bytes = register_gauge_with_registry!(
            Opts::new("system_memory_bytes", "System memory usage in bytes"),
            registry
        )?;
        
        let model_count = register_int_gauge_with_registry!(
            Opts::new("loaded_models", "Number of loaded models"),
            registry
        )?;
        
        Ok(MetricsCollector {
            registry,
            requests_total,
            errors_total,
            memories_stored_total,
            memories_recalled_total,
            request_duration,
            store_duration,
            recall_duration,
            embedding_duration,
            neo4j_query_duration,
            active_connections,
            memory_count,
            cache_hit_rate_l1,
            cache_hit_rate_l2,
            system_memory_bytes,
            model_count,
        })
    }
    
    /// Export metrics in Prometheus format
    pub fn export_metrics(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

/// System monitoring and health tracking
pub struct SystemMonitor {
    start_time: Instant,
    metrics: Arc<MetricsCollector>,
    health_checks: Arc<RwLock<HashMap<String, ComponentHealth>>>,
    system_stats: Arc<Mutex<SystemMetrics>>,
    
    // Performance counters
    request_count: AtomicU64,
    error_count: AtomicU64,
    total_latency: AtomicU64,
    active_requests: AtomicUsize,
}

impl SystemMonitor {
    pub fn new() -> MemoryResult<Self> {
        let metrics = Arc::new(MetricsCollector::new()?);
        
        Ok(SystemMonitor {
            start_time: Instant::now(),
            metrics,
            health_checks: Arc::new(RwLock::new(HashMap::new())),
            system_stats: Arc::new(Mutex::new(SystemMetrics {
                timestamp: chrono::Utc::now(),
                uptime_seconds: 0,
                memory_usage: MemoryMetrics {
                    total_memories: 0,
                    cache_hit_rate_l1: 0.0,
                    cache_hit_rate_l2: 0.0,
                    cache_size_bytes: 0,
                    embedding_cache_size: 0,
                },
                performance: PerformanceMetrics {
                    avg_store_latency_ms: 0.0,
                    avg_recall_latency_ms: 0.0,
                    avg_embedding_latency_ms: 0.0,
                    requests_per_second: 0.0,
                    errors_per_minute: 0.0,
                    cpu_usage_percent: 0.0,
                    memory_usage_bytes: 0,
                },
                storage: StorageMetrics {
                    neo4j_connection_pool_active: 0,
                    neo4j_query_success_rate: 1.0,
                    avg_query_time_ms: 0.0,
                    storage_size_bytes: 0,
                },
                requests: RequestMetrics {
                    total_requests: 0,
                    successful_requests: 0,
                    failed_requests: 0,
                    active_connections: 0,
                },
                models: ModelMetrics {
                    loaded_models: 0,
                    model_cache_size_bytes: 0,
                    model_load_time_ms: 0.0,
                    inference_time_ms: 0.0,
                },
            })),
            request_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            total_latency: AtomicU64::new(0),
            active_requests: AtomicUsize::new(0),
        })
    }
    
    /// Record a request with timing
    pub fn record_request(&self, duration: Duration, success: bool) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.total_latency.fetch_add(duration.as_millis() as u64, Ordering::Relaxed);
        
        if success {
            self.metrics.requests_total.inc();
        } else {
            self.metrics.errors_total.inc();
            self.error_count.fetch_add(1, Ordering::Relaxed);
        }
        
        self.metrics.request_duration.observe(duration.as_secs_f64());
    }
    
    /// Record memory operation
    pub fn record_memory_operation(&self, operation: &str, duration: Duration, success: bool) {
        match operation {
            "store" => {
                self.metrics.store_duration.observe(duration.as_secs_f64());
                if success {
                    self.metrics.memories_stored_total.inc();
                }
            },
            "recall" => {
                self.metrics.recall_duration.observe(duration.as_secs_f64());
                if success {
                    self.metrics.memories_recalled_total.inc();
                }
            },
            "embedding" => {
                self.metrics.embedding_duration.observe(duration.as_secs_f64());
            },
            _ => {}
        }
    }
    
    /// Update component health
    pub fn update_component_health(&self, component: &str, status: HealthStatus, message: String) {
        let health = ComponentHealth {
            name: component.to_string(),
            status,
            message,
            last_check: chrono::Utc::now(),
            check_duration_ms: 0, // Would be measured in real health check
            details: None,
        };
        
        if let Ok(mut health_checks) = self.health_checks.write() {
            health_checks.insert(component.to_string(), health);
        }
    }
    
    /// Get overall system health
    pub fn get_system_health(&self) -> HealthStatus {
        if let Ok(health_checks) = self.health_checks.read() {
            if health_checks.is_empty() {
                return HealthStatus::Unknown;
            }
            
            let mut _healthy_count = 0;
            let mut degraded_count = 0;
            let mut unhealthy_count = 0;
            
            for health in health_checks.values() {
                match health.status {
                    HealthStatus::Healthy => _healthy_count += 1,
                    HealthStatus::Degraded => degraded_count += 1,
                    HealthStatus::Unhealthy => unhealthy_count += 1,
                    HealthStatus::Unknown => {},
                }
            }
            
            if unhealthy_count > 0 {
                HealthStatus::Unhealthy
            } else if degraded_count > 0 {
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            }
        } else {
            HealthStatus::Unknown
        }
    }
    
    /// Get system statistics
    pub async fn get_system_stats(&self) -> SystemMetrics {
        let mut stats = self.system_stats.lock().await;
        
        // Update basic metrics
        stats.timestamp = chrono::Utc::now();
        stats.uptime_seconds = self.start_time.elapsed().as_secs();
        
        // Update request metrics
        let total_requests = self.request_count.load(Ordering::Relaxed);
        let error_count = self.error_count.load(Ordering::Relaxed);
        
        stats.requests.total_requests = total_requests;
        stats.requests.successful_requests = total_requests.saturating_sub(error_count);
        stats.requests.failed_requests = error_count;
        stats.requests.active_connections = self.active_requests.load(Ordering::Relaxed) as u32;
        
        // Calculate averages
        if total_requests > 0 {
            let total_latency = self.total_latency.load(Ordering::Relaxed);
            stats.performance.avg_store_latency_ms = total_latency as f64 / total_requests as f64;
        }
        
        // Update system memory usage (Windows-specific)
        #[cfg(target_os = "windows")]
        {
            if let Ok(memory_info) = get_windows_memory_info() {
                stats.performance.memory_usage_bytes = memory_info;
            }
        }
        
        stats.clone()
    }
    
    /// Export Prometheus metrics
    pub fn export_prometheus_metrics(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.metrics.export_metrics()
    }
}

/// Get Windows memory information
#[cfg(target_os = "windows")]
fn get_windows_memory_info() -> Result<u64, Box<dyn std::error::Error>> {
    use std::mem;
    use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use winapi::um::processthreadsapi::GetCurrentProcess;
    
    unsafe {
        let mut pmc: PROCESS_MEMORY_COUNTERS = mem::zeroed();
        pmc.cb = mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        
        if GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
            Ok(pmc.WorkingSetSize as u64)
        } else {
            Err("Failed to get process memory info".into())
        }
    }
}

/// Global system monitor instance
static SYSTEM_MONITOR: OnceLock<Arc<SystemMonitor>> = OnceLock::new();

impl SystemMonitor {
    /// Initialize global monitor
    pub fn initialize() -> MemoryResult<Arc<Self>> {
        // Check if already initialized
        if let Some(existing) = SYSTEM_MONITOR.get() {
            return Ok(existing.clone());
        }
        
        // Create new monitor
        let monitor = Arc::new(Self::new()?);
        
        // Try to set it, or return existing if another thread beat us
        match SYSTEM_MONITOR.set(monitor.clone()) {
            Ok(()) => {
                info!("System monitor initialized");
                Ok(monitor)
            },
            Err(_) => {
                // Another thread initialized it first
                Ok(SYSTEM_MONITOR.get().expect("Monitor should be initialized by another thread").clone())
            }
        }
    }
    
    /// Get global monitor instance
    pub fn global() -> Option<Arc<Self>> {
        SYSTEM_MONITOR.get().cloned()
    }
}

/// Create monitoring router for HTTP endpoints
pub fn create_monitoring_router() -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/stats", get(stats_handler))
        .route("/ready", get(ready_handler))
        .route("/live", get(live_handler))
}

/// Health check endpoint
async fn health_handler() -> Result<Json<serde_json::Value>, StatusCode> {
    if let Some(monitor) = SystemMonitor::global() {
        let health_status = monitor.get_system_health();
        let health_checks = monitor.health_checks.read().unwrap();
        
        let response = serde_json::json!({
            "status": health_status,
            "timestamp": chrono::Utc::now(),
            "uptime_seconds": monitor.start_time.elapsed().as_secs(),
            "components": health_checks.clone(),
        });
        
        match health_status {
            HealthStatus::Healthy => Ok(Json(response)),
            _ => Err(StatusCode::SERVICE_UNAVAILABLE),
        }
    } else {
        Err(StatusCode::INTERNAL_SERVER_ERROR)
    }
}

/// Prometheus metrics endpoint
async fn metrics_handler() -> Result<Response<String>, StatusCode> {
    if let Some(monitor) = SystemMonitor::global() {
        match monitor.export_prometheus_metrics() {
            Ok(metrics) => {
                let response = Response::builder()
                    .header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    .body(metrics)
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                Ok(response)
            },
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// System statistics endpoint
async fn stats_handler() -> Result<Json<SystemMetrics>, StatusCode> {
    if let Some(monitor) = SystemMonitor::global() {
        let stats = monitor.get_system_stats().await;
        Ok(Json(stats))
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Readiness probe endpoint
async fn ready_handler() -> Result<Json<serde_json::Value>, StatusCode> {
    if let Some(monitor) = SystemMonitor::global() {
        let health_status = monitor.get_system_health();
        match health_status {
            HealthStatus::Healthy | HealthStatus::Degraded => {
                Ok(Json(serde_json::json!({
                    "status": "ready",
                    "timestamp": chrono::Utc::now()
                })))
            },
            _ => Err(StatusCode::SERVICE_UNAVAILABLE),
        }
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

/// Liveness probe endpoint
async fn live_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "alive",
        "timestamp": chrono::Utc::now()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_health_status() {
        let health = ComponentHealth {
            name: "test".to_string(),
            status: HealthStatus::Healthy,
            message: "OK".to_string(),
            last_check: chrono::Utc::now(),
            check_duration_ms: 10,
            details: None,
        };
        
        assert_eq!(health.status, HealthStatus::Healthy);
    }
    
    #[tokio::test]
    async fn test_system_monitor() {
        let monitor = SystemMonitor::new().expect("Failed to create monitor");
        
        monitor.record_request(Duration::from_millis(100), true);
        let stats = monitor.get_system_stats().await;
        
        assert_eq!(stats.requests.total_requests, 1);
        assert_eq!(stats.requests.successful_requests, 1);
    }
}