//! Metrics collection and monitoring
//! 
//! Provides Prometheus-compatible metrics for monitoring service health and performance.

use prometheus::{
    register_counter_vec, register_gauge_vec, register_histogram_vec,
    CounterVec, GaugeVec, HistogramVec, TextEncoder, Encoder,
};
use lazy_static::lazy_static;

lazy_static! {
    /// Memory operation counters
    pub static ref MEMORY_OPS: CounterVec = register_counter_vec!(
        "memory_operations_total",
        "Total number of memory operations",
        &["operation", "status"]
    ).unwrap();
    
    /// Memory recall latency histogram
    pub static ref RECALL_LATENCY: HistogramVec = register_histogram_vec!(
        "memory_recall_duration_seconds",
        "Memory recall duration in seconds",
        &["query_type"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    ).unwrap();
    
    /// Active memory count gauge
    pub static ref MEMORY_COUNT: GaugeVec = register_gauge_vec!(
        "memory_count",
        "Number of memories by type",
        &["memory_type"]
    ).unwrap();
    
    /// Cache hit rate gauge
    pub static ref CACHE_HIT_RATE: GaugeVec = register_gauge_vec!(
        "cache_hit_rate",
        "Cache hit rate by level",
        &["cache_level"]
    ).unwrap();
    
    /// Storage size gauge
    pub static ref STORAGE_SIZE: GaugeVec = register_gauge_vec!(
        "storage_size_bytes",
        "Storage size in bytes",
        &["storage_type"]
    ).unwrap();
    
    /// Embedding generation latency
    pub static ref EMBEDDING_LATENCY: HistogramVec = register_histogram_vec!(
        "embedding_generation_duration_seconds",
        "Embedding generation duration in seconds",
        &["model"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
    ).unwrap();
    
    /// Active connections gauge
    pub static ref ACTIVE_CONNECTIONS: GaugeVec = register_gauge_vec!(
        "active_connections",
        "Number of active connections",
        &["connection_type"]
    ).unwrap();
}

/// Record a memory operation
pub fn record_memory_op(operation: &str, success: bool) {
    let status = if success { "success" } else { "failure" };
    MEMORY_OPS.with_label_values(&[operation, status]).inc();
}

/// Record recall latency
pub fn record_recall_latency(query_type: &str, duration: f64) {
    RECALL_LATENCY
        .with_label_values(&[query_type])
        .observe(duration);
}

/// Update memory count
pub fn update_memory_count(memory_type: &str, count: f64) {
    MEMORY_COUNT
        .with_label_values(&[memory_type])
        .set(count);
}

/// Update cache hit rate
pub fn update_cache_hit_rate(level: &str, rate: f64) {
    CACHE_HIT_RATE
        .with_label_values(&[level])
        .set(rate);
}

/// Update storage size
pub fn update_storage_size(storage_type: &str, size: f64) {
    STORAGE_SIZE
        .with_label_values(&[storage_type])
        .set(size);
}

/// Record embedding generation latency
pub fn record_embedding_latency(model: &str, duration: f64) {
    EMBEDDING_LATENCY
        .with_label_values(&[model])
        .observe(duration);
}

/// Update active connections
pub fn update_active_connections(conn_type: &str, count: f64) {
    ACTIVE_CONNECTIONS
        .with_label_values(&[conn_type])
        .set(count);
}

/// Export metrics in Prometheus format
pub fn export_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

/// Initialize metrics system
pub fn init_metrics() {
    // Force lazy_static initialization
    let _ = &*MEMORY_OPS;
    let _ = &*RECALL_LATENCY;
    let _ = &*MEMORY_COUNT;
    let _ = &*CACHE_HIT_RATE;
    let _ = &*STORAGE_SIZE;
    let _ = &*EMBEDDING_LATENCY;
    let _ = &*ACTIVE_CONNECTIONS;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_operations() {
        record_memory_op("store", true);
        record_memory_op("recall", false);
        
        // Verify metrics are recorded
        let metrics = export_metrics();
        assert!(metrics.contains("memory_operations_total"));
    }

    #[test]
    fn test_latency_recording() {
        record_recall_latency("semantic", 0.05);
        record_embedding_latency("gemma-300m", 0.02);
        
        let metrics = export_metrics();
        assert!(metrics.contains("memory_recall_duration_seconds"));
        assert!(metrics.contains("embedding_generation_duration_seconds"));
    }
}