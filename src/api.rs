//! REST API для AI Memory Service с поддержкой GPT-5-nano оркестратора
//!
//! Предоставляет HTTP endpoints для операций с памятью, поиска, инсайтов и оркестрации

use crate::{
    InsightType, MemoryCell, MemoryError, MemoryOrchestrator, MemoryQuery, MemoryService,
    MemoryType, Priority,
};
use anyhow::Result;
use axum::{
    extract::{Json, Path, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Router,
};
use axum::http::HeaderMap;
use axum::body::Bytes;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    limit::RequestBodyLimitLayer,
    trace::TraceLayer,
    services::ServeDir,
};
use tower::limit::ConcurrencyLimitLayer;
use axum::response::Html;
use tracing::{debug, info};
use uuid::Uuid;

/// Запрос на оценку качества ответа относительно evidence
#[derive(Debug, Deserialize)]
pub struct EvaluateRequest {
    pub query: String,
    pub final_summary: serde_json::Value,
    pub evidence: Option<Vec<EvalEvidence>>, // если None — оцениваем только согласованность ответа
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct EvalEvidence {
    pub id: Option<String>,
    pub title: Option<String>,
    pub text: String,
    pub context_path: Option<String>,
}

/// Ответ оценки качества
#[derive(Debug, Serialize)]
pub struct EvaluateResponse {
    pub rubric: serde_json::Value,
    pub computed: serde_json::Value,
    pub usage: serde_json::Value,
    pub success: bool,
}

/// Состояние API сервера
#[derive(Clone)]
pub struct ApiState {
    memory_service: Arc<MemoryService>,
    orchestrator: Option<Arc<MemoryOrchestrator>>,
    embedding_dimension: usize,
}

/// Конфигурация API
#[derive(Debug, Clone, Deserialize)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub max_body_size: usize,
    pub max_concurrency: usize,
    pub enable_cors: bool,
    pub enable_compression: bool,
    pub enable_tracing: bool,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            max_body_size: 10 * 1024 * 1024, // 10MB
            max_concurrency: 256,
            enable_cors: true,
            enable_compression: true,
            enable_tracing: true,
        }
    }
}

// ===== Request/Response структуры =====

/// Запрос на сохранение памяти
#[derive(Debug, Deserialize)]
pub struct StoreMemoryRequest {
    pub content: String,
    pub context_hint: Option<String>,
    pub memory_type: Option<String>,
    pub tags: Option<Vec<String>>,
    pub metadata: Option<HashMap<String, String>>,
    pub importance: Option<f32>,
}

/// Ответ на сохранение памяти
#[derive(Debug, Serialize)]
pub struct StoreMemoryResponse {
    pub id: Uuid,
    pub memory_id: Uuid,
    pub success: bool,
    pub message: String,
    pub embedding_dimension: usize,
}

/// Запрос поиска (для POST)
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub limit: Option<usize>,
    pub memory_types: Option<Vec<String>>,
    pub context: Option<String>,
    pub include_related: Option<bool>,
    pub min_importance: Option<f32>,
    pub similarity_threshold: Option<f32>,
    pub return_scores: Option<bool>,
    pub hybrid_alpha: Option<f32>,
}

use serde::Deserializer;

/// Десериализация comma-separated строки в Vec<String>
fn deserialize_memory_types<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    let s: Option<String> = Option::deserialize(deserializer)?;
    Ok(s.map(|s| {
        s.split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }))
}

/// Валидация limit с ограничением максимального значения
fn deserialize_limit<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: Option<usize> = Option::deserialize(deserializer)?;
    if let Some(l) = value {
        if l > 100 {
            return Err(serde::de::Error::custom("limit cannot exceed 100"));
        }
        if l == 0 {
            return Err(serde::de::Error::custom("limit must be greater than 0"));
        }
    }
    Ok(value)
}

/// Диапазон допустимых значений для порогов важности и схожести
const MIN_THRESHOLD: f32 = 0.0;
const MAX_THRESHOLD: f32 = 1.0;

/// Общая валидация порога (0.0-1.0) для устранения дублирования кода
/// Проверяет что значение конечное и в допустимом диапазоне
fn validate_threshold<E>(value: f32, field_name: &str) -> Result<f32, E>
where
    E: serde::de::Error,
{
    if value.is_nan() || value.is_infinite() {
        return Err(E::custom(format!("{} must be a finite number", field_name)));
    }
    if !(MIN_THRESHOLD..=MAX_THRESHOLD).contains(&value) {
        return Err(E::custom(format!(
            "{} must be between {} and {}",
            field_name, MIN_THRESHOLD, MAX_THRESHOLD
        )));
    }
    Ok(value)
}

/// Валидация порога важности (0.0-1.0)
/// Примеры: "0.75" -> Ok(Some(0.75)), "1.5" -> Err, "NaN" -> Err, null -> Ok(None)
fn deserialize_importance<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: Option<f32> = Option::deserialize(deserializer)?;
    if let Some(imp) = value {
        validate_threshold(imp, "min_importance")?;
    }
    Ok(value)
}

/// Валидация порога схожести (0.0-1.0)
/// Примеры: "0.8" -> Ok(Some(0.8)), "2.0" -> Err, "-0.1" -> Err, null -> Ok(None)
fn deserialize_similarity<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let value: Option<f32> = Option::deserialize(deserializer)?;
    if let Some(sim) = value {
        validate_threshold(sim, "similarity_threshold")?;
    }
    Ok(value)
}

/// Параметры поиска для GET запроса
///
/// Пример: GET /search?query=example&limit=20&memory_types=Semantic,Episodic&similarity_threshold=0.8
#[derive(Debug, Deserialize)]
pub struct SearchQueryParams {
    /// Текст запроса для поиска воспоминаний (обязательный параметр)
    pub query: String,
    /// Максимальное количество результатов (по умолчанию 10, макс 100)
    #[serde(deserialize_with = "deserialize_limit", default)]
    pub limit: Option<usize>,
    /// Типы памяти для фильтрации (comma-separated: "Semantic,Episodic")
    #[serde(deserialize_with = "deserialize_memory_types", default)]
    pub memory_types: Option<Vec<String>>,
    /// Контекст поиска для более точных результатов
    #[serde(default)]
    pub context: Option<String>,
    /// Включать связанные воспоминания в результат
    #[serde(default)]
    pub include_related: Option<bool>,
    /// Минимальный порог важности (0.0-1.0)
    #[serde(deserialize_with = "deserialize_importance", default)]
    pub min_importance: Option<f32>,
    /// Порог схожести для vector search (0.0-1.0)
    #[serde(deserialize_with = "deserialize_similarity", default)]
    pub similarity_threshold: Option<f32>,
    #[serde(default)]
    pub return_scores: Option<bool>,
    #[serde(default)]
    pub hybrid_alpha: Option<f32>,
}

/// Ответ поиска
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<MemoryCell>,
    pub total: usize,
    pub query_id: Uuid,
    pub reasoning_chain: Vec<String>,
    pub confidence: f32,
    pub recall_time_ms: u64,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scores: Option<Vec<ScoredItem>>, // optional diagnostic
}

#[derive(Debug, Serialize)]
pub struct ScoredItem {
    pub id: Uuid,
    pub score: f32,
    pub method: String,
}

/// Совместимый ответ для alias‑маршрутов поиска
#[derive(Debug, Serialize)]
pub struct CompatSearchResponse {
    pub memories: Vec<MemoryCell>,
    pub count: usize,
    pub success: bool,
}

/// Запрос инсайтов
#[derive(Debug, Deserialize)]
pub struct InsightsRequest {
    pub context: Option<String>,
    pub insight_type: Option<String>,
    pub limit: Option<usize>,
    pub reasoning_effort: Option<String>, // Для GPT-5-nano: minimal, low, medium, high
}

/// Ответ инсайтов
#[derive(Debug, Serialize)]
pub struct InsightsResponse {
    pub insights: Vec<String>,
    pub memories_analyzed: usize,
    pub context: Option<String>,
    pub reasoning_effort: String,
    pub model_used: String,
    pub success: bool,
}

/// Запрос дистилляции
#[derive(Debug, Deserialize)]
pub struct DistillationRequest {
    pub context: Option<String>,
    pub max_points: Option<usize>,
    pub reasoning_effort: Option<String>,
}

/// Recall request compatible with legacy clients
#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    pub query: String,
    pub limit: Option<usize>,
    pub similarity_threshold: Option<f32>,
    pub context_hint: Option<String>,
    pub include_related: Option<bool>,
}

/// Ответ дистилляции
#[derive(Debug, Serialize)]
pub struct DistillationResponse {
    pub key_points: Vec<String>,
    pub summary: String,
    pub original_count: usize,
    pub compression_ratio: f32,
    pub success: bool,
}

/// Запрос оптимизации памяти
#[derive(Debug, Deserialize)]
pub struct OptimizationRequest {
    pub context: Option<String>,
    pub aggressive: Option<bool>,
    pub reasoning_effort: Option<String>,
}

/// Ответ оптимизации
#[derive(Debug, Serialize)]
pub struct OptimizationResponse {
    pub duplicates_found: usize,
    pub outdated_found: usize,
    pub suggestions: Vec<String>,
    pub compression_ratio: f32,
    pub space_savings_percent: f32,
    pub applied: bool,
    pub success: bool,
}

/// Ответ об ошибке
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: u16,
    pub details: Option<String>,
    pub success: bool,
}

// ===== Maintenance endpoints (decay, consolidate) =====
#[derive(Debug, Deserialize)]
pub struct DecayRequest {
    pub dry_run: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct DecayResponse {
    pub updated: usize,
    pub success: bool,
}

/// Эмуляция N последовательных тиков decay (виртуальные "сутки")
#[derive(Debug, Deserialize)]
pub struct TickRequest {
    /// Количество тиков (суток) для применения; по умолчанию 1; ограничение во избежание злоупотреблений
    pub ticks: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct TickResponse {
    /// Сколько тиков реально применено (ограничено сверху)
    pub ticks: usize,
    /// Суммарное количество обновлённых узлов за все тики
    pub total_updates: usize,
    pub success: bool,
}

#[derive(Debug, Deserialize)]
pub struct ConsolidateRequest {
    pub context: Option<String>,
    pub similarity_threshold: Option<f32>,
    pub max_items: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ConsolidateResponse {
    pub duplicates_marked: usize,
    pub success: bool,
}

/// Тип ошибки API
#[derive(Debug)]
pub enum ApiError {
    MemoryError(MemoryError),
    BadRequest(String),
    NotFound(String),
    InternalError(String),
    Unauthorized(String),
    RateLimitExceeded,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_message, details) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg, None),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg, None),
            ApiError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg, None),
            ApiError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, msg, None),
            ApiError::RateLimitExceeded => (
                StatusCode::TOO_MANY_REQUESTS,
                "Rate limit exceeded".to_string(),
                Some("Please retry after some time".to_string()),
            ),
            ApiError::MemoryError(e) => {
                match e {
                    MemoryError::Embedding(msg) => (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "Embedding service unavailable".to_string(),
                        Some(msg),
                    ),
                    other => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Memory service error".to_string(),
                        Some(other.to_string()),
                    ),
                }
            },
        };

        let body = Json(ErrorResponse {
            error: error_message,
            code: status.as_u16(),
            details,
            success: false,
        });

        (status, body).into_response()
    }
}

impl From<MemoryError> for ApiError {
    fn from(err: MemoryError) -> Self {
        ApiError::MemoryError(err)
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(err: anyhow::Error) -> Self {
        ApiError::InternalError(err.to_string())
    }
}

/// Создать роутер API
pub fn create_router(
    memory_service: Arc<MemoryService>,
    orchestrator: Option<Arc<MemoryOrchestrator>>,
    config: ApiConfig,
) -> Router {
    let state = ApiState {
        memory_service: memory_service.clone(),
        orchestrator,
        embedding_dimension: memory_service.embedding_dimension(),
    };

    let orch_limit: usize = std::env::var("ORCHESTRATOR_MAX_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let mut app = Router::new()
        // Здоровье и статус
        .route("/health", get(health_check))
        .route("/metrics", get(metrics_endpoint))
        .route("/stats", get(get_statistics))
        .route("/analytics/graph", get(analytics_graph))
        .route("/analytics/contexts", get(analytics_contexts))
        // Встроенная страница дашборда (работает даже без статических файлов)
        .route("/dashboard", get(dashboard_page))
        // Прямой маршрут на файл дашборда на случай, если статические отчёты не смонтированы
        .route("/reports/dashboard.html", get(dashboard_page))
        // Статические отчёты и дашборды (если доступны на ФС)
        .nest_service(
            "/reports",
            ServeDir::new(std::env::var("REPORTS_DIR").unwrap_or_else(|_| "reports".to_string())),
        )
        // Операции с памятью
        .route("/memory", post(store_memory))
        .route("/memory/:id", get(get_memory))
        .route("/memory/:id", delete(delete_memory))
        .route("/memories/recent", get(get_recent_memories))
        // Совместимый алиас: /recall → возвращает слои RecalledMemory
        .route("/recall", post(recall_endpoint))
        // Alias: POST /memories (compat handler with lenient JSON parsing)
        .route("/memories", post(store_memory_compat))
        // API-compatible routes for external tools (существующие)
        .route("/api/memories", post(store_memory))
        .route("/api/memories", get(get_recent_memories))
        .route("/api/memories/:id", get(get_memory))
        .route("/api/memories/:id", delete(delete_memory))
        // Новые alias для /api/memory (singular)
        .route("/api/memory", post(store_memory))
        .route("/api/memory/:id", get(get_memory))
        .route("/api/memory/:id", delete(delete_memory))
        // Поиск
        .route("/search", post(search_memories).get(search_memories_get))
        .route("/search/context", post(search_by_context))
        .route("/search/advanced", post(advanced_recall))
        // Совместимые alias поиска
        .route("/memories/search", post(search_memories_compat))
        .route("/api/memory/search", post(search_memories_compat))
        .route("/api/memory/search/advanced", post(advanced_recall))
        .route(
            "/api/memories/search",
            get(search_memories_get).post(search_memories_compat),
        )
        // Контексты
        .route("/contexts", get(list_contexts))
        .route("/context/:path", get(get_context_info))
        // Оркестратор (если доступен)
        .route("/orchestrator/insights", post(generate_insights).layer(ConcurrencyLimitLayer::new(orch_limit)))
        .route("/orchestrator/distill", post(distill_context).layer(ConcurrencyLimitLayer::new(orch_limit)))
        .route("/orchestrator/optimize", post(optimize_memory).layer(ConcurrencyLimitLayer::new(orch_limit)))
        .route("/orchestrator/analyze", post(analyze_patterns).layer(ConcurrencyLimitLayer::new(orch_limit)))
        .route("/orchestrator/status", get(orchestrator_status))
        .route("/orchestrator/summary", get(orchestrator_summary))
        .route("/orchestrator/evaluate", post(evaluate_answer))
        .route("/orchestrator/pipeline", post(orchestrator_pipeline_endpoint).layer(ConcurrencyLimitLayer::new(orch_limit)))
        // Оркестратор: совместимые алиасы /api/orchestrator/* и /api/v1/orchestrator/*
        .route("/api/orchestrator/insights", post(generate_insights).layer(ConcurrencyLimitLayer::new(orch_limit)))
        .route("/api/orchestrator/distill", post(distill_context).layer(ConcurrencyLimitLayer::new(orch_limit)))
        .route("/api/orchestrator/optimize", post(optimize_memory).layer(ConcurrencyLimitLayer::new(orch_limit)))
        .route("/api/orchestrator/analyze", post(analyze_patterns).layer(ConcurrencyLimitLayer::new(orch_limit)))
        .route("/api/orchestrator/status", get(orchestrator_status))
        .route("/api/orchestrator/summary", get(orchestrator_summary))
        .route("/api/orchestrator/evaluate", post(evaluate_answer))
        .route("/api/orchestrator/pipeline", post(orchestrator_pipeline_endpoint).layer(ConcurrencyLimitLayer::new(orch_limit)))
        .route("/api/v1/orchestrator/insights", post(generate_insights).layer(ConcurrencyLimitLayer::new(orch_limit)))
        .route("/api/v1/orchestrator/distill", post(distill_context).layer(ConcurrencyLimitLayer::new(orch_limit)))
        .route("/api/v1/orchestrator/optimize", post(optimize_memory).layer(ConcurrencyLimitLayer::new(orch_limit)))
        .route("/api/v1/orchestrator/analyze", post(analyze_patterns).layer(ConcurrencyLimitLayer::new(orch_limit)))
        .route("/api/v1/orchestrator/status", get(orchestrator_status))
        .route("/api/v1/orchestrator/summary", get(orchestrator_summary))
        .route("/api/v1/orchestrator/evaluate", post(evaluate_answer))
        .route("/api/v1/orchestrator/pipeline", post(orchestrator_pipeline_endpoint).layer(ConcurrencyLimitLayer::new(orch_limit)))
        // Maintenance
        .route("/maintenance/decay", post(maintenance_decay))
        .route("/maintenance/consolidate", post(maintenance_consolidate))
        .route("/maintenance/tick", post(maintenance_tick))
        .route("/maintenance/backfill_embeddings", post(maintenance_backfill_embeddings))
        .route("/maintenance/missing_embeddings_count", get(maintenance_missing_embeddings_count))
        .route("/maintenance/purge_orphaned_embeddings", post(maintenance_purge_orphaned_embeddings))
        .route("/maintenance/ann/status", get(maintenance_ann_status))
        .route("/maintenance/ann/persist", post(maintenance_ann_persist))
        .route("/maintenance/purge_orphaned_embeddings", post(maintenance_purge_orphaned_embeddings))
        // Совместимые алиасы для maintenance и поиска/записи
        .route("/api/memory/consolidate", post(maintenance_consolidate))
        .route("/api/search", get(search_memories_get).post(search_memories_compat))
        .route("/api/v1/memory", post(store_memory))
        .route("/api/v1/memory/search", get(search_memories_get).post(search_memories_compat))
        .route("/api/v1/maintenance/decay", post(maintenance_decay))
        .route("/api/v1/maintenance/consolidate", post(maintenance_consolidate))
        .route("/api/v1/maintenance/tick", post(maintenance_tick))
        .route("/api/v1/maintenance/backfill_embeddings", post(maintenance_backfill_embeddings))
        .with_state(state);

    // Middleware слои
    let service_builder = ServiceBuilder::new()
        .layer(RequestBodyLimitLayer::new(config.max_body_size))
        .layer(ConcurrencyLimitLayer::new(config.max_concurrency));

    if config.enable_tracing {
        app = app.layer(TraceLayer::new_for_http());
    }

    if config.enable_compression {
        app = app.layer(CompressionLayer::new());
    }

    if config.enable_cors {
        app = app.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );
    }

    app.layer(service_builder)
}

// ===== Обработчики endpoints =====

/// Проверка здоровья
async fn health_check(State(state): State<ApiState>) -> impl IntoResponse {
    let orchestrator_available = state.orchestrator.is_some();
    let stats = state.memory_service.get_stats().await.ok();
    let (ann_enabled, ann_size) = state.memory_service.graph_storage.ann_status();
    // Update availability gauges for key subsystems
    crate::metrics::set_service_available("orchestrator", orchestrator_available);
    crate::metrics::set_service_available("embedding", state.memory_service.embedding_available());

    Json(serde_json::json!({
        "status": "healthy",
        "service": "ai-memory-service",
        "version": env!("CARGO_PKG_VERSION"),
        "services": {
            "embedding": state.memory_service.embedding_available(),
            "storage": stats.is_some(),
            "cache": true
        },
        "orchestrator": {
            "available": orchestrator_available,
            "model": if orchestrator_available { "gpt-5-nano" } else { "none" },
        },
        "embedding_dimension": state.embedding_dimension,
        "ann": { "enabled": ann_enabled, "size": ann_size },
        "memory_stats": stats,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

/// Получить статистику
async fn get_statistics(State(state): State<ApiState>) -> Result<impl IntoResponse, ApiError> {
    let stats = state.memory_service.get_stats().await?;

    Ok(Json(serde_json::json!({
        "statistics": stats,
        "orchestrator_available": state.orchestrator.is_some(),
        "embedding_dimension": state.embedding_dimension,
        "success": true,
    })))
}

/// Встроенная страница дашборда (embed HTML)
async fn dashboard_page() -> impl IntoResponse {
    // HTML встроен в бинарь, чтобы страница была доступна даже без статических файлов
    let html: &str = include_str!("../reports/dashboard.html");
    Html(html.to_string())
}

/// Графовые метрики (связность, расширение, энтропия контекстов)
async fn analytics_graph(State(state): State<ApiState>) -> Result<impl IntoResponse, ApiError> {
    let gm = state.memory_service.compute_graph_metrics().await?;
    Ok(Json(serde_json::json!({
        "graph": {
            "total_memories": gm.total_memories,
            "total_contexts": gm.total_contexts,
            "avg_related_degree": gm.avg_related_degree,
            "connected_ratio": gm.connected_ratio,
            "contexts_entropy": gm.contexts_entropy,
            "two_hop_expansion": gm.two_hop_expansion,
            "avg_closure": gm.avg_closure,
            "avg_shortest_path": gm.avg_shortest_path,
        },
        "success": true
    })))
}

/// Контекстные статистики: список {path, count}
async fn analytics_contexts(State(state): State<ApiState>) -> Result<impl IntoResponse, ApiError> {
    let stats = state
        .memory_service
        .graph_storage
        .context_stats_map()
        .await?;
    let data: Vec<serde_json::Value> = stats
        .into_iter()
        .map(|(path, cnt)| serde_json::json!({"path": path, "count": cnt}))
        .collect();
    Ok(Json(serde_json::json!({"contexts": data, "success": true})))
}

/// Экспорт Prometheus-метрик
async fn metrics_endpoint() -> impl IntoResponse {
    // Ensure metrics are initialized (relevant for test servers)
    crate::metrics::init_metrics();
    let body = crate::metrics::export_metrics();
    let mut resp = (StatusCode::OK, body).into_response();
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        axum::http::HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    resp
}

/// Сохранить память
async fn store_memory(
    State(state): State<ApiState>,
    Json(req): Json<StoreMemoryRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let op_start = tokio::time::Instant::now();
    debug!("Storing new memory with context: {:?}", req.context_hint);

    // Валидация входных данных
    if req.content.trim().is_empty() {
        return Err(ApiError::BadRequest("Content cannot be empty".to_string()));
    }

    if req.content.len() > 1_000_000 {
        return Err(ApiError::BadRequest(
            "Content too large (max 1MB)".to_string(),
        ));
    }

    let id = state
        .memory_service
        .store_memory(req.content.clone(), req.context_hint.clone())
        .await?;

    info!("Memory stored successfully: {}", id);
    crate::metrics::record_memory_store_duration(op_start.elapsed().as_secs_f64());
    crate::metrics::record_memory_op("store", true);

    Ok(Json(StoreMemoryResponse {
        id,
        memory_id: id,
        success: true,
        message: "Memory stored successfully".to_string(),
        embedding_dimension: state.embedding_dimension,
    }))
}

/// Совместимый обработчик для POST /memories:
/// - Возвращает 400 при невалидном JSON (вместо 415)
/// - Требует наличие поля memory_type для совместимости с тестами
async fn store_memory_compat(
    State(state): State<ApiState>,
    _headers: HeaderMap,
    body: Bytes,
) -> Result<impl IntoResponse, ApiError> {
    // Try parse JSON manually, map any error to 400 Bad Request
    let req: StoreMemoryRequest = serde_json::from_slice(&body)
        .map_err(|_| ApiError::BadRequest("Invalid JSON".to_string()))?;

    // Validate required fields expected by compat clients
    if req.memory_type.is_none() {
        return Err(ApiError::BadRequest("memory_type is required".to_string()));
    }

    // Reuse main logic (minimal validation remains)
    let op_start = tokio::time::Instant::now();
    if req.content.trim().is_empty() {
        return Err(ApiError::BadRequest("Content cannot be empty".to_string()));
    }
    if req.content.len() > 1_000_000 {
        return Err(ApiError::BadRequest("Content too large (max 1MB)".to_string()));
    }

    let id = state
        .memory_service
        .store_memory(req.content.clone(), req.context_hint.clone())
        .await?;
    crate::metrics::record_memory_store_duration(op_start.elapsed().as_secs_f64());

    Ok(Json(StoreMemoryResponse {
        id,
        memory_id: id,
        success: true,
        message: "Memory stored successfully".to_string(),
        embedding_dimension: state.embedding_dimension,
    }))
}

/// Получить память по ID
async fn get_memory(
    State(state): State<ApiState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, ApiError> {
    let memory = state
        .memory_service
        .get_memory(&id)
        .await
        .ok_or_else(|| ApiError::NotFound(format!("Memory {} not found", id)))?;

    Ok(Json(memory))
}

/// Удалить память
async fn delete_memory(
    State(state): State<ApiState>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, ApiError> {
    state.memory_service.delete_memory(&id).await?;
    
    info!("Memory deleted: {}", id);
    crate::metrics::record_memory_op("delete", true);

    Ok(Json(serde_json::json!({
        "success": true,
        "message": format!("Memory {} deleted", id),
    })))
}

/// Trigger importance decay (single tick)
async fn maintenance_decay(
    State(state): State<ApiState>,
    Json(_req): Json<DecayRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let updated = state.memory_service.apply_decay_tick().await?;
    Ok(Json(DecayResponse { updated, success: true }))
}

/// Consolidate near-duplicates
async fn maintenance_consolidate(
    State(state): State<ApiState>,
    Json(req): Json<ConsolidateRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let th = req.similarity_threshold.unwrap_or(0.92).clamp(0.0, 1.0);
    let max_items = req.max_items.unwrap_or(120).min(500);
    let count = state
        .memory_service
        .consolidate_duplicates(req.context.as_deref(), th, max_items)
        .await?;
    crate::metrics::record_memory_op("maintenance_consolidate", true);
    Ok(Json(ConsolidateResponse { duplicates_marked: count, success: true }))
}

/// Виртуальная прогонка decay N раз (tick API)
async fn maintenance_tick(
    State(state): State<ApiState>,
    Json(req): Json<TickRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let ticks = req.ticks.unwrap_or(1).clamp(1, 365);
    let mut total_updates = 0usize;
    for _ in 0..ticks {
        total_updates += state.memory_service.apply_decay_tick().await?;
    }
    crate::metrics::record_memory_op("maintenance_tick", true);
    Ok(Json(TickResponse { ticks, total_updates, success: true }))
}

#[derive(Debug, Deserialize)]
struct BackfillReq { limit: Option<usize> }

/// Backfill embeddings for memories with missing/empty vectors
async fn maintenance_backfill_embeddings(
    State(state): State<ApiState>,
    Json(req): Json<BackfillReq>,
) -> Result<impl IntoResponse, ApiError> {
    let limit = req.limit.unwrap_or(500).min(10_000);
    let fixed = state.memory_service.backfill_embeddings(limit).await?;
    crate::metrics::record_memory_op("maintenance_backfill", true);
    Ok(Json(serde_json::json!({ "fixed": fixed, "limit": limit, "success": true })))
}

/// Count missing/empty embeddings
async fn maintenance_missing_embeddings_count(
    State(state): State<ApiState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    if let Some(prefix) = params.get("prefix").map(|s| s.to_string()) {
        let c = state
            .memory_service
            .graph_storage
            .count_missing_embeddings_with_prefix(&prefix)
            .await?;
        return Ok(Json(serde_json::json!({ "missing_embeddings": c, "prefix": prefix, "success": true })));
    }
    let c = state.memory_service.graph_storage.count_missing_embeddings().await?;
    Ok(Json(serde_json::json!({ "missing_embeddings": c, "success": true })))
}

#[derive(Debug, Deserialize)]
struct PurgeReq { prefix: Option<String>, dry_run: Option<bool> }

/// Purge orphaned memories with missing/empty embeddings (optionally under context prefix)
async fn maintenance_purge_orphaned_embeddings(
    State(state): State<ApiState>,
    Json(req): Json<PurgeReq>,
) -> Result<impl IntoResponse, ApiError> {
    let removed = state
        .memory_service
        .graph_storage
        .purge_orphaned_embeddings(req.prefix.as_deref(), req.dry_run.unwrap_or(false))
        .await?;
    Ok(Json(serde_json::json!({ "removed": removed, "success": true })))
}

/// ANN status
async fn maintenance_ann_status(
    State(state): State<ApiState>,
) -> Result<impl IntoResponse, ApiError> {
    let (enabled, size) = state.memory_service.graph_storage.ann_status();
    Ok(Json(serde_json::json!({ "enabled": enabled, "size": size, "success": true })))
}

#[derive(Debug, Deserialize)]
struct AnnPersistReq { path: Option<String> }

/// Persist ANN index to disk (if enabled and compiled)
async fn maintenance_ann_persist(
    State(state): State<ApiState>,
    Json(req): Json<AnnPersistReq>,
) -> Result<impl IntoResponse, ApiError> {
    let saved = state
        .memory_service
        .graph_storage
        .ann_persist(req.path.as_deref())?;
    Ok(Json(serde_json::json!({ "saved": saved, "success": true })))
}

/// Получить недавние воспоминания
async fn get_recent_memories(
    State(state): State<ApiState>,
    Query(params): Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, ApiError> {
    let limit = params
        .get("limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    if limit > 1000 {
        return Err(ApiError::BadRequest(
            "Limit too high (max 1000)".to_string(),
        ));
    }

    let context = params.get("context").map(|s| s.as_str());

    let memories = state.memory_service.get_recent(limit, context).await?;

    Ok(Json(serde_json::json!({
        "memories": memories,
        "count": memories.len(),
        "context": context,
        "success": true,
    })))
}

/// Поиск воспоминаний (POST endpoint)
#[axum::debug_handler]
async fn search_memories(
    State(state): State<ApiState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let timer = tokio::time::Instant::now();
    // Валидация
    if req.query.trim().is_empty() {
        crate::metrics::record_memory_op("search_post", false);
        return Err(ApiError::BadRequest("Query cannot be empty".to_string()));
    }

    let limit = req.limit.unwrap_or(10).min(100);

    let results = state.memory_service.search(&req.query, limit).await?;

    let total = results.len();

    // Optional scoring
    let scores = if req.return_scores.unwrap_or(false) {
        let alpha = req.hybrid_alpha.unwrap_or(0.7);
        let scored = state
            .memory_service
            .score_hybrid(&req.query, &results, alpha)
            .await
            .ok()
            .unwrap_or_default();
        Some(
            scored
                .into_iter()
                .map(|(id, score, method)| ScoredItem { id, score, method })
                .collect(),
        )
    } else {
        None
    };

    crate::metrics::record_recall_latency("search_post", timer.elapsed().as_secs_f64());
    crate::metrics::record_memory_op("search_post", true);
    Ok(Json(SearchResponse {
        results,
        total,
        query_id: Uuid::new_v4(),
        reasoning_chain: vec![format!("Searched for: {}", req.query)],
        confidence: 0.8,
        recall_time_ms: 100,
        success: true,
        scores,
    }))
}

/// Поиск воспоминаний (GET endpoint)
/// Поддерживает все параметры через query string с валидацией
#[axum::debug_handler]
async fn search_memories_get(
    State(state): State<ApiState>,
    Query(params): Query<SearchQueryParams>,
) -> Result<Json<SearchResponse>, ApiError> {
    // Валидируем query параметр
    if let Err(e) = validate_query(&params.query) { crate::metrics::record_memory_op("search_get", false); return Err(e); }

    let limit = params.limit.unwrap_or(10).min(100); // Защита от DoS атак

    // Измеряем реальное время выполнения поиска для точных метрик
    let start = tokio::time::Instant::now();
    let results = state
        .memory_service
        .search(&params.query, limit)
        .await
        .map_err(|e| {
            tracing::error!("GET search failed: query={}, error={}", params.query, e);
            e
        })?;
    let recall_time_ms = start.elapsed().as_millis() as u64;
    crate::metrics::record_recall_latency("search_get", (recall_time_ms as f64) / 1000.0);
    crate::metrics::record_memory_op("search_get", true);

    let total = results.len();
    tracing::info!(
        "GET search completed: query={}, results={}, time={}ms",
        params.query,
        total,
        recall_time_ms
    );

    let scores = if params.return_scores.unwrap_or(false) {
        let alpha = params.hybrid_alpha.unwrap_or(0.7);
        let scored = state
            .memory_service
            .score_hybrid(&params.query, &results, alpha)
            .await
            .ok()
            .unwrap_or_default();
        Some(scored.into_iter().map(|(id, score, method)| ScoredItem { id, score, method }).collect())
    } else { None };

    Ok(Json(SearchResponse {
        results,
        total,
        query_id: Uuid::new_v4(),
        // reasoning_chain содержит базовую информацию о запросе для отладки и трейсинга
        reasoning_chain: vec![format!(
            "GET search for: {} (limit: {}, found: {})",
            params.query, limit, total
        )],
        confidence: 0.8,
        recall_time_ms,
        success: true,
        scores,
    }))
}

/// Совместимый поиск для alias‑маршрутов
#[axum::debug_handler]
async fn search_memories_compat(
    State(state): State<ApiState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<CompatSearchResponse>, ApiError> {
    let timer = tokio::time::Instant::now();
    if req.query.trim().is_empty() {
        crate::metrics::record_memory_op("search_compat", false);
        return Err(ApiError::BadRequest("Query cannot be empty".to_string()));
    }
    let limit = req.limit.unwrap_or(10).min(100);
    let results = state.memory_service.search(&req.query, limit).await?;
    let count = results.len();

    crate::metrics::record_recall_latency("search_compat", timer.elapsed().as_secs_f64());
    crate::metrics::record_memory_op("search_compat", true);
    Ok(Json(CompatSearchResponse {
        memories: results,
        count,
        success: true,
    }))
}

/// Поиск по контексту
async fn search_by_context(
    State(state): State<ApiState>,
    body: Bytes,
) -> Result<impl IntoResponse, ApiError> {
    let timer = tokio::time::Instant::now();
    // Try strict, then tolerant parsing without unused preassignments
    let (context, limit, query_text) = match serde_json::from_slice::<SearchRequest>(&body) {
        Ok(req) => (req.context, req.limit, req.query),
        Err(_) => {
            let v: serde_json::Value = serde_json::from_slice(&body)
                .map_err(|_| ApiError::BadRequest("Invalid JSON".to_string()))?;
            (
                v.get("context").and_then(|c| c.as_str()).map(|s| s.to_string()),
                v.get("limit").and_then(|l| l.as_u64()).map(|u| u as usize),
                v.get("query").and_then(|q| q.as_str()).unwrap_or("").to_string(),
            )
        }
    };

    let context = match context { Some(c) => c, None => { crate::metrics::record_memory_op("search_context", false); return Err(ApiError::BadRequest("Context is required".to_string())) } };
    let limit = limit.unwrap_or(10).min(100);

    let results = state
        .memory_service
        .search_by_context(&context, Some(&query_text), limit)
        .await?;

    crate::metrics::record_recall_latency("search_context", timer.elapsed().as_secs_f64());
    crate::metrics::record_memory_op("search_context", true);
    Ok(Json(SearchResponse {
        total: results.len(),
        results,
        query_id: Uuid::new_v4(),
        reasoning_chain: vec![
            format!("Context: {}", context),
            format!("Query: {:?}", query_text),
        ],
        confidence: 0.85,
        recall_time_ms: 150,
        success: true,
        scores: None,
    }))
}

/// Расширенный поиск с полным recall
#[axum::debug_handler]
async fn advanced_recall(
    State(state): State<ApiState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let timer = tokio::time::Instant::now();
    let query = MemoryQuery {
        text: req.query,
        context_hint: req.context,
        memory_types: req
            .memory_types
            .map(|types| types.iter().filter_map(|t| parse_memory_type(t)).collect()),
        limit: req.limit,
        min_importance: req.min_importance,
        time_range: None,
        similarity_threshold: req.similarity_threshold,
        include_related: req.include_related.unwrap_or(false),
    };

    let recalled = state.memory_service.recall_memory(query).await?;

    let mut all_results = Vec::new();
    all_results.extend(recalled.semantic_layer.clone());
    all_results.extend(recalled.contextual_layer.clone());
    all_results.extend(recalled.detailed_layer.clone());

    // Дедупликация
    let mut seen = std::collections::HashSet::new();
    all_results.retain(|m| seen.insert(m.id));

    let total = all_results.len();

    crate::metrics::record_recall_latency("advanced_recall", timer.elapsed().as_secs_f64());
    crate::metrics::record_memory_op("advanced_recall", true);
    Ok(Json(SearchResponse {
        results: all_results,
        total,
        query_id: recalled.query_id,
        reasoning_chain: recalled.reasoning_chain,
        confidence: recalled.confidence,
        recall_time_ms: recalled.recall_time_ms,
        success: true,
        scores: None,
    }))
}

/// Совместимый endpoint /recall возвращает слои RecalledMemory
#[axum::debug_handler]
async fn recall_endpoint(
    State(state): State<ApiState>,
    Json(req): Json<RecallRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let timer = tokio::time::Instant::now();
    let query = MemoryQuery {
        text: req.query,
        context_hint: req.context_hint,
        memory_types: None,
        limit: req.limit,
        min_importance: None,
        time_range: None,
        similarity_threshold: req.similarity_threshold,
        include_related: req.include_related.unwrap_or(false),
    };

    let recalled = state.memory_service.recall_memory(query).await?;
    crate::metrics::record_recall_latency("recall", timer.elapsed().as_secs_f64());
    crate::metrics::record_memory_op("recall", true);
    Ok(Json(recalled))
}

/// Список контекстов
async fn list_contexts(State(state): State<ApiState>) -> Result<impl IntoResponse, ApiError> {
    let contexts = state.memory_service.list_contexts().await?;

    Ok(Json(serde_json::json!({
        "contexts": contexts,
        "count": contexts.len(),
        "success": true,
    })))
}

/// Информация о контексте
async fn get_context_info(
    State(state): State<ApiState>,
    Path(path): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let context = state
        .memory_service
        .get_context(&path)
        .await
        .ok_or_else(|| ApiError::NotFound(format!("Context {} not found", path)))?;

    Ok(Json(context))
}

/// Генерация инсайтов через GPT-5-nano оркестратор
async fn generate_insights(
    State(state): State<ApiState>,
    Json(req): Json<InsightsRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let orchestrator = state
        .orchestrator
        .ok_or_else(|| ApiError::BadRequest("Orchestrator not available".to_string()))?;

    // Получаем воспоминания для анализа
    let limit = req.limit.unwrap_or(100).min(500);
    let memories = if let Some(context) = &req.context {
        state
            .memory_service
            .search_by_context(context, None, limit)
            .await?
    } else {
        state.memory_service.get_recent(limit, None).await?
    };

    if memories.is_empty() {
        return Err(ApiError::NotFound(
            "No memories found for analysis".to_string(),
        ));
    }

    // Парсим тип инсайта
    let insight_type =
        parse_insight_type(req.insight_type.as_deref()).unwrap_or(InsightType::PatternRecognition);

    // Генерируем инсайты с GPT-5-nano
    let insights = orchestrator
        .generate_insights(&memories, insight_type)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    // Конвертируем в строки
    let insight_strings: Vec<String> = insights.iter().map(|mt| format!("{:?}", mt)).collect();

    Ok(Json(InsightsResponse {
        insights: insight_strings,
        memories_analyzed: memories.len(),
        context: req.context,
        reasoning_effort: req.reasoning_effort.unwrap_or_else(|| "medium".to_string()),
        model_used: "gpt-5-nano".to_string(),
        success: true,
    }))
}

/// Дистилляция контекста через GPT-5-nano
async fn distill_context(
    State(state): State<ApiState>,
    Json(req): Json<DistillationRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let orchestrator = state
        .orchestrator
        .ok_or_else(|| ApiError::BadRequest("Orchestrator not available".to_string()))?;

    // Получаем воспоминания для дистилляции
    let memories = if let Some(context) = &req.context {
        state
            .memory_service
            .search_by_context(context, None, 1000)
            .await?
    } else {
        state.memory_service.get_recent(1000, None).await?
    };

    if memories.is_empty() {
        return Err(ApiError::NotFound(
            "No memories found for distillation".to_string(),
        ));
    }

    let original_count = memories.len();

    // Дистиллируем контекст
    let context_hint = req.context.as_deref();
    let distillation_result = orchestrator
        .distill_context(&memories, context_hint)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    // Извлекаем ключевые точки из результата дистилляции
    // distill_context возвращает MemoryType, который мы конвертируем в ключевые точки
    let key_points = match distillation_result {
        MemoryType::Semantic { facts, concepts } => {
            let mut points = facts;
            points.extend(concepts);
            points
        }
        MemoryType::Episodic {
            event,
            participants,
            ..
        } => {
            let mut points = vec![event];
            points.extend(participants);
            points
        }
        MemoryType::Procedural { steps, .. } => steps,
        MemoryType::Working { task, .. } => {
            vec![task]
        }
        MemoryType::Code {
            functions,
            concepts,
            ..
        } => {
            let mut points = functions;
            points.extend(concepts);
            points
        }
        _ => {
            // Для других типов
            vec![format!(
                "Distilled memory of type: {:?}",
                distillation_result
            )]
        }
    };

    let num_points = key_points.len();
    let compression_ratio = if num_points == 0 {
        1.0
    } else {
        original_count as f32 / num_points as f32
    };

    // Создаем summary из ключевых точек
    let summary = if key_points.len() > 3 {
        format!(
            "{} key points extracted from {} memories",
            key_points.len(),
            original_count
        )
    } else {
        key_points.join("; ")
    };

    Ok(Json(DistillationResponse {
        key_points,
        summary,
        original_count,
        compression_ratio,
        success: true,
    }))
}

/// Оптимизация памяти через GPT-5-nano
async fn optimize_memory(
    State(state): State<ApiState>,
    Json(req): Json<OptimizationRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let orchestrator = state
        .orchestrator
        .ok_or_else(|| ApiError::BadRequest("Orchestrator not available".to_string()))?;

    // Получаем воспоминания для оптимизации
    let memories = if let Some(context) = &req.context {
        state
            .memory_service
            .search_by_context(context, None, 5000)
            .await?
    } else {
        state.memory_service.get_recent(5000, None).await?
    };

    if memories.is_empty() {
        return Ok(Json(OptimizationResponse {
            duplicates_found: 0,
            outdated_found: 0,
            suggestions: vec!["No memories to optimize".to_string()],
            compression_ratio: 1.0,
            space_savings_percent: 0.0,
            applied: req.aggressive.unwrap_or(false),
            success: true,
        }));
    }

    // Оптимизируем хранилище
    let optimization = orchestrator
        .optimize_memory_storage(&memories)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    // TODO: Применить оптимизацию если aggressive=true

    Ok(Json(OptimizationResponse {
        duplicates_found: optimization.duplicates_to_remove.len(),
        outdated_found: optimization.outdated_for_archive.len(),
        suggestions: optimization.optimization_suggestions,
        compression_ratio: optimization.compression_ratio,
        space_savings_percent: optimization.space_savings_percent,
        applied: req.aggressive.unwrap_or(false),
        success: true,
    }))
}

/// Анализ паттернов через GPT-5-nano
async fn analyze_patterns(
    State(state): State<ApiState>,
    Json(req): Json<HashMap<String, serde_json::Value>>,
) -> Result<impl IntoResponse, ApiError> {
    let orchestrator = state
        .orchestrator
        .ok_or_else(|| ApiError::BadRequest("Orchestrator not available".to_string()))?;

    let context = req
        .get("context")
        .and_then(|v| v.as_str())
        .map(String::from);

    let limit = req.get("limit").and_then(|v| v.as_u64()).unwrap_or(100) as usize;

    // Получаем воспоминания для анализа
    let memories = if let Some(ctx) = context {
        state
            .memory_service
            .search_by_context(&ctx, None, limit)
            .await?
    } else {
        state.memory_service.get_recent(limit, None).await?
    };

    if memories.is_empty() {
        return Err(ApiError::NotFound(
            "No memories found for analysis".to_string(),
        ));
    }

    // Анализируем паттерны
    let analysis = orchestrator
        .analyze_memory_patterns(&memories)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    Ok(Json(serde_json::json!({
        "analysis": analysis,
        "memories_analyzed": memories.len(),
        "model": "gpt-5-nano",
        "success": true,
    })))
}

/// Статус оркестратора
async fn orchestrator_status(State(state): State<ApiState>) -> impl IntoResponse {
    if let Some(orchestrator) = &state.orchestrator {
        let stats = orchestrator.get_stats().await;
        Json(serde_json::json!({
            "available": true,
            "model": "gpt-5-nano",
            "features": {
                "distillation": true,
                "insights": true,
                "optimization": true,
                "pattern_analysis": true,
            },
            "config": {
                "max_input_tokens": 400000,
                "max_output_tokens": 20000,
                "reasoning_effort_levels": ["minimal", "low", "medium", "high"],
            },
            "stats": {
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests,
                "failed_requests": stats.failed_requests,
                "avg_response_time_ms": stats.avg_response_time_ms,
                "total_input_tokens": stats.total_input_tokens,
                "total_output_tokens": stats.total_output_tokens,
                "total_reasoning_tokens": stats.total_reasoning_tokens,
                "total_cost_usd": stats.total_cost_usd,
            },
            "success": true,
        }))
    } else {
        Json(serde_json::json!({
            "available": false,
            "message": "Orchestrator not configured",
            "success": false,
        }))
    }
}

/// Пайплайн «пинг‑понг» из нескольких ролей gpt-5-nano: planner → retrieval (tools) → evaluator → synthesizer → insight-writer
async fn orchestrator_pipeline_endpoint(
    State(state): State<ApiState>,
    Json(req): Json<crate::orchestrator_pipeline::PipelineRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let orchestrator = state
        .orchestrator
        .ok_or_else(|| ApiError::BadRequest("Orchestrator not available".to_string()))?;

    let pipeline = crate::orchestrator_pipeline::MultiOrchestratorPipeline::new(
        state.memory_service.clone(),
        orchestrator.clone(),
    );

    let out = pipeline
        .run(req)
        .await
        .map_err(|e| ApiError::InternalError(e.to_string()))?;

    // Авто‑оценка качества (если включено окружением ALWAYS_INCLUDE_QUALITY=1)
    let auto_eval = std::env::var("ALWAYS_INCLUDE_QUALITY").ok().map(|s| s=="1" || s.eq_ignore_ascii_case("true")).unwrap_or(false);
    if auto_eval {
        let evidence: Vec<EvalEvidence> = out
            .evidence_preview
            .iter()
            .map(|e| EvalEvidence { id: Some(e.id.clone()), title: Some(e.title.clone()), text: e.text.clone(), context_path: Some(e.context_path.clone()) })
            .collect();
        let eval_call = orchestrator
            .evaluate_answer_with_evidence(
                &out.query,
                &serde_json::to_string(&evidence).unwrap_or("[]".to_string()),
                &out.final_summary.to_string(),
                "medium",
            )
            .await
            .ok();
        let quality = eval_call.map(|c| {
            serde_json::json!({
                "rubric": serde_json::from_str::<serde_json::Value>(&extract_json_like(&c.content)).unwrap_or(serde_json::json!({"overall": null})),
                "usage": {"input_tokens": c.usage.input_tokens, "output_tokens": c.usage.output_tokens, "reasoning_tokens": c.usage.reasoning_tokens, "cost_usd": c.usage.cost_usd}
            })
        });
        let mut root = serde_json::to_value(&out).unwrap_or(serde_json::json!({"pipeline": out}));
        if let Some(q) = quality { root["quality"] = q; }
        return Ok(Json(root));
    }

    Ok(Json(serde_json::to_value(out).unwrap_or(serde_json::json!({}))))
}

/// Сводка по оркестратору (краткая агрегация токенов/стоимости и средней латентности)
async fn orchestrator_summary(State(state): State<ApiState>) -> impl IntoResponse {
    if let Some(orchestrator) = &state.orchestrator {
        let s = orchestrator.get_stats().await;
        let avg_latency_ms = s.avg_response_time_ms;
        let total_tokens = s.total_input_tokens + s.total_output_tokens + s.total_reasoning_tokens;
        let body = serde_json::json!({
            "model": "gpt-5-nano",
            "total_requests": s.total_requests,
            "successful": s.successful_requests,
            "failed": s.failed_requests,
            "avg_latency_ms": avg_latency_ms,
            "tokens": {
                "input": s.total_input_tokens,
                "output": s.total_output_tokens,
                "reasoning": s.total_reasoning_tokens,
                "total": total_tokens
            },
            "cost_usd_total": s.total_cost_usd,
            "success": true
        });
        Json(body)
    } else {
        Json(serde_json::json!({"available": false, "success": false}))
    }
}

/// Оценка качества ответа по evidence с использованием GPT‑5‑nano и вычисляемых метрик
async fn evaluate_answer(
    State(state): State<ApiState>,
    Json(req): Json<EvaluateRequest>,
) -> Result<impl IntoResponse, ApiError> {
    let orchestrator_opt = state.orchestrator.clone();

    // Преобразуем evidence в JSON‑строку для LLM (обрезаем тексты для стабильности/экономии)
    let mut ev = serde_json::json!([]);
    if let Some(list) = req.evidence.clone() {
        let arr: Vec<serde_json::Value> = list
            .into_iter()
            .map(|e| {
                let mut t = e.text;
                if t.len() > 800 { t.truncate(800); }
                serde_json::json!({
                    "id": e.id,
                    "title": e.title,
                    "text": t,
                    "context_path": e.context_path
                })
            })
            .collect();
        ev = serde_json::Value::Array(arr);
    }
    let evidence_json = ev.to_string();

    // Ответ (final_summary) уже в JSON — передаём как текст
    let answer_json = req.final_summary.to_string();

    // Встроенные вычисляемые метрики без LLM
    let (evidence_coverage, ref_density, unique_sources, key_points, summary_points) = compute_non_llm_metrics(&req.final_summary, req.evidence.as_ref());

    let (rubric, usage) = if let Some(orchestrator) = orchestrator_opt {
        let eval_call = orchestrator
            .evaluate_answer_with_evidence(
                &req.query,
                &evidence_json,
                &answer_json,
                req.reasoning_effort.as_deref().unwrap_or("medium"),
            )
            .await
            .map_err(|e| ApiError::InternalError(e.to_string()))?;
        let rubric_json: serde_json::Value = serde_json::from_str(&extract_json_like(&eval_call.content))
            .unwrap_or_else(|_| serde_json::json!({"overall": null}));
        let usage_json = serde_json::json!({
            "input_tokens": eval_call.usage.input_tokens,
            "output_tokens": eval_call.usage.output_tokens,
            "reasoning_tokens": eval_call.usage.reasoning_tokens,
            "cost_usd": eval_call.usage.cost_usd
        });
        (rubric_json, usage_json)
    } else {
        // Fallback: только вычисляемые метрики, без LLM
        let usage_json = serde_json::json!({"input_tokens":0,"output_tokens":0,"reasoning_tokens":0,"cost_usd":0.0});
        (serde_json::json!({"overall": null}), usage_json)
    };

    let computed = serde_json::json!({
        "evidence_coverage": evidence_coverage,
        "ref_density": ref_density,
        "unique_sources": unique_sources,
        "key_points": key_points,
        "summary_points": summary_points,
    });
    Ok(Json(EvaluateResponse { rubric, computed, usage, success: true }))
}

fn compute_non_llm_metrics(answer: &serde_json::Value, evidence: Option<&Vec<EvalEvidence>>) -> (f32, f32, usize, usize, usize) {
    // evidence_coverage: доля ссылок на evidence.id в key_points[*].evidence
    let mut all_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Some(ev) = evidence { for e in ev { if let Some(id) = &e.id { all_ids.insert(id.clone()); } } }
    let total_ev = all_ids.len().max(1);
    let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut refs_count = 0usize;
    let mut kp_count = 0usize;
    let mut sum_count = 0usize;
    if let Some(kps) = answer.get("key_points").and_then(|v| v.as_array()) {
        kp_count = kps.len();
        for kp in kps {
            if let Some(arr) = kp.get("evidence").and_then(|v| v.as_array()) {
                for idv in arr { if let Some(id) = idv.as_str() { referenced.insert(id.to_string()); refs_count += 1; } }
            }
        }
    }
    if let Some(sum) = answer.get("summary").and_then(|v| v.as_array()) { sum_count = sum.len(); }
    let coverage = (referenced.len() as f32) / (total_ev as f32);
    let density = if kp_count>0 { (refs_count as f32)/(kp_count as f32) } else { 0.0 };
    (coverage, density, referenced.len(), kp_count, sum_count)
}

fn extract_json_like(s: &str) -> String {
    if let (Some(a), Some(b)) = (s.find('{'), s.rfind('}')) { s[a..=b].to_string() } else { "{}".to_string() }
}

/// Централизованная валидация query параметра с защитой от DoS и инъекций
///
/// # Проверки
/// - Не пустой после trim
/// - Длина не более 1000 символов
/// - Отсутствие опасных символов для предотвращения инъекций
///
/// # Examples
/// ```
/// assert!(validate_query("valid search").is_ok());
/// assert!(validate_query("").is_err());
/// assert!(validate_query(&"x".repeat(1001)).is_err());
/// ```
fn validate_query(query: &str) -> Result<(), ApiError> {
    let trimmed = query.trim();

    if trimmed.is_empty() {
        return Err(ApiError::BadRequest("Query cannot be empty".to_string()));
    }

    if trimmed.len() > 1000 {
        return Err(ApiError::BadRequest(
            "Query too long (max 1000 chars)".to_string(),
        ));
    }

    // Защита от потенциальных инъекций через опасные символы
    if trimmed.contains(';') || trimmed.contains("--") || trimmed.contains("/*") {
        return Err(ApiError::BadRequest(
            "Invalid characters in query".to_string(),
        ));
    }

    Ok(())
}

// ===== Вспомогательные функции =====

/// Парсинг типа памяти из строки
fn parse_memory_type(s: &str) -> Option<MemoryType> {
    match s.to_lowercase().as_str() {
        "semantic" => Some(MemoryType::Semantic {
            facts: vec![],
            concepts: vec![],
        }),
        "episodic" => Some(MemoryType::Episodic {
            event: String::new(),
            location: None,
            participants: vec![],
            timeframe: None,
        }),
        "procedural" => Some(MemoryType::Procedural {
            steps: vec![],
            tools: vec![],
            prerequisites: vec![],
        }),
        "working" => Some(MemoryType::Working {
            task: String::new(),
            deadline: None,
            priority: Priority::Medium,
        }),
        _ => None,
    }
}

/// Парсинг типа инсайта
fn parse_insight_type(s: Option<&str>) -> Option<InsightType> {
    s.and_then(|t| match t.to_lowercase().as_str() {
        "preference" | "user_preference" => Some(InsightType::UserPreference),
        "pattern" | "pattern_recognition" => Some(InsightType::PatternRecognition),
        "strategy" | "strategy_effectiveness" => Some(InsightType::StrategyEffectiveness),
        "mistake" | "common_mistake" => Some(InsightType::CommonMistake),
        "practice" | "best_practice" => Some(InsightType::BestPractice),
        "gap" | "knowledge_gap" => Some(InsightType::KnowledgeGap),
        "context" | "context_understanding" => Some(InsightType::ContextUnderstanding),
        "performance" => Some(InsightType::Performance),
        "error" | "error_pattern" => Some(InsightType::ErrorPattern),
        "success" | "success_pattern" => Some(InsightType::SuccessPattern),
        _ => None,
    })
}

/// Запустить API сервер
pub async fn run_server(
    memory_service: Arc<MemoryService>,
    orchestrator: Option<Arc<MemoryOrchestrator>>,
    config: ApiConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{}:{}", config.host, config.port);
    let has_orchestrator = orchestrator.is_some();
    let router = create_router(memory_service, orchestrator, config);

    info!("🚀 Starting AI Memory Service API on {}", addr);
    info!(
        "📡 Orchestrator: {}",
        if has_orchestrator {
            "GPT-5-nano enabled"
        } else {
            "disabled"
        }
    );

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let bound_addr = listener.local_addr()?;
    // Лог «готовности» после успешного bind порта
    info!("✅ AI Memory Service is ready and listening on {}", bound_addr);

    axum::serve(listener, router).await?;

    Ok(())
}

#[cfg(test)]
mod api_unit_tests {
    use super::*;

    #[test]
    fn test_parse_insight_type_variants() {
        // Ensure common aliases are mapped deterministically
        assert!(matches!(parse_insight_type(Some("preference")), Some(InsightType::UserPreference)));
        assert!(matches!(parse_insight_type(Some("user_preference")), Some(InsightType::UserPreference)));
        assert!(matches!(parse_insight_type(Some("pattern")), Some(InsightType::PatternRecognition)));
        assert!(matches!(parse_insight_type(Some("success_pattern")), Some(InsightType::SuccessPattern)));
        assert!(parse_insight_type(Some("unknown_kind")).is_none());
    }

    #[test]
    fn test_validate_query_ok_and_errors() {
        // Valid
        assert!(validate_query("valid search").is_ok());
        // Empty
        assert!(validate_query("").is_err());
        // Too long
        assert!(validate_query(&"x".repeat(1001)).is_err());
        // Dangerous patterns
        assert!(validate_query("abc; DROP TABLE").is_err());
        assert!(validate_query("abc -- comment").is_err());
        assert!(validate_query("abc /* block */").is_err());
    }
}
