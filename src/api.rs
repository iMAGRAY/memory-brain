//! REST API endpoints for memory service
//! 
//! Provides HTTP endpoints for storing, retrieving and managing memories.

use crate::memory::MemoryService;
use crate::types::MemoryQuery;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// API state containing shared services
#[derive(Clone)]
pub struct ApiState {
    pub memory_service: Arc<MemoryService>,
}

/// Create API router with all endpoints
pub fn create_router(state: ApiState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/memories", post(store_memory))
        .route("/memories/:id", get(get_memory))
        .route("/memories/:id", delete(delete_memory))
        .route("/recall", post(recall_memories))
        .route("/contexts", get(list_contexts))
        .route("/contexts/:path", get(get_context))
        .route("/stats", get(get_stats))
        .with_state(state)
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Store a new memory
async fn store_memory(
    State(state): State<ApiState>,
    Json(request): Json<StoreMemoryRequest>,
) -> impl IntoResponse {
    match state.memory_service.store_memory(request.content, request.context_hint).await {
        Ok(memory_id) => (StatusCode::CREATED, Json(StoreMemoryResponse { id: memory_id })).into_response(),
        Err(e) => ApiError::Internal(e.to_string()).into_response(),
    }
}

/// Get a specific memory by ID
async fn get_memory(
    State(state): State<ApiState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.memory_service.get_memory(&id).await {
        Some(memory) => (StatusCode::OK, Json(memory)).into_response(),
        None => ApiError::NotFound(format!("Memory {} not found", id)).into_response(),
    }
}

/// Delete a memory
async fn delete_memory(
    State(state): State<ApiState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.memory_service.delete_memory(&id).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => ApiError::Internal(e.to_string()).into_response(),
    }
}


/// Recall memories based on query
async fn recall_memories(
    State(_state): State<ApiState>,
    Json(_query): Json<MemoryQuery>,
) -> impl IntoResponse {
    // Temporary simple response for compilation test
    (StatusCode::NOT_IMPLEMENTED, Json(serde_json::json!({
        "error": "recall_memories not fully implemented yet"
    }))).into_response()
}

/// List all contexts
async fn list_contexts(State(state): State<ApiState>) -> impl IntoResponse {
    match state.memory_service.list_contexts().await {
        Ok(contexts) => (StatusCode::OK, Json(contexts)).into_response(),
        Err(e) => ApiError::Internal(e.to_string()).into_response(),
    }
}

/// Get context details
async fn get_context(
    State(state): State<ApiState>,
    Path(path): Path<String>,
) -> impl IntoResponse {
    match state.memory_service.get_context(&path).await {
        Some(context) => {
            let context_info = ContextInfo {
                path: context.path,
                name: context.name,
                description: context.description,
                memory_count: context.memory_count,
                activity_level: context.activity_level,
            };
            (StatusCode::OK, Json(context_info)).into_response()
        }
        None => ApiError::NotFound(format!("Context {} not found", path)).into_response(),
    }
}

/// Get service statistics
async fn get_stats(State(state): State<ApiState>) -> impl IntoResponse {
    match state.memory_service.get_stats().await {
        Ok(stats) => (StatusCode::OK, Json(stats)).into_response(),
        Err(e) => ApiError::Internal(e.to_string()).into_response(),
    }
}

// Request/Response types

#[derive(Deserialize)]
struct StoreMemoryRequest {
    content: String,
    context_hint: Option<String>,
}

#[derive(Serialize)]
struct StoreMemoryResponse {
    id: Uuid,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

#[derive(Serialize)]
struct ContextInfo {
    path: String,
    name: String,
    description: String,
    memory_count: usize,
    activity_level: f32,
}


// Error handling

#[derive(Debug)]
enum ApiError {
    NotFound(String),
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        (status, Json(ErrorResponse { error: message })).into_response()
    }
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}