//! GPT-5-nano Memory Orchestrator
//!
//! Интеллектуальный оркестратор памяти, использующий GPT-5-nano для:
//! - Автономной обработки и дистилляции больших объемов памяти
//! - Генерации инсайтов и паттернов
//! - Оптимизации хранения и поиска
//! - Кросс-сессионного обучения

use crate::types::*;
use anyhow::{anyhow, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info};
use uuid::Uuid;

/// Конфигурация GPT-5-nano оркестратора
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// OpenAI API ключ
    pub api_key: String,
    /// Модель GPT-5-nano
    pub model: String,
    /// Максимальное количество токенов в запросе
    pub max_input_tokens: usize,
    /// Максимальное количество токенов в ответе
    pub max_output_tokens: usize,
    /// Температура для креативности
    pub temperature: f32,
    /// Timeout для запросов
    pub timeout_seconds: u64,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            model: "gpt-5-nano".to_string(), // Используем GPT-5-nano как требуется
            max_input_tokens: 400000,        // GPT-5-nano поддерживает 400K входного контекста
            max_output_tokens: 20000,        // Увеличиваем для GPT-5-nano для полного вывода
            temperature: 1.0,                // GPT-5 не поддерживает изменение, только 1.0
            timeout_seconds: 120,
        }
    }
}

/// Запрос к GPT-5-nano API
#[derive(Debug, Serialize)]
struct GPTRequest {
    model: String,
    messages: Vec<GPTMessage>,
    /// For Chat Completions with GPT-5: use `max_tokens` (not `max_completion_tokens`).
    /// Validated in send_gpt_request() to ensure it doesn't exceed model limits.
    max_tokens: usize,
    // GPT-5-nano не поддерживает temperature, top_p, stop и другие параметры
    // temperature: f32,  // УБРАНО - не поддерживается в GPT-5
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>, // Используем reasoning_effort вместо temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GPTMessage {
    role: String,
    content: String,
}

/// Ответ от GPT-5-nano API
// We parse Chat Completions response as generic JSON to access usage fields consistently.

/// Основной класс GPT-5-nano Memory Orchestrator
pub struct MemoryOrchestrator {
    /// HTTP клиент для API запросов
    client: Client,
    /// Конфигурация
    config: OrchestratorConfig,
    /// Статистика работы
    stats: Arc<RwLock<OrchestratorStats>>,
}

/// Статистика работы оркестратора
#[derive(Debug, Default)]
pub struct OrchestratorStats {
    /// Общее количество запросов к GPT-5
    pub total_requests: u64,
    /// Успешные запросы
    pub successful_requests: u64,
    /// Неудачные запросы
    pub failed_requests: u64,
    /// Среднее время ответа (мс)
    pub avg_response_time_ms: f64,
    /// Сгенерированные инсайты
    pub generated_insights: u64,
    /// Дистиллированные контексты
    pub distilled_contexts: u64,
    /// Накопленные входные токены
    pub total_input_tokens: u64,
    /// Накопленные выходные токены
    pub total_output_tokens: u64,
    /// Накопленные reasoning токены (если доступны)
    pub total_reasoning_tokens: u64,
    /// Оценка совокупной стоимости (USD)
    pub total_cost_usd: f64,
}

/// Информация об использовании токенов и стоимости одного вызова
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GPTUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
    pub cost_usd: f64,
}

/// Результат JSON‑вызова с учётом usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GPTCallResult {
    pub content: String,
    pub usage: GPTUsage,
}

impl MemoryOrchestrator {
    /// Публичный тонкий враппер для нестримингового JSON‑вызова GPT‑5‑nano
    /// Используется внешними пайплайнами (planner/evaluator/synth/insight‑writer)
    pub async fn call_json(
        &self,
        operation: &str,
        system_prompt: &str,
        user_prompt: &str,
        reasoning_effort: &str,
        response_schema: Option<serde_json::Value>,
    ) -> Result<String> {
        self
            .send_gpt_request(
                operation,
                system_prompt,
                user_prompt,
                reasoning_effort,
                response_schema,
            )
            .await
    }

    /// Расширенный вызов: возвращает контент и usage; допускает per‑call override max_tokens
    pub async fn call_json_with_usage(
        &self,
        operation: &str,
        system_prompt: &str,
        user_prompt: &str,
        reasoning_effort: &str,
        response_schema: Option<serde_json::Value>,
        max_tokens_override: Option<usize>,
    ) -> Result<GPTCallResult> {
        let (resp_json, elapsed) = self
            .send_gpt_request_with_params(
                operation,
                system_prompt,
                user_prompt,
                reasoning_effort,
                response_schema,
                max_tokens_override,
            )
            .await?;

        // Извлечь текст ответа
        let content = resp_json
            .get("choices")
            .and_then(|arr| arr.as_array())
            .and_then(|arr| arr.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|msg| msg.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| anyhow!("No response content from GPT-5"))?
            .to_string();

        // Извлечь usage (если есть)
        let usage = resp_json
            .get("usage")
            .cloned()
            .unwrap_or(serde_json::json!({}));
        let input_tokens = usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let output_tokens = usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        let reasoning_tokens = usage
            .get("completion_tokens_details")
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        // Стоимость как в send_gpt_request
        let (price_in, price_out) = match self.config.model.as_str() {
            m if m.contains("gpt-5-nano") => (0.05_f64, 0.40_f64),
            m if m.contains("gpt-5-mini") => (0.25_f64, 2.00_f64),
            _ => (1.25_f64, 10.00_f64),
        };
        let cost = (input_tokens as f64 / 1_000_000f64) * price_in
            + (output_tokens as f64 / 1_000_000f64) * price_out;

        crate::metrics::record_orchestrator_latency(operation, &self.config.model, elapsed);

        Ok(GPTCallResult {
            content,
            usage: GPTUsage {
                input_tokens,
                output_tokens,
                reasoning_tokens,
                cost_usd: cost,
            },
        })
    }

    /// Внутренний метод с per‑call override max_tokens: возвращает полный JSON и длительность
    async fn send_gpt_request_with_params(
        &self,
        operation: &str,
        system_prompt: &str,
        user_prompt: &str,
        reasoning_effort: &str,
        response_schema: Option<serde_json::Value>,
        max_tokens_override: Option<usize>,
    ) -> Result<(Value, f64)> {
        let start_time = std::time::Instant::now();
        let max_tokens = max_tokens_override.unwrap_or(self.config.max_output_tokens);
        if max_tokens > 128000 {
            return Err(anyhow!(
                "per-call max_tokens {} exceeds GPT-5-nano limit of 128K",
                max_tokens
            ));
        }
        let request = GPTRequest {
            model: self.config.model.clone(),
            messages: vec![
                GPTMessage { role: "system".to_string(), content: system_prompt.to_string() },
                GPTMessage { role: "user".to_string(), content: user_prompt.to_string() },
            ],
            max_tokens,
            reasoning_effort: Some(reasoning_effort.to_string()),
            response_format: response_schema,
        };

        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
        }

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;
        let elapsed = start_time.elapsed().as_secs_f64();

        if !response.status().is_success() {
            let error_text = response.text().await?;
            {
                let mut stats = self.stats.write().await;
                stats.failed_requests += 1;
            }
            return Err(anyhow!("GPT-5 API error: {}", error_text));
        }

        let gpt_response_json: Value = response.json().await?;

        // Update global stats (tokens/cost) — тот же блок, что и в send_gpt_request
        {
            let mut stats = self.stats.write().await;
            stats.successful_requests += 1;
            let elapsed_ms = elapsed * 1000.0;
            stats.avg_response_time_ms =
                (stats.avg_response_time_ms * (stats.successful_requests - 1) as f64 + elapsed_ms)
                    / stats.successful_requests as f64;
            if let Some(usage) = gpt_response_json.get("usage") {
                let input_tokens = usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                let output_tokens = usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                let reasoning_tokens = usage
                    .get("completion_tokens_details")
                    .and_then(|d| d.get("reasoning_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                stats.total_input_tokens += input_tokens;
                stats.total_output_tokens += output_tokens;
                stats.total_reasoning_tokens += reasoning_tokens;

                let (price_in, price_out) = match self.config.model.as_str() {
                    m if m.contains("gpt-5-nano") => (0.05_f64, 0.40_f64),
                    m if m.contains("gpt-5-mini") => (0.25_f64, 2.00_f64),
                    _ => (1.25_f64, 10.00_f64),
                };
                let cost = (input_tokens as f64 / 1_000_000f64) * price_in
                    + (output_tokens as f64 / 1_000_000f64) * price_out;
                stats.total_cost_usd += cost;
                crate::metrics::add_orchestrator_tokens(&self.config.model, input_tokens, output_tokens, reasoning_tokens);
                crate::metrics::add_orchestrator_cost(&self.config.model, cost);
            }
        }

        Ok((gpt_response_json, elapsed))
    }

    /// Оценка качества ответа по evidence (использует строгую рубрику и JSON‑схему)
    pub async fn evaluate_answer_with_evidence(
        &self,
        query: &str,
        evidence_json: &str,
        answer_json: &str,
        reasoning_effort: &str,
    ) -> Result<GPTCallResult> {
        let system_prompt = "Ты — строгий научный рецензент. Оцени качество ответа по приведённым evidence. Оцени строго по рубрике и верни только JSON по схеме.";
        let user_prompt = format!(
            "ВОПРОС:\n{}\n\nEVIDENCE (JSON):\n{}\n\nОТВЕТ (JSON):\n{}\n\nПравила:\n- Оцени только на основе evidence; не добавляй фактов из памяти.\n- Дай оценки по шкале [0..1].\n- Кратко зафиксируй ключевые замечания.",
            query, evidence_json, answer_json
        );
        let schema = serde_json::json!({
            "type":"json_schema",
            "json_schema":{
                "name":"quality_evaluation",
                "schema":{
                    "type":"object",
                    "properties":{
                        "groundedness":{"type":"number","minimum":0.0,"maximum":1.0},
                        "faithfulness":{"type":"number","minimum":0.0,"maximum":1.0},
                        "completeness":{"type":"number","minimum":0.0,"maximum":1.0},
                        "actionability":{"type":"number","minimum":0.0,"maximum":1.0},
                        "clarity":{"type":"number","minimum":0.0,"maximum":1.0},
                        "structure":{"type":"number","minimum":0.0,"maximum":1.0},
                        "overall":{"type":"number","minimum":0.0,"maximum":1.0},
                        "comments":{"type":"array","items":{"type":"string"}},
                        "issues":{"type":"array","items":{"type":"string"}},
                        "improvements":{"type":"array","items":{"type":"string"}},
                        "score_explanation":{"type":"string"}
                    },
                    "required":["overall"],
                    "additionalProperties":false
                },
                "strict": true
            }
        });
        let max_tokens = std::env::var("PIPELINE_EVAL_MAX_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(900);
        self.call_json_with_usage(
            "answer_evaluate",
            system_prompt,
            &user_prompt,
            reasoning_effort,
            Some(schema),
            Some(max_tokens),
        ).await
    }
    /// Создать новый экземпляр оркестратора
    pub fn new(config: OrchestratorConfig) -> Result<Self> {
        if config.api_key.is_empty() {
            return Err(anyhow!("OpenAI API key is required"));
        }

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()?;

        Ok(Self {
            client,
            config,
            stats: Arc::new(RwLock::new(OrchestratorStats::default())),
        })
    }

    /// Дистиллировать контекст из большого объема памяти
    pub async fn distill_context(
        &self,
        memories: &[MemoryCell],
        context_hint: Option<&str>,
    ) -> Result<MemoryType> {
        info!(
            "Starting context distillation for {} memories",
            memories.len()
        );

        // Подготовить данные для дистилляции
        let context_data = self.prepare_context_data(memories)?;
        let total_size = context_data.len();

        // Создать промпт для дистилляции
        let system_prompt = self.get_distillation_system_prompt();
        let user_prompt = self.get_distillation_user_prompt(&context_data, context_hint);

        // Отправить запрос к GPT-5-nano
        // Enforce strict JSON output via response_format schema
        let distill_schema = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "distillation_output",
                "schema": {
                    "type": "object",
                    "properties": {
                        "key_points": {"type": "array", "items": {"type": "string"}},
                        "relationships": {"type": "array", "items": {"type": "string"}},
                        "actionable_insights": {"type": "array", "items": {"type": "string"}},
                        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    },
                    "required": ["key_points", "relationships", "actionable_insights", "confidence_score"],
                    "additionalProperties": false
                },
                "strict": true
            }
        });

        let response = self
            .send_gpt_request("distill", &system_prompt, &user_prompt, "medium", Some(distill_schema))
            .await?;

        // Парсинг ответа и создание ContextDistillation
        let distilled = self.parse_distillation_response(&response, total_size)?;

        // Обновить статистику
        let mut stats = self.stats.write().await;
        stats.distilled_contexts += 1;

        Ok(distilled)
    }

    /// Генерировать инсайты из паттернов памяти
    pub async fn generate_insights(
        &self,
        memories: &[MemoryCell],
        insight_type: InsightType,
    ) -> Result<Vec<MemoryType>> {
        info!(
            "Generating insights of type {:?} from {} memories",
            insight_type,
            memories.len()
        );

        // Анализировать паттерны в памяти
        let patterns = self.analyze_memory_patterns(memories).await?;

        // Создать промпт для генерации инсайтов
        let system_prompt = self.get_insight_generation_system_prompt();
        let user_prompt = self.get_insight_generation_user_prompt(&patterns, &insight_type);

        // Запрос к GPT-5-nano
        // Enforce strict JSON output for insights
        let insights_schema = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "insights_output",
                "schema": {
                    "type": "object",
                    "properties": {
                        "insights": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                    "insight": {"type": "string"},
                                    "implications": {"type": "array", "items": {"type": "string"}},
                                    "actionable_items": {"type": "array", "items": {"type": "string"}},
                                    "source_evidence": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["type", "confidence", "insight"],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": ["insights"],
                    "additionalProperties": false
                },
                "strict": true
            }
        });

        let response = self
            .send_gpt_request("insights", &system_prompt, &user_prompt, "high", Some(insights_schema))
            .await?;

        // Парсинг ответа и создание инсайтов
        let insights = self.parse_insights_response(&response, memories)?;

        // Обновить статистику
        let mut stats = self.stats.write().await;
        stats.generated_insights += insights.len() as u64;

        Ok(insights)
    }

    /// Оптимизировать память путем выявления дубликатов и неактуальных данных
    pub async fn optimize_memory_storage(
        &self,
        memories: &[MemoryCell],
    ) -> Result<MemoryOptimization> {
        info!(
            "Starting memory optimization for {} memories",
            memories.len()
        );

        // Анализ дубликатов и важности
        let analysis_data = self.analyze_memory_for_optimization(memories).await?;

        // Промпт для оптимизации
        let system_prompt = self.get_optimization_system_prompt();
        let user_prompt = self.get_optimization_user_prompt(&analysis_data);

        // Запрос к GPT-5-nano
        // Enforce strict JSON output for optimization
        let optimize_schema = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "optimization_output",
                "schema": {
                    "type": "object",
                    "properties": {
                        "duplicates_to_remove": {"type": "array", "items": {"type": "string"}},
                        "outdated_for_archive": {"type": "array", "items": {"type": "string"}},
                        "optimization_suggestions": {"type": "array", "items": {"type": "string"}},
                        "compression_ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "space_savings_percent": {"type": "number", "minimum": 0.0, "maximum": 100.0}
                    },
                    "required": ["optimization_suggestions", "compression_ratio", "space_savings_percent"],
                    "additionalProperties": false
                },
                "strict": true
            }
        });

        let response = self
            .send_gpt_request("optimize", &system_prompt, &user_prompt, "medium", Some(optimize_schema))
            .await?;

        // Парсинг рекомендаций по оптимизации
        let optimization = self.parse_optimization_response(&response, memories)?;

        Ok(optimization)
    }

    /// Отправить запрос к GPT-5-nano API
    async fn send_gpt_request(
        &self,
        operation: &str,
        system_prompt: &str,
        user_prompt: &str,
        reasoning_effort: &str,
        response_schema: Option<serde_json::Value>,
    ) -> Result<String> {
        let start_time = std::time::Instant::now();

        // Валидация параметров GPT-5-nano
        if self.config.max_output_tokens > 128000 {
            return Err(anyhow!(
                "max_output_tokens {} exceeds GPT-5-nano limit of 128K",
                self.config.max_output_tokens
            ));
        }

        let request = GPTRequest {
            model: self.config.model.clone(),
            messages: vec![
                GPTMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                GPTMessage {
                    role: "user".to_string(),
                    content: user_prompt.to_string(),
                },
            ],
            max_tokens: self.config.max_output_tokens,
            // temperature убрана - GPT-5-nano не поддерживает
            reasoning_effort: Some(reasoning_effort.to_string()),
            response_format: response_schema,
        };

        // Обновить статистику запросов
        {
            let mut stats = self.stats.write().await;
            stats.total_requests += 1;
        }

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        let elapsed = start_time.elapsed().as_secs_f64();

        if !response.status().is_success() {
            let error_text = response.text().await?;
            error!("GPT-5 API error: {}", error_text);

            let mut stats = self.stats.write().await;
            stats.failed_requests += 1;

            return Err(anyhow!("GPT-5 API error: {}", error_text));
        }

        let gpt_response_json: Value = response.json().await?;

        // Обновить статистику успешных запросов
        {
            let mut stats = self.stats.write().await;
            stats.successful_requests += 1;
            let elapsed_ms = elapsed * 1000.0;
            stats.avg_response_time_ms =
                (stats.avg_response_time_ms * (stats.successful_requests - 1) as f64 + elapsed_ms)
                    / stats.successful_requests as f64;
            // Токены и стоимость (best-effort)
            if let Some(usage) = gpt_response_json.get("usage") {
                let input_tokens = usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                let output_tokens = usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                let reasoning_tokens = usage
                    .get("completion_tokens_details")
                    .and_then(|d| d.get("reasoning_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                stats.total_input_tokens += input_tokens;
                stats.total_output_tokens += output_tokens;
                stats.total_reasoning_tokens += reasoning_tokens;

                // Simple cost estimate based on model
                let (price_in, price_out) = match self.config.model.as_str() {
                    m if m.contains("gpt-5-nano") => (0.05_f64, 0.40_f64),
                    m if m.contains("gpt-5-mini") => (0.25_f64, 2.00_f64),
                    _ => (1.25_f64, 10.00_f64),
                };
                let cost = (input_tokens as f64 / 1_000_000f64) * price_in
                    + (output_tokens as f64 / 1_000_000f64) * price_out;
                stats.total_cost_usd += cost;
                // Export to Prometheus
                crate::metrics::add_orchestrator_tokens(&self.config.model, input_tokens, output_tokens, reasoning_tokens);
                crate::metrics::add_orchestrator_cost(&self.config.model, cost);
            }
        }

        // Извлечь текст ответа
        if let Some(content) = gpt_response_json
            .get("choices")
            .and_then(|arr| arr.as_array())
            .and_then(|arr| arr.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|msg| msg.get("content"))
            .and_then(|c| c.as_str())
        {
            crate::metrics::record_orchestrator_latency(operation, &self.config.model, elapsed);
            debug!("GPT-5 response received in {:.2}s for op={}", elapsed, operation);
            Ok(content.to_string())
        } else {
            Err(anyhow!("No response content from GPT-5"))
        }
    }

    /// Получить статистику работы оркестратора
    pub async fn get_stats(&self) -> OrchestratorStats {
        let guard = self.stats.read().await;
        OrchestratorStats {
            total_requests: guard.total_requests,
            successful_requests: guard.successful_requests,
            failed_requests: guard.failed_requests,
            avg_response_time_ms: guard.avg_response_time_ms,
            generated_insights: guard.generated_insights,
            distilled_contexts: guard.distilled_contexts,
            total_input_tokens: guard.total_input_tokens,
            total_output_tokens: guard.total_output_tokens,
            total_reasoning_tokens: guard.total_reasoning_tokens,
            total_cost_usd: guard.total_cost_usd,
        }
    }

    // Приватные методы для подготовки данных и промптов

    /// Подготовить данные контекста для дистилляции
    fn prepare_context_data(&self, memories: &[MemoryCell]) -> Result<String> {
        let mut context_data = String::new();

        for (i, memory) in memories.iter().enumerate() {
            context_data.push_str(&format!(
                "ПАМЯТЬ {}: [{}] {}\nВажность: {:.2}, Контекст: {}, Теги: {}\n\n",
                i + 1,
                memory.memory_type.type_name(),
                memory.content.chars().take(500).collect::<String>(),
                memory.importance,
                memory.context_path,
                memory.tags.join(", ")
            ));
        }

        Ok(context_data)
    }

    /// Системный промпт для дистилляции контекста
    fn get_distillation_system_prompt(&self) -> String {
        "Ты - эксперт по дистилляции информации для AI Memory Service.

ЗАДАЧА: Анализируй предоставленную информацию и создай сжатый, но информативный дистиллированный контекст.

ПРИНЦИПЫ:
1. Сохраняй ключевую информацию и связи
2. Удаляй избыточность и дубликаты  
3. Выделяй паттерны и важные инсайты
4. Создавай actionable рекомендации

ФОРМАТ ОТВЕТА (JSON):
{
  \"key_points\": [\"точка1\", \"точка2\", ...],
  \"relationships\": [\"связь1\", \"связь2\", ...],
  \"actionable_insights\": [\"инсайт1\", \"инсайт2\", ...],
  \"confidence_score\": 0.85
}".to_string()
    }

    /// Пользовательский промпт для дистилляции
    fn get_distillation_user_prompt(&self, context_data: &str, hint: Option<&str>) -> String {
        let mut prompt = format!("Дистиллируй следующий контекст:\n\n{}", context_data);

        if let Some(hint) = hint {
            prompt.push_str(&format!("\n\nДополнительный контекст: {}", hint));
        }

        prompt.push_str("\n\nСоздай дистиллированный контекст в JSON формате.");
        prompt
    }

    /// Парсинг ответа дистилляции с надежным JSON parsing
    fn parse_distillation_response(
        &self,
        response: &str,
        original_size: usize,
    ) -> Result<MemoryType> {
        // Извлечь JSON из ответа (может содержать дополнительный текст)
        let json_str = self.extract_json_from_response(response)?;

        // Парсинг с помощью serde_json
        let parsed: Value = serde_json::from_str(&json_str)
            .map_err(|e| anyhow!("Failed to parse JSON response: {}", e))?;

        // Извлечение полей с валидацией
        let key_points = parsed["key_points"]
            .as_array()
            .ok_or_else(|| anyhow!("Missing or invalid 'key_points' field"))?
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect::<Vec<String>>();

        let relationships = parsed["relationships"]
            .as_array()
            .ok_or_else(|| anyhow!("Missing or invalid 'relationships' field"))?
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect::<Vec<String>>();

        let actionable_insights = parsed["actionable_insights"]
            .as_array()
            .ok_or_else(|| anyhow!("Missing or invalid 'actionable_insights' field"))?
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect::<Vec<String>>();

        let confidence_score = parsed["confidence_score"]
            .as_f64()
            .ok_or_else(|| anyhow!("Missing or invalid 'confidence_score' field"))?
            as f32;

        // Валидация значений
        if key_points.is_empty() {
            return Err(anyhow!("Distillation must contain at least one key point"));
        }

        if !(0.0..=1.0).contains(&confidence_score) {
            return Err(anyhow!("Confidence score must be between 0.0 and 1.0"));
        }

        Ok(MemoryType::ContextDistillation {
            original_context_size: original_size,
            key_points,
            relationships,
            actionable_insights,
            confidence_score,
        })
    }

    /// Извлечь JSON из ответа модели (удаляет лишний текст)
    fn extract_json_from_response(&self, response: &str) -> Result<String> {
        // Поиск первой открывающей скобки и последней закрывающей
        let json_start = response
            .find('{')
            .ok_or_else(|| anyhow!("No JSON found in response"))?;
        let json_end = response
            .rfind('}')
            .ok_or_else(|| anyhow!("Incomplete JSON in response"))?;

        if json_start >= json_end {
            return Err(anyhow!("Invalid JSON structure in response"));
        }

        Ok(response[json_start..=json_end].to_string())
    }

    // Реализация методов анализа и генерации инсайтов

    /// Анализ паттернов в памяти для выявления закономерностей
    pub async fn analyze_memory_patterns(&self, memories: &[MemoryCell]) -> Result<String> {
        debug!("Analyzing patterns in {} memories", memories.len());

        // Группировка по типам памяти
        let mut type_groups: HashMap<String, Vec<&MemoryCell>> = HashMap::new();
        for memory in memories {
            let type_name = memory.memory_type.type_name();
            type_groups
                .entry(type_name.to_string())
                .or_default()
                .push(memory);
        }

        // Анализ временных паттернов
        let time_patterns = self.analyze_temporal_patterns(memories);

        // Анализ важности и доступа
        let importance_analysis = self.analyze_importance_patterns(memories);

        // Анализ контекстных паттернов
        let context_patterns = self.analyze_context_patterns(memories);

        // Создание структурированного анализа
        let mut pattern_analysis = String::new();
        pattern_analysis.push_str("ТИПЫ ПАМЯТИ:\n");
        for (mem_type, group) in &type_groups {
            let avg_importance =
                group.iter().map(|m| m.importance).sum::<f32>() / group.len() as f32;
            pattern_analysis.push_str(&format!(
                "- {}: {} записей, средняя важность: {:.2}\n",
                mem_type,
                group.len(),
                avg_importance
            ));
        }

        pattern_analysis.push_str(&format!("\nВРЕМЕННЫЕ ПАТТЕРНЫ:\n{}\n", time_patterns));
        pattern_analysis.push_str(&format!("АНАЛИЗ ВАЖНОСТИ:\n{}\n", importance_analysis));
        pattern_analysis.push_str(&format!("КОНТЕКСТНЫЕ ПАТТЕРНЫ:\n{}", context_patterns));

        Ok(pattern_analysis)
    }

    /// Анализ временных паттернов создания и доступа к памяти
    fn analyze_temporal_patterns(&self, memories: &[MemoryCell]) -> String {
        use chrono::{Duration, Utc};

        let now = Utc::now();
        let mut recent = 0;
        let mut old = 0;
        let mut accessed_recently = 0;

        for memory in memories {
            let age = now.signed_duration_since(memory.created_at);

            if age < Duration::days(7) {
                recent += 1;
            } else if age > Duration::days(30) {
                old += 1;
            }

            let last_access_age = now.signed_duration_since(memory.last_accessed);
            if last_access_age < Duration::days(3) {
                accessed_recently += 1;
            }
        }

        format!("- Недавно созданных (< 7 дней): {}\n- Старых (> 30 дней): {}\n- Недавно использованных: {}", 
            recent, old, accessed_recently)
    }

    /// Анализ паттернов важности памяти
    fn analyze_importance_patterns(&self, memories: &[MemoryCell]) -> String {
        let mut high_importance = 0;
        let mut low_importance = 0;
        let avg_importance =
            memories.iter().map(|m| m.importance).sum::<f32>() / memories.len() as f32;

        for memory in memories {
            if memory.importance > 0.7 {
                high_importance += 1;
            } else if memory.importance < 0.3 {
                low_importance += 1;
            }
        }

        format!("- Средняя важность: {:.2}\n- Высокой важности (>0.7): {}\n- Низкой важности (<0.3): {}", 
            avg_importance, high_importance, low_importance)
    }

    /// Анализ контекстных паттернов
    fn analyze_context_patterns(&self, memories: &[MemoryCell]) -> String {
        let mut context_counts: HashMap<String, usize> = HashMap::new();

        for memory in memories {
            let parts: Vec<&str> = memory.context_path.split('/').collect();
            for (i, part) in parts.iter().enumerate() {
                let context_level = format!("level_{}: {}", i, part);
                *context_counts.entry(context_level).or_insert(0) += 1;
            }
        }

        let mut result = String::new();
        let mut sorted_contexts: Vec<_> = context_counts.iter().collect();
        sorted_contexts.sort_by(|a, b| b.1.cmp(a.1));

        for (context, count) in sorted_contexts.iter().take(5) {
            result.push_str(&format!("- {}: {} записей\n", context, count));
        }

        result
    }

    /// Системный промпт для генерации инсайтов
    fn get_insight_generation_system_prompt(&self) -> String {
        "Ты - эксперт по анализу паттернов и генерации инсайтов для AI Memory Service.

ЗАДАЧА: На основе анализа паттернов памяти генерируй actionable инсайты.

ПРИНЦИПЫ:
1. Фокусируйся на практичных выводах
2. Выявляй скрытые закономерности  
3. Предлагай улучшения процессов
4. Учитывай контекст и важность

ФОРМАТ ОТВЕТА (JSON):
{
  \"insights\": [
    {
      \"type\": \"UserPreference\",
      \"confidence\": 0.85,
      \"insight\": \"Описание инсайта\",
      \"implications\": [\"следствие1\", \"следствие2\"],
      \"actionable_items\": [\"действие1\", \"действие2\"],
      \"source_evidence\": [\"доказательство1\"]
    }
  ]
}"
        .to_string()
    }

    /// Пользовательский промпт для генерации инсайтов
    fn get_insight_generation_user_prompt(
        &self,
        patterns: &str,
        insight_type: &InsightType,
    ) -> String {
        format!(
            "Проанализируй паттерны и сгенерируй инсайты типа {:?}:

АНАЛИЗ ПАТТЕРНОВ:
{}

Создай 2-3 практичных инсайта в JSON формате.",
            insight_type, patterns
        )
    }

    /// Парсинг ответа с инсайтами
    fn parse_insights_response(
        &self,
        response: &str,
        memories: &[MemoryCell],
    ) -> Result<Vec<MemoryType>> {
        let json_str = self.extract_json_from_response(response)?;
        let parsed: Value = serde_json::from_str(&json_str)?;

        let insights_array = parsed["insights"]
            .as_array()
            .ok_or_else(|| anyhow!("Missing 'insights' array in response"))?;

        let mut memory_insights = Vec::new();

        for insight_value in insights_array {
            let insight_type_str = insight_value["type"]
                .as_str()
                .ok_or_else(|| anyhow!("Missing 'type' field in insight"))?;

            let insight_type = self.parse_insight_type(insight_type_str)?;

            let confidence = insight_value["confidence"]
                .as_f64()
                .ok_or_else(|| anyhow!("Missing 'confidence' field"))?
                as f32;

            let implications = insight_value["implications"]
                .as_array()
                .ok_or_else(|| anyhow!("Missing 'implications' field"))?
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect();

            let actionable_items = insight_value["actionable_items"]
                .as_array()
                .ok_or_else(|| anyhow!("Missing 'actionable_items' field"))?
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect();

            // Выбрать случайные source memories для демонстрации
            let source_memories = memories.iter().take(3).map(|m| m.id).collect();

            let memory_insight = MemoryType::Insight {
                insight_type,
                confidence,
                source_memories,
                implications,
                actionable_items,
            };

            memory_insights.push(memory_insight);
        }

        Ok(memory_insights)
    }

    /// Парсинг типа инсайта из строки
    fn parse_insight_type(&self, type_str: &str) -> Result<InsightType> {
        match type_str {
            "UserPreference" => Ok(InsightType::UserPreference),
            "PatternRecognition" => Ok(InsightType::PatternRecognition),
            "StrategyEffectiveness" => Ok(InsightType::StrategyEffectiveness),
            "CommonMistake" => Ok(InsightType::CommonMistake),
            "BestPractice" => Ok(InsightType::BestPractice),
            "KnowledgeGap" => Ok(InsightType::KnowledgeGap),
            "ContextUnderstanding" => Ok(InsightType::ContextUnderstanding),
            "Performance" => Ok(InsightType::Performance),
            "ErrorPattern" => Ok(InsightType::ErrorPattern),
            "SuccessPattern" => Ok(InsightType::SuccessPattern),
            "WorkflowOptimization" => Ok(InsightType::WorkflowOptimization),
            "LearningPath" => Ok(InsightType::LearningPath),
            "ToolUsage" => Ok(InsightType::ToolUsage),
            "CodingStyle" => Ok(InsightType::CodingStyle),
            _ => Err(anyhow!("Unknown insight type: {}", type_str)),
        }
    }

    /// Анализ памяти для оптимизации хранения
    async fn analyze_memory_for_optimization(&self, memories: &[MemoryCell]) -> Result<String> {
        debug!(
            "Analyzing {} memories for optimization opportunities",
            memories.len()
        );

        // Поиск потенциальных дубликатов по содержимому
        let duplicate_candidates = self.find_duplicate_candidates(memories);

        // Анализ устаревшей информации
        let outdated_analysis = self.analyze_outdated_memories(memories);

        // Анализ неиспользуемых записей
        let unused_analysis = self.analyze_unused_memories(memories);

        let analysis = format!(
            "ДУБЛИКАТЫ: {}\nУСТАРЕВШИЕ: {}\nНЕИСПОЛЬЗУЕМЫЕ: {}",
            duplicate_candidates, outdated_analysis, unused_analysis
        );

        Ok(analysis)
    }

    /// Поиск кандидатов на дубликаты
    fn find_duplicate_candidates(&self, memories: &[MemoryCell]) -> String {
        let mut candidates = 0;
        let mut content_hashes: HashMap<String, Vec<Uuid>> = HashMap::new();

        for memory in memories {
            // Простой хеш на первых 100 символах содержимого
            let content_preview = memory.content.chars().take(100).collect::<String>();
            let hash = format!(
                "{:x}",
                content_preview
                    .chars()
                    .fold(0u64, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u64))
            );

            content_hashes.entry(hash).or_default().push(memory.id);
        }

        for (_, ids) in content_hashes {
            if ids.len() > 1 {
                candidates += ids.len() - 1; // Один оставляем, остальные - кандидаты
            }
        }

        format!("{} потенциальных дубликатов", candidates)
    }

    /// Анализ устаревших записей
    fn analyze_outdated_memories(&self, memories: &[MemoryCell]) -> String {
        use chrono::{Duration, Utc};

        let now = Utc::now();
        let mut outdated = 0;

        for memory in memories {
            let age = now.signed_duration_since(memory.created_at);
            let last_access = now.signed_duration_since(memory.last_accessed);

            // Считаем устаревшими записи старше 90 дней без доступа более 30 дней
            if age > Duration::days(90)
                && last_access > Duration::days(30)
                && memory.importance < 0.5
            {
                outdated += 1;
            }
        }

        format!("{} записей для архивации", outdated)
    }

    /// Анализ неиспользуемых записей
    fn analyze_unused_memories(&self, memories: &[MemoryCell]) -> String {
        let unused = memories
            .iter()
            .filter(|m| m.access_frequency == 0 && m.importance < 0.3)
            .count();

        format!("{} неиспользуемых записей", unused)
    }

    /// Системный промпт для оптимизации
    fn get_optimization_system_prompt(&self) -> String {
        "Ты - эксперт по оптимизации хранения данных для AI Memory Service.

ЗАДАЧА: Проанализируй данные и предложи оптимизации для экономии места и повышения производительности.

ПРИНЦИПЫ:
1. Безопасное удаление дубликатов
2. Архивация неактуальных данных
3. Сжатие содержимого
4. Сохранение важной информации

ФОРМАТ ОТВЕТА (JSON):
{
  \"duplicates_to_remove\": [\"uuid1\", \"uuid2\"],
  \"outdated_for_archive\": [\"uuid3\", \"uuid4\"],
  \"optimization_suggestions\": [\"предложение1\", \"предложение2\"],
  \"compression_ratio\": 0.75,
  \"space_savings_percent\": 25.5
}".to_string()
    }

    /// Пользовательский промпт для оптимизации
    fn get_optimization_user_prompt(&self, analysis: &str) -> String {
        format!(
            "Проанализируй данные оптимизации и предложи конкретные действия:

АНАЛИЗ:
{}

Создай план оптимизации в JSON формате с конкретными рекомендациями.",
            analysis
        )
    }

    /// Парсинг ответа оптимизации
    fn parse_optimization_response(
        &self,
        response: &str,
        _memories: &[MemoryCell],
    ) -> Result<MemoryOptimization> {
        let json_str = self.extract_json_from_response(response)?;
        let parsed: Value = serde_json::from_str(&json_str)?;

        let duplicates_to_remove = parsed["duplicates_to_remove"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_str())
            .filter_map(|s| Uuid::parse_str(s).ok())
            .collect();

        let outdated_for_archive = parsed["outdated_for_archive"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_str())
            .filter_map(|s| Uuid::parse_str(s).ok())
            .collect();

        let optimization_suggestions = parsed["optimization_suggestions"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect();

        let compression_ratio = parsed["compression_ratio"].as_f64().unwrap_or(0.8) as f32;

        let space_savings_percent = parsed["space_savings_percent"].as_f64().unwrap_or(20.0) as f32;

        Ok(MemoryOptimization {
            duplicates_to_remove,
            outdated_for_archive,
            optimization_suggestions,
            compression_ratio,
            space_savings_percent,
        })
    }
}

/// Результат оптимизации памяти
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryOptimization {
    /// Дубликаты для удаления
    pub duplicates_to_remove: Vec<Uuid>,
    /// Устаревшие записи для архивации
    pub outdated_for_archive: Vec<Uuid>,
    /// Рекомендации по улучшению
    pub optimization_suggestions: Vec<String>,
    /// Общий коэффициент сжатия
    pub compression_ratio: f32,
    /// Прогнозируемая экономия места
    pub space_savings_percent: f32,
}

impl MemoryOrchestrator {
    /// Интеллектуальная подготовка промптов для EmbeddingGemma
    /// Оптимизирует промпты на основе типа памяти и контекста
    pub fn prepare_embedding_prompt(&self, memory: &MemoryCell, _context: Option<&str>) -> String {
        match &memory.memory_type {
            MemoryType::Semantic { facts, concepts } => {
                // Для семантической памяти используем поисковый промпт
                let main_fact = facts.first().unwrap_or(&String::new()).clone();
                let concept_list = concepts.join(", ");
                format!(
                    "task: search result | query: {} - concepts: {}",
                    main_fact, concept_list
                )
            }
            MemoryType::Episodic {
                event,
                participants,
                ..
            } => {
                // Для эпизодической памяти используем документный формат
                let participants_str = participants.join(", ");
                format!(
                    "title: event_{} | text: {} - participants: {}",
                    event.chars().take(50).collect::<String>(),
                    event,
                    participants_str
                )
            }
            MemoryType::Procedural { steps, tools, .. } => {
                // Для процедурной памяти используем код-специфичный промпт
                let procedure_desc = steps.join(" -> ");
                let tools_str = tools.join(", ");
                format!(
                    "task: code retrieval | query: procedure: {} using {}",
                    procedure_desc, tools_str
                )
            }
            MemoryType::Working { task, priority, .. } => {
                // Для рабочей памяти используем QA формат
                let priority_str = match priority {
                    Priority::Urgent => "urgent",
                    Priority::High => "high",
                    Priority::Medium => "medium",
                    Priority::Low => "low",
                };
                format!(
                    "task: question answering | query: {} task - priority: {}",
                    task, priority_str
                )
            }
            MemoryType::Code {
                language,
                code_type,
                functions,
                concepts,
                ..
            } => {
                // Для кода используем специализированный промпт
                let code_context = match code_type {
                    CodeType::Function => "function implementation",
                    CodeType::Class => "class definition",
                    CodeType::Configuration => "config settings",
                    CodeType::Api => "API endpoint",
                    CodeType::Database => "database schema",
                    CodeType::Test => "test code",
                    _ => "code snippet",
                };
                let functions_str = functions.join(", ");
                let concepts_str = concepts.join(", ");
                format!(
                    "task: code retrieval | query: {} {} - functions: {} concepts: {}",
                    language, code_context, functions_str, concepts_str
                )
            }
            MemoryType::Documentation {
                doc_type,
                topics,
                examples,
                ..
            } => {
                // Для документации используем документный формат с типом
                let doc_title = match doc_type {
                    DocumentationType::Api => "API Documentation",
                    DocumentationType::Tutorial => "Tutorial",
                    DocumentationType::Reference => "Reference",
                    DocumentationType::Architecture => "Architecture",
                    _ => "Documentation",
                };
                let topics_str = topics.join(", ");
                let examples_preview = examples.first().map(|s| s.as_str()).unwrap_or("");
                format!(
                    "title: {} | text: topics: {} example: {}",
                    doc_title, topics_str, examples_preview
                )
            }
            MemoryType::Conversation {
                session_id,
                intent,
                user_requests,
                ai_actions,
                ..
            } => {
                // Для разговоров используем similarity промпт
                let intent_str = match intent {
                    ConversationIntent::CodeHelp => "code assistance",
                    ConversationIntent::Debug => "debugging",
                    ConversationIntent::Explanation => "explanation",
                    ConversationIntent::FeatureImplementation => "feature implementation",
                    _ => "conversation",
                };
                let user_request = user_requests.first().map(|s| s.as_str()).unwrap_or("");
                let ai_action = ai_actions.first().map(|s| s.as_str()).unwrap_or("");
                format!(
                    "task: sentence similarity | query: {} session {} - request: {} response: {}",
                    intent_str, session_id, user_request, ai_action
                )
            }
            MemoryType::Insight {
                insight_type,
                implications,
                actionable_items,
                ..
            } => {
                // Для инсайтов используем classification промпт
                let insight_context = match insight_type {
                    InsightType::UserPreference => "user preference pattern",
                    InsightType::PatternRecognition => "recognized pattern",
                    InsightType::BestPractice => "best practice",
                    InsightType::ErrorPattern => "error pattern",
                    _ => "insight",
                };
                let main_implication = implications.first().map(|s| s.as_str()).unwrap_or("");
                let action = actionable_items.first().map(|s| s.as_str()).unwrap_or("");
                format!(
                    "task: classification | text: {} - {} action: {}",
                    insight_context, main_implication, action
                )
            }
            MemoryType::ProblemPattern {
                problem_category,
                typical_solutions,
                prevention_strategies,
                ..
            } => {
                // Для паттернов проблем используем fact checking
                let solution_preview = typical_solutions.first().map(|s| s.as_str()).unwrap_or("");
                let prevention = prevention_strategies
                    .first()
                    .map(|s| s.as_str())
                    .unwrap_or("");
                format!(
                    "task: fact checking | query: {} problem - solution: {} prevention: {}",
                    problem_category, solution_preview, prevention
                )
            }
            MemoryType::UserPreference {
                preference_type,
                examples,
                contexts,
                ..
            } => {
                // Для предпочтений используем clustering
                let example = examples.first().map(|s| s.as_str()).unwrap_or("");
                let context = contexts.first().map(|s| s.as_str()).unwrap_or("");
                format!(
                    "task: clustering | text: user preference {} - example: {} in context: {}",
                    preference_type, example, context
                )
            }
            MemoryType::ContextDistillation {
                key_points,
                relationships,
                ..
            } => {
                // Для дистиллированного контекста используем summary формат
                let points_summary = key_points.join("; ");
                let relations = relationships.join("; ");
                format!(
                    "task: search result | query: distilled context - points: {} relations: {}",
                    points_summary.chars().take(200).collect::<String>(),
                    relations.chars().take(100).collect::<String>()
                )
            }
            MemoryType::ProblemSolution {
                problem,
                solution_steps,
                prevention,
                ..
            } => {
                // Для решений проблем используем QA промпт
                let solution_preview = solution_steps.join(" -> ");
                let prevention_str = prevention.join(", ");
                format!(
                    "task: question answering | query: problem: {} solution: {} prevention: {}",
                    problem,
                    solution_preview.chars().take(150).collect::<String>(),
                    prevention_str.chars().take(100).collect::<String>()
                )
            }
            MemoryType::MetaCognitive {
                insight,
                context,
                implications,
                confidence,
                ..
            } => {
                // Для метакогнитивных инсайтов используем classification
                let implications_str = implications.join("; ");
                format!("task: classification | text: insight: {} context: {} implications: {} confidence: {:.2}", 
                    insight.chars().take(100).collect::<String>(),
                    context.chars().take(100).collect::<String>(),
                    implications_str.chars().take(100).collect::<String>(),
                    confidence)
            }
        }
    }

    /// Оптимизация batch промптов для массовой обработки
    pub fn prepare_batch_prompts(&self, memories: &[MemoryCell]) -> Vec<String> {
        memories
            .iter()
            .map(|memory| self.prepare_embedding_prompt(memory, None))
            .collect()
    }

    /// Анализ и выбор оптимального task type для набора памяти
    pub fn determine_optimal_task_type(&self, memories: &[MemoryCell]) -> String {
        let mut type_counts = HashMap::new();

        for memory in memories {
            let task_type = match &memory.memory_type {
                MemoryType::Code { .. } => "code",
                MemoryType::Documentation { .. } => "document",
                MemoryType::Conversation { .. } => "similarity",
                MemoryType::Insight { .. } => "classification",
                MemoryType::ProblemPattern { .. } => "fact",
                MemoryType::UserPreference { .. } => "clustering",
                _ => "search",
            };
            *type_counts.entry(task_type).or_insert(0) += 1;
        }

        type_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(task_type, _)| task_type.to_string())
            .unwrap_or_else(|| "search".to_string())
    }
}

impl MemoryOptimization {
    /// Валидация результатов оптимизации
    pub fn validate(&self) -> Result<()> {
        if self.compression_ratio < 0.0 || self.compression_ratio > 1.0 {
            return Err(anyhow!("Invalid compression ratio"));
        }

        if self.space_savings_percent < 0.0 || self.space_savings_percent > 100.0 {
            return Err(anyhow!("Invalid space savings percentage"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let config = OrchestratorConfig {
            api_key: "test-key".to_string(),
            ..Default::default()
        };

        let orchestrator = MemoryOrchestrator::new(config);
        assert!(orchestrator.is_ok());
    }

    #[test]
    fn test_memory_optimization_validation() {
        let mut optimization = MemoryOptimization::default();
        optimization.compression_ratio = 0.5;
        optimization.space_savings_percent = 25.0;

        assert!(optimization.validate().is_ok());

        optimization.compression_ratio = 1.5; // Невалидное значение
        assert!(optimization.validate().is_err());
    }
}
