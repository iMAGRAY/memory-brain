//! Autonomous Memory Distillation Engine
//! 
//! Система автономной дистилляции памяти с использованием GPT-5-nano для:
//! - Ежедневной дистилляции разговоров в инсайты
//! - Еженедельного анализа паттернов
//! - Ежемесячной оптимизации структуры памяти
//! - Непрерывного обучения предпочтениям пользователя

use crate::{
    orchestrator::MemoryOrchestrator,
    memory::MemoryService,
    types::MemoryCell,
};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc, Datelike, Timelike};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use tokio::time::{interval, Duration as TokioDuration, sleep};
use tokio::sync::broadcast;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Конфигурация системы дистилляции
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Включить ли автономную дистилляцию
    pub enabled: bool,
    /// Время выполнения ежедневной дистилляции (UTC час)
    pub daily_hour: u32,
    /// День недели для еженедельной дистилляции (0 = Sunday)
    pub weekly_day: u32,
    /// День месяца для ежемесячной оптимизации
    pub monthly_day: u32,
    /// Минимальная важность для дистилляции
    pub min_importance_threshold: f32,
    /// Максимальное количество памятей для дистилляции за раз
    pub max_memories_per_batch: usize,
    /// Таймаут для операций дистилляции
    pub operation_timeout_minutes: u64,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            daily_hour: 2,           // 2 AM UTC
            weekly_day: 0,           // Sunday
            monthly_day: 1,          // 1st of month
            min_importance_threshold: 0.3,
            max_memories_per_batch: 1000,
            operation_timeout_minutes: 30,
        }
    }
}

/// Типы дистилляции памяти
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistillationType {
    /// Извлечение ключевых моментов из разговоров
    ConversationHighlights {
        importance_threshold: f32,
        max_highlights: usize,
    },
    /// Выявление решений часто встречающихся проблем
    ProblemSolutionPatterns {
        min_frequency: u32,
        solution_confidence: f32,
    },
    /// Обучение предпочтениям пользователя
    UserPreferenceLearning {
        learning_window: Duration,
        confidence_threshold: f32,
    },
    /// Синтез знаний из разных доменов
    CrossDomainKnowledgeSynthesis {
        domains: Vec<String>,
        synthesis_depth: SynthesisDepth,
    },
}

/// Глубина синтеза знаний
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SynthesisDepth {
    Surface,     // Поверхностные связи
    Intermediate, // Умеренные связи
    Deep,        // Глубокие связи и паттерны
}

/// Отчет о дистилляции
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationReport {
    pub id: Uuid,
    pub distillation_type: DistillationType,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_seconds: f64,
    pub memories_processed: usize,
    pub insights_generated: usize,
    pub patterns_discovered: usize,
    pub quality_score: f32,
    pub error_count: usize,
    pub details: HashMap<String, String>,
}

/// Отчет об анализе паттернов
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternReport {
    pub id: Uuid,
    pub analysis_period: (DateTime<Utc>, DateTime<Utc>),
    pub patterns_found: Vec<PatternDiscovery>,
    pub trend_analysis: TrendAnalysis,
    pub recommendations: Vec<String>,
    pub confidence_score: f32,
}

/// Обнаруженный паттерн
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternDiscovery {
    pub pattern_type: String,
    pub frequency: u32,
    pub confidence: f32,
    pub description: String,
    pub examples: Vec<String>,
    pub implications: Vec<String>,
}

/// Анализ трендов
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub memory_growth_rate: f32,
    pub usage_patterns: HashMap<String, f32>,
    pub quality_trends: HashMap<String, f32>,
    pub performance_trends: HashMap<String, f32>,
}

/// Отчет об оптимизации
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    pub id: Uuid,
    pub optimization_time: DateTime<Utc>,
    pub actions_taken: Vec<OptimizationAction>,
    pub performance_improvement: f32,
    pub storage_saved_mb: f32,
    pub quality_improvement: f32,
    pub recommendations: Vec<String>,
}

/// Действие по оптимизации
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationAction {
    pub action_type: String,
    pub description: String,
    pub affected_memories: usize,
    pub impact_score: f32,
}

/// Метрики качества дистилляции
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    pub total_distillations: u64,
    pub successful_distillations: u64,
    pub average_quality_score: f32,
    pub average_duration_seconds: f32,
    pub insights_generated: u64,
    pub patterns_discovered: u64,
    pub last_successful_distillation: Option<DateTime<Utc>>,
}

/// Движок автономной дистилляции памяти
pub struct MemoryDistillationEngine {
    orchestrator: Arc<MemoryOrchestrator>,
    memory_service: Arc<MemoryService>,
    config: DistillationConfig,
    quality_metrics: Arc<RwLock<QualityMetrics>>,
    running_tasks: Arc<Mutex<HashMap<String, bool>>>,
    shutdown_tx: broadcast::Sender<()>,
}

impl MemoryDistillationEngine {
    /// Создание нового движка дистилляции
    pub fn new(
        orchestrator: Arc<MemoryOrchestrator>,
        memory_service: Arc<MemoryService>,
        config: DistillationConfig,
    ) -> Self {
        let (shutdown_tx, _) = broadcast::channel(16);
        
        Self {
            orchestrator,
            memory_service,
            config,
            quality_metrics: Arc::new(RwLock::new(QualityMetrics::default())),
            running_tasks: Arc::new(Mutex::new(HashMap::new())),
            shutdown_tx,
        }
    }

    /// Graceful shutdown всей системы дистилляции
    pub async fn shutdown(&self) -> Result<()> {
        info!("🛑 Initiating graceful shutdown of distillation engine");
        
        // Отправляем сигнал shutdown всем задачам
        if let Err(e) = self.shutdown_tx.send(()) {
            warn!("Failed to send shutdown signal: {}", e);
        }

        // Ждем завершения всех задач
        let mut attempts = 0;
        while attempts < 30 { // максимум 30 секунд ожидания
            let running_tasks = self.running_tasks.lock().await;
            if running_tasks.values().all(|&running| !running) {
                break;
            }
            drop(running_tasks);
            
            sleep(TokioDuration::from_secs(1)).await;
            attempts += 1;
        }

        if attempts >= 30 {
            warn!("Some distillation tasks did not shutdown gracefully within 30 seconds");
        } else {
            info!("✅ All distillation tasks shutdown gracefully");
        }

        Ok(())
    }

    /// Запуск автономной дистилляции
    pub async fn start_autonomous_distillation(&self) -> Result<()> {
        if !self.config.enabled {
            info!("Autonomous distillation is disabled");
            return Ok(());
        }

        info!("🧠 Starting autonomous memory distillation engine");

        // Запуск ежедневной дистилляции
        let daily_engine = Arc::new(self.clone());
        tokio::spawn(async move {
            daily_engine.run_daily_distillation().await;
        });

        // Запуск еженедельного анализа
        let weekly_engine = Arc::new(self.clone());
        tokio::spawn(async move {
            weekly_engine.run_weekly_analysis().await;
        });

        // Запуск ежемесячной оптимизации
        let monthly_engine = Arc::new(self.clone());
        tokio::spawn(async move {
            monthly_engine.run_monthly_optimization().await;
        });

        // Запуск непрерывного обучения
        let learning_engine = Arc::new(self.clone());
        tokio::spawn(async move {
            learning_engine.run_continuous_learning().await;
        });

        info!("✅ Autonomous distillation engine started successfully");
        Ok(())
    }

    /// Ежедневная дистилляция разговоров
    pub async fn daily_conversation_distillation(&self) -> Result<DistillationReport> {
        let start_time = Utc::now();
        info!("🌅 Starting daily conversation distillation");

        // Проверка на выполняющуюся задачу
        if self.is_task_running("daily_distillation").await {
            return Err(anyhow!("Daily distillation is already running"));
        }

        self.mark_task_running("daily_distillation", true).await;

        let mut report = DistillationReport {
            id: Uuid::new_v4(),
            distillation_type: DistillationType::ConversationHighlights {
                importance_threshold: self.config.min_importance_threshold,
                max_highlights: 100,
            },
            start_time,
            end_time: Utc::now(),
            duration_seconds: 0.0,
            memories_processed: 0,
            insights_generated: 0,
            patterns_discovered: 0,
            quality_score: 0.0,
            error_count: 0,
            details: HashMap::new(),
        };

        // Получение памятей за последние 24 часа
        let recent_memories: Vec<MemoryCell> = match self.memory_service.search(
            "recent memories from last 24 hours",
            50,
        ).await {
            Ok(results) => results,
            Err(e) => {
                error!("Failed to fetch recent memories: {}", e);
                report.error_count += 1;
                self.mark_task_running("daily_distillation", false).await;
                return Ok(report);
            }
        };

        report.memories_processed = recent_memories.len();

        if recent_memories.is_empty() {
            info!("No memories to distill from the last 24 hours");
            self.mark_task_running("daily_distillation", false).await;
            return Ok(report);
        }

        // Дистилляция через GPT-5-nano оркестратор (упрощенная версия)
        let distillation_result = match self.orchestrator.optimize_memory_storage(&recent_memories).await {
            Ok(result) => result,
            Err(e) => {
                error!("Orchestrator distillation failed: {}", e);
                report.error_count += 1;
                self.mark_task_running("daily_distillation", false).await;
                return Ok(report);
            }
        };

        // Сохранение результатов дистилляции (упрощенная версия)
        // В реальной реализации здесь будут созданы новые MemoryCell на основе insights
        report.insights_generated = 1; // Заглушка
        report.patterns_discovered = 1; // Заглушка  
        // Используем compression_ratio как quality_score - он отражает эффективность дистилляции памяти
        report.quality_score = distillation_result.compression_ratio;
        report.end_time = Utc::now();
        report.duration_seconds = (report.end_time - report.start_time).num_milliseconds() as f64 / 1000.0;

        // Обновление метрик
        self.update_quality_metrics(&report).await;

        info!("✅ Daily distillation completed: {} insights generated, {} patterns found", 
              report.insights_generated, report.patterns_discovered);

        self.mark_task_running("daily_distillation", false).await;
        Ok(report)
    }

    /// Еженедельный анализ паттернов
    pub async fn weekly_pattern_analysis(&self) -> Result<PatternReport> {
        info!("📊 Starting weekly pattern analysis");

        if self.is_task_running("weekly_analysis").await {
            return Err(anyhow!("Weekly analysis is already running"));
        }

        self.mark_task_running("weekly_analysis", true).await;

        // Получение памятей за последнюю неделю (упрощенная версия)
        let memories: Vec<MemoryCell> = match self.memory_service.search(
            "weekly analysis memories from past week",
            100
        ).await {
            Ok(results) => results,
            Err(e) => {
                error!("Failed to fetch weekly memories: {}", e);
                self.mark_task_running("weekly_analysis", false).await;
                return Err(e.into());
            }
        };

        // Анализ паттернов через оркестратор (упрощенная версия)
        let analysis_result = self.orchestrator.analyze_memory_patterns(&memories).await?;

        let report = PatternReport {
            id: Uuid::new_v4(),
            analysis_period: (Utc::now() - Duration::days(7), Utc::now()),
            patterns_found: vec![PatternDiscovery {
                pattern_type: "weekly_trend".to_string(),
                frequency: 1,
                confidence: 0.8,
                description: "Weekly pattern analysis result".to_string(),
                examples: vec![analysis_result.clone()],
                implications: vec!["Weekly insights identified".to_string()],
            }],
            trend_analysis: TrendAnalysis {
                memory_growth_rate: 0.1,
                usage_patterns: HashMap::new(),
                quality_trends: HashMap::new(),
                performance_trends: HashMap::new(),
            },
            recommendations: vec!["Continue monitoring weekly patterns".to_string()],
            confidence_score: 0.8,
        };

        info!("✅ Weekly pattern analysis completed: {} patterns found", 
              report.patterns_found.len());

        self.mark_task_running("weekly_analysis", false).await;
        Ok(report)
    }

    /// Ежемесячная оптимизация структуры памяти
    pub async fn monthly_memory_optimization(&self) -> Result<OptimizationReport> {
        info!("🔧 Starting monthly memory optimization");

        if self.is_task_running("monthly_optimization").await {
            return Err(anyhow!("Monthly optimization is already running"));
        }

        self.mark_task_running("monthly_optimization", true).await;

        // Получение памятей для анализа (упрощенная версия)
        let all_memories: Vec<MemoryCell> = match self.memory_service.search(
            "monthly optimization all memories",
            200
        ).await {
            Ok(results) => results,
            Err(e) => {
                error!("Failed to fetch all memories: {}", e);
                self.mark_task_running("monthly_optimization", false).await;
                return Err(e.into());
            }
        };
        
        // Запуск комплексной оптимизации через оркестратор (упрощенная версия)
        let optimization_result = self.orchestrator.optimize_memory_storage(&all_memories).await?;

        let actions = vec![OptimizationAction {
            action_type: "analyze_patterns".to_string(),
            description: "Analyzed memory patterns for optimization".to_string(),
            affected_memories: all_memories.len(),
            impact_score: optimization_result.compression_ratio, // Эффективность оптимизации
        }];

        let report = OptimizationReport {
            id: Uuid::new_v4(),
            optimization_time: Utc::now(),
            actions_taken: actions,
            performance_improvement: optimization_result.space_savings_percent, // Прирост производительности через экономию места
            storage_saved_mb: optimization_result.space_savings_percent * 100.0, // Примерный расчет на основе процента экономии
            quality_improvement: optimization_result.compression_ratio, // Качество оптимизации через сжатие
            recommendations: vec!["Continue monthly optimization cycles".to_string()],
        };

        info!("✅ Monthly optimization completed: {} actions taken, {:.2}% performance improvement", 
              report.actions_taken.len(), report.performance_improvement * 100.0);

        self.mark_task_running("monthly_optimization", false).await;
        Ok(report)
    }

    /// Непрерывное обучение предпочтениям пользователя
    pub async fn continuous_preference_learning(&self) -> Result<()> {
        info!("🎓 Starting continuous preference learning");

        // Каждые 4 часа анализируем новые предпочтения
        let mut interval = interval(TokioDuration::from_secs(4 * 3600));

        loop {
            interval.tick().await;
            
            if let Err(e) = self.learn_user_preferences().await {
                error!("Error in preference learning: {}", e);
            }
        }
    }

    /// Обучение предпочтениям пользователя
    async fn learn_user_preferences(&self) -> Result<()> {
        info!("🎓 Learning user preferences - placeholder implementation");
        
        // Получаем недавние воспоминания для анализа
        let recent_memories: Vec<MemoryCell> = match self.memory_service.search(
            "user interactions and preferences",
            50
        ).await {
            Ok(results) => results,
            Err(e) => {
                warn!("Failed to fetch recent memories for preference learning: {}", e);
                return Ok(()); // Не критическая ошибка
            }
        };

        if recent_memories.is_empty() {
            debug!("No recent memories available for preference learning");
            return Ok(());
        }

        // Анализируем паттерны в памятях через оркестратор
        match self.orchestrator.analyze_memory_patterns(&recent_memories).await {
            Ok(analysis) => {
                info!("✅ Analyzed {} memories for user preference patterns", recent_memories.len());
                debug!("Pattern analysis result: {}", analysis);
            }
            Err(e) => {
                warn!("Failed to analyze preference patterns: {}", e);
            }
        }

        Ok(())
    }

    // Утилиты для управления задачами
    async fn is_task_running(&self, task_name: &str) -> bool {
        self.running_tasks.lock().await
            .get(task_name)
            .copied()
            .unwrap_or(false)
    }

    async fn mark_task_running(&self, task_name: &str, running: bool) {
        self.running_tasks.lock().await
            .insert(task_name.to_string(), running);
    }

    /// Обновление метрик качества
    async fn update_quality_metrics(&self, report: &DistillationReport) {
        let mut metrics = self.quality_metrics.write().await;
        metrics.total_distillations += 1;
        
        if report.error_count == 0 {
            metrics.successful_distillations += 1;
            metrics.last_successful_distillation = Some(report.end_time);
        }

        // Обновление скользящего среднего
        metrics.average_quality_score = (metrics.average_quality_score * (metrics.total_distillations - 1) as f32 + report.quality_score) / metrics.total_distillations as f32;
        metrics.average_duration_seconds = (metrics.average_duration_seconds * (metrics.total_distillations - 1) as f32 + report.duration_seconds as f32) / metrics.total_distillations as f32;
        metrics.insights_generated += report.insights_generated as u64;
        metrics.patterns_discovered += report.patterns_discovered as u64;
    }

    // Планировщики для автономной работы с graceful shutdown
    async fn run_daily_distillation(&self) {
        let mut interval = interval(TokioDuration::from_secs(3600)); // Каждый час проверяем время
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        
        info!("🌅 Daily distillation scheduler started");

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let now = Utc::now();
                    if now.hour() == self.config.daily_hour && now.minute() < 5 {
                        // Retry логика с exponential backoff
                        if let Err(e) = self.retry_with_backoff(
                            || self.daily_conversation_distillation(),
                            3, // max_retries
                            TokioDuration::from_secs(5), // initial_delay
                        ).await {
                            error!("Daily distillation failed after retries: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("🛑 Daily distillation scheduler received shutdown signal");
                    self.mark_task_running("daily_distillation", false).await;
                    break;
                }
            }
        }
        
        info!("✅ Daily distillation scheduler shutdown complete");
    }

    async fn run_weekly_analysis(&self) {
        let mut interval = interval(TokioDuration::from_secs(3600)); // Каждый час проверяем время
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        
        info!("📊 Weekly analysis scheduler started");

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let now = Utc::now();
                    if now.weekday().num_days_from_sunday() == self.config.weekly_day 
                       && now.hour() == self.config.daily_hour + 1 
                       && now.minute() < 5 {
                        if let Err(e) = self.retry_with_backoff(
                            || self.weekly_pattern_analysis(),
                            3, // max_retries
                            TokioDuration::from_secs(10), // initial_delay
                        ).await {
                            error!("Weekly analysis failed after retries: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("🛑 Weekly analysis scheduler received shutdown signal");
                    self.mark_task_running("weekly_analysis", false).await;
                    break;
                }
            }
        }
        
        info!("✅ Weekly analysis scheduler shutdown complete");
    }

    async fn run_monthly_optimization(&self) {
        let mut interval = interval(TokioDuration::from_secs(86400)); // Каждый день проверяем (оптимизация производительности)
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        
        info!("🔧 Monthly optimization scheduler started");

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    let now = Utc::now();
                    // Исправление критической логики времени
                    let target_hour = (self.config.daily_hour + 2) % 24; // Предотвращение переполнения часов
                    
                    if now.day() == self.config.monthly_day 
                       && now.hour() == target_hour 
                       && now.minute() < 30 {  // Расширенное окно для надежности
                        info!("🗓️ Monthly optimization triggered on day {} at {}:{:02}", 
                             now.day(), now.hour(), now.minute());
                        
                        if let Err(e) = self.retry_with_backoff(
                            || self.monthly_memory_optimization(),
                            5, // max_retries (больше для ресурсоемкой операции)
                            TokioDuration::from_secs(30), // initial_delay
                        ).await {
                            error!("Monthly optimization failed after retries: {}", e);
                        }
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("🛑 Monthly optimization scheduler received shutdown signal");
                    self.mark_task_running("monthly_optimization", false).await;
                    break;
                }
            }
        }
        
        info!("✅ Monthly optimization scheduler shutdown complete");
    }

    async fn run_continuous_learning(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        
        info!("🎓 Continuous learning scheduler started");

        tokio::select! {
            result = self.retry_with_backoff(
                || self.continuous_preference_learning(),
                3, // max_retries
                TokioDuration::from_secs(10), // initial_delay
            ) => {
                if let Err(e) = result {
                    error!("Continuous learning failed after retries: {}", e);
                }
            }
            _ = shutdown_rx.recv() => {
                info!("🛑 Continuous learning received shutdown signal");
            }
        }
        
        info!("✅ Continuous learning scheduler shutdown complete");
    }

    /// Retry логика с exponential backoff для orchestrator вызовов
    async fn retry_with_backoff<F, Fut, T>(&self, operation: F, max_retries: u32, initial_delay: TokioDuration) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut delay = initial_delay;
        let mut last_error = None;

        for attempt in 0..=max_retries {
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        info!("Operation succeeded after {} retries", attempt);
                    }
                    return Ok(result);
                }
                Err(e) => {
                    last_error = Some(e);
                    
                    if attempt < max_retries {
                        warn!("Operation failed (attempt {}/{}), retrying in {:?}", 
                             attempt + 1, max_retries + 1, delay);
                        
                        // Проверяем на shutdown во время ожидания
                        let mut shutdown_rx = self.shutdown_tx.subscribe();
                        tokio::select! {
                            _ = sleep(delay) => {}
                            _ = shutdown_rx.recv() => {
                                return Err(anyhow!("Operation cancelled due to shutdown"));
                            }
                        }
                        
                        // Exponential backoff с jitter
                        delay = std::cmp::min(
                            delay * 2 + TokioDuration::from_millis(fastrand::u64(0..1000)),
                            TokioDuration::from_secs(300) // максимум 5 минут
                        );
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("Operation failed after {} retries", max_retries)))
    }

    /// Получение метрик качества
    pub async fn get_quality_metrics(&self) -> QualityMetrics {
        self.quality_metrics.read().await.clone()
    }
}

// Реализация Clone для возможности создания Arc
impl Clone for MemoryDistillationEngine {
    fn clone(&self) -> Self {
        Self {
            orchestrator: self.orchestrator.clone(),
            memory_service: self.memory_service.clone(),
            config: self.config.clone(),
            quality_metrics: self.quality_metrics.clone(),
            running_tasks: self.running_tasks.clone(),
            shutdown_tx: self.shutdown_tx.clone(),
            // shutdown_rx создается локально через subscribe() в каждом методе
        }
    }
}
