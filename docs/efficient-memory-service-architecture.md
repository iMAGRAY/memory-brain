# Эффективный архитектурный план локальной эмбеддинг памяти для ИИ агентов

## 1. Общая архитектура системы

### 1.1 Упрощенная многослойная архитектура

```
┌─────────────────────────────────────────────────────────┐
│                    Interface Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  MCP Server  │  │  REST API    │  │  Direct SDK  │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│                Memory Service Core                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Memory Brain │  │ Context      │  │ Query        │   │
│  │ (GPT-5-nano) │  │ Manager      │  │ Processor    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│                Memory Layers                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Semantic     │  │ Contextual   │  │ Long-term    │   │
│  │ (Tags/Facts) │  │ (Relations)  │  │ (Details)    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│                Storage Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Neo4j      │  │  ONNX        │  │ Simple       │   │
│  │   Graph      │  │ Gemma-300m   │  │ Cache        │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## 2. Основные структуры данных

### 2.1 Память как человеческая система

```rust
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

// Основная ячейка памяти
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCell {
    pub id: Uuid,
    pub content: String,
    pub summary: String,
    pub tags: Vec<String>,
    pub embedding: Vec<f32>,
    pub memory_type: MemoryType,
    pub importance: f32,        // 0.0-1.0
    pub access_frequency: u32,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub context_path: String,   // "work/projects/rust_memory"
}

// Типы памяти как у человека
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryType {
    Semantic {      // Семантическая (факты, теги)
        facts: Vec<String>,
        concepts: Vec<String>,
    },
    Episodic {      // Эпизодическая (события, опыт)
        event: String,
        location: Option<String>,
        participants: Vec<String>,
    },
    Procedural {    // Процедурная (как делать)
        steps: Vec<String>,
        tools: Vec<String>,
    },
    Working {       // Рабочая (временная)
        task: String,
        deadline: Option<DateTime<Utc>>,
    },
}

// Контекст как в человеческом мозге
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    pub path: String,           // "work/projects/rust_memory"
    pub name: String,           // "Rust Memory Service"
    pub description: String,
    pub parent: Option<String>, // "work/projects"
    pub children: Vec<String>,  // ["rust_memory/architecture", "rust_memory/implementation"]
    pub embedding: Vec<f32>,
    pub activity_level: f32,    // Как часто используется
    pub memory_count: usize,
}

// Запрос памяти
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQuery {
    pub text: String,
    pub context_hint: Option<String>,
    pub memory_types: Option<Vec<MemoryType>>,
    pub limit: Option<usize>,
    pub min_importance: Option<f32>,
}

// Восстановленный контекст
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecalledMemory {
    pub query_id: Uuid,
    pub semantic_layer: Vec<MemoryCell>,    // Быстрые факты/теги
    pub contextual_layer: Vec<MemoryCell>,  // Связанные воспоминания
    pub detailed_layer: Vec<MemoryCell>,    // Подробная информация
    pub reasoning_chain: Vec<String>,       // Как пришли к результату
    pub confidence: f32,
    pub recall_time_ms: u64,
}
```

### 2.2 Конфигурация системы

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub neo4j: Neo4jConfig,
    pub onnx: OnnxConfig,
    pub gpt5_nano: Gpt5NanoConfig,
    pub cache: CacheConfig,
    pub memory: MemorySettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySettings {
    pub max_semantic_cache: usize,     // 1000
    pub max_contextual_cache: usize,   // 500
    pub max_detailed_cache: usize,     // 100
    pub importance_decay_days: f32,    // 30.0
    pub access_boost_factor: f32,      // 1.1
    pub context_depth_limit: u8,       // 3
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neo4jConfig {
    pub uri: String,
    pub user: String,
    pub password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxConfig {
    pub model_path: String,
    pub threads: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gpt5NanoConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,      // "gpt-5-nano"
    pub max_tokens: u32,    // 1000
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub ttl_seconds: u64,   // 3600
    pub max_size: usize,    // 10000
}
```

## 3. Core Memory Service

### 3.1 Главный сервис памяти

```rust
use tokio::sync::RwLock;
use std::sync::Arc;
use anyhow::Result;

pub struct MemoryService {
    embedding_engine: Arc<EmbeddingEngine>,
    graph_storage: Arc<GraphStorage>,
    memory_brain: Arc<MemoryBrain>,
    cache: Arc<MemoryCache>,
    config: MemoryConfig,
}

impl MemoryService {
    pub async fn new(config: MemoryConfig) -> Result<Self> {
        let embedding_engine = Arc::new(EmbeddingEngine::new(&config.onnx).await?);
        let graph_storage = Arc::new(GraphStorage::new(&config.neo4j).await?);
        let memory_brain = Arc::new(MemoryBrain::new(&config.gpt5_nano));
        let cache = Arc::new(MemoryCache::new(&config.cache));

        Ok(Self {
            embedding_engine,
            graph_storage,
            memory_brain,
            cache,
            config,
        })
    }

    // Сохранение памяти
    pub async fn store_memory(&self, content: String, context_path: String) -> Result<Uuid> {
        // 1. Создание эмбеддинга
        let embedding = self.embedding_engine.embed(&content).await?;
        
        // 2. Анализ контента GPT-5-nano
        let analysis = self.memory_brain.analyze_content(&content).await?;
        
        // 3. Создание ячейки памяти
        let memory_cell = MemoryCell {
            id: Uuid::new_v4(),
            content: content.clone(),
            summary: analysis.summary,
            tags: analysis.tags,
            embedding,
            memory_type: analysis.memory_type,
            importance: analysis.importance,
            access_frequency: 0,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            context_path: context_path.clone(),
        };

        // 4. Сохранение в граф
        self.graph_storage.store_memory(&memory_cell, &context_path).await?;
        
        // 5. Обновление индексов
        self.update_context_indices(&context_path).await?;

        Ok(memory_cell.id)
    }

    // Восстановление памяти (как человеческий мозг)
    pub async fn recall_memory(&self, query: MemoryQuery) -> Result<RecalledMemory> {
        let start_time = std::time::Instant::now();
        let query_id = Uuid::new_v4();

        // 1. Семантический поиск (быстрая память - теги, факты)
        let semantic_results = self.semantic_recall(&query).await?;
        
        // 2. Контекстуальный поиск (средняя память - связи)
        let contextual_results = self.contextual_recall(&query, &semantic_results).await?;
        
        // 3. Подробный поиск (долговременная память - детали)
        let detailed_results = self.detailed_recall(&query, &contextual_results).await?;
        
        // 4. GPT-5-nano обрабатывает и структурирует результаты
        let processed_memory = self.memory_brain.process_recall(
            &query,
            &semantic_results,
            &contextual_results,
            &detailed_results,
        ).await?;

        Ok(RecalledMemory {
            query_id,
            semantic_layer: processed_memory.semantic,
            contextual_layer: processed_memory.contextual,
            detailed_layer: processed_memory.detailed,
            reasoning_chain: processed_memory.reasoning,
            confidence: processed_memory.confidence,
            recall_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    // Семантический поиск (мгновенный доступ к тегам/фактам)
    async fn semantic_recall(&self, query: &MemoryQuery) -> Result<Vec<MemoryCell>> {
        // Проверка кэша
        if let Some(cached) = self.cache.get_semantic(&query.text).await {
            return Ok(cached);
        }

        // Быстрый поиск по тегам и эмбеддингам
        let query_embedding = self.embedding_engine.embed(&query.text).await?;
        let results = self.graph_storage.semantic_search(&query_embedding, 10).await?;

        // Кэшируем результат
        self.cache.store_semantic(&query.text, &results).await;

        Ok(results)
    }

    // Контекстуальный поиск (связанные воспоминания)
    async fn contextual_recall(&self, query: &MemoryQuery, semantic: &[MemoryCell]) -> Result<Vec<MemoryCell>> {
        let mut contextual_memories = Vec::new();

        for memory in semantic {
            // Поиск связанных воспоминаний через граф
            let related = self.graph_storage.find_related(&memory.id, 2).await?;
            contextual_memories.extend(related);
        }

        // Удаление дубликатов и сортировка по релевантности
        contextual_memories.sort_by(|a, b| {
            b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal)
        });
        contextual_memories.dedup_by_key(|m| m.id);

        Ok(contextual_memories.into_iter().take(20).collect())
    }

    // Подробный поиск (полная информация)
    async fn detailed_recall(&self, query: &MemoryQuery, contextual: &[MemoryCell]) -> Result<Vec<MemoryCell>> {
        let mut detailed_memories = Vec::new();

        // Для каждого контекстуального воспоминания ищем детали
        for memory in contextual {
            if memory.importance > 0.7 { // Только важные воспоминания
                let details = self.graph_storage.get_memory_details(&memory.id).await?;
                detailed_memories.extend(details);
            }
        }

        Ok(detailed_memories.into_iter().take(10).collect())
    }

    // Обновление индексов контекста
    async fn update_context_indices(&self, context_path: &str) -> Result<()> {
        // Пересчет активности контекста
        let memory_count = self.graph_storage.count_memories_in_context(context_path).await?;
        let activity_level = self.calculate_context_activity(context_path).await?;

        self.graph_storage.update_context_stats(context_path, memory_count, activity_level).await?;
        
        Ok(())
    }

    async fn calculate_context_activity(&self, _context_path: &str) -> Result<f32> {
        // Упрощенный расчет активности
        Ok(0.5)
    }
}
```

### 3.2 Memory Brain (GPT-5-nano интеграция)

```rust
use reqwest::Client;
use serde_json::json;

pub struct MemoryBrain {
    client: Client,
    config: Gpt5NanoConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ContentAnalysis {
    pub summary: String,
    pub tags: Vec<String>,
    pub memory_type: MemoryType,
    pub importance: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessedRecall {
    pub semantic: Vec<MemoryCell>,
    pub contextual: Vec<MemoryCell>,
    pub detailed: Vec<MemoryCell>,
    pub reasoning: Vec<String>,
    pub confidence: f32,
}

impl MemoryBrain {
    pub fn new(config: &Gpt5NanoConfig) -> Self {
        Self {
            client: Client::new(),
            config: config.clone(),
        }
    }

    // Анализ контента для сохранения
    pub async fn analyze_content(&self, content: &str) -> Result<ContentAnalysis> {
        let prompt = format!(
            "Проанализируй следующий текст как память для ИИ агента. Верни JSON:
            
Текст: {}

Структура ответа:
{{
    \"summary\": \"краткое резюме в 1-2 предложения\",
    \"tags\": [\"тег1\", \"тег2\", \"тег3\"],
    \"memory_type\": \"Semantic\" | \"Episodic\" | \"Procedural\" | \"Working\",
    \"importance\": 0.8
}}

Важность от 0.0 до 1.0 где 1.0 - критически важная информация.", content);

        let response = self.call_gpt5_nano(&prompt).await?;
        let analysis: ContentAnalysis = serde_json::from_str(&response)?;
        
        Ok(analysis)
    }

    // Обработка результатов поиска
    pub async fn process_recall(
        &self,
        query: &MemoryQuery,
        semantic: &[MemoryCell],
        contextual: &[MemoryCell],
        detailed: &[MemoryCell],
    ) -> Result<ProcessedRecall> {
        let prompt = format!(
            "Как мозг человека, обработай результаты поиска памяти для запроса: '{}'

Семантические воспоминания (быстрые факты): {}
Контекстуальные воспоминания (связи): {}
Детальные воспоминания (подробности): {}

Верни JSON:
{{
    \"semantic\": [отфильтрованные и ранжированные семантические],
    \"contextual\": [наиболее релевантные контекстуальные],
    \"detailed\": [самые важные детали],
    \"reasoning\": [\"шаг 1\", \"шаг 2\", \"шаг 3\"],
    \"confidence\": 0.85
}}

Рассуждай как человеческий мозг: сначала быстрые ассоциации, затем контекст, потом детали.",
            query.text,
            self.serialize_memories(semantic),
            self.serialize_memories(contextual),
            self.serialize_memories(detailed)
        );

        let response = self.call_gpt5_nano(&prompt).await?;
        let processed: ProcessedRecall = serde_json::from_str(&response)?;
        
        Ok(processed)
    }

    // Структурирование и обновление памяти
    pub async fn restructure_memory(&self, context_path: &str, memories: &[MemoryCell]) -> Result<Vec<String>> {
        let prompt = format!(
            "Как мозг во сне, реструктурируй воспоминания в контексте '{}':

Воспоминания: {}

Предложи улучшения:
{{
    \"improvements\": [
        \"объединить схожие воспоминания X и Y\",
        \"создать новые связи между A и B\",
        \"повысить важность воспоминания Z\"
    ]
}}",
            context_path,
            self.serialize_memories(memories)
        );

        let response = self.call_gpt5_nano(&prompt).await?;
        let improvements: serde_json::Value = serde_json::from_str(&response)?;
        
        Ok(improvements["improvements"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect())
    }

    async fn call_gpt5_nano(&self, prompt: &str) -> Result<String> {
        let payload = json!({
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты - мозг ИИ агента, отвечающий за память. Отвечай только в JSON формате, без дополнительного текста."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": 0.3
        });

        let response = self.client
            .post(&format!("{}/chat/completions", self.config.base_url))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&payload)
            .send()
            .await?;

        let response_json: serde_json::Value = response.json().await?;
        
        Ok(response_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("{}")
            .to_string())
    }

    fn serialize_memories(&self, memories: &[MemoryCell]) -> String {
        memories.iter()
            .take(5)
            .map(|m| format!("{}: {} (теги: {})", m.id, m.summary, m.tags.join(", ")))
            .collect::<Vec<_>>()
            .join("\\n")
    }
}
```

### 3.3 ONNX Embedding Engine

```rust
use ort::{Environment, Session, Value};
use tokenizers::Tokenizer;

pub struct EmbeddingEngine {
    session: Session,
    tokenizer: Tokenizer,
}

impl EmbeddingEngine {
    pub async fn new(config: &OnnxConfig) -> Result<Self> {
        let environment = Environment::builder()
            .with_name("embedding")
            .build()?;

        let session = Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::All)?
            .with_intra_threads(config.threads.unwrap_or(4))?
            .commit_from_file(&config.model_path)?;

        // Загрузка токенизатора Gemma
        let tokenizer = Tokenizer::from_file("tokenizer.json")?;

        Ok(Self {
            session,
            tokenizer,
        })
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Токенизация
        let encoding = self.tokenizer.encode(text, false)?;
        let tokens = encoding.get_ids();
        let attention_mask: Vec<i64> = vec![1; tokens.len()];

        // Подготовка входных данных
        let input_ids = Value::from_array(([1, tokens.len()], tokens))?;
        let attention_mask = Value::from_array(([1, attention_mask.len()], attention_mask))?;

        // Инференс
        let outputs = self.session.run([
            ("input_ids".into(), input_ids),
            ("attention_mask".into(), attention_mask),
        ])?;

        // Извлечение эмбеддингов
        let embeddings = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()?;

        // Mean pooling
        let embedding_dim = 768;
        let seq_len = embeddings.shape()[1];
        let mut pooled = vec![0.0f32; embedding_dim];

        for i in 0..seq_len {
            for j in 0..embedding_dim {
                pooled[j] += embeddings[[0, i, j]];
            }
        }

        // Нормализация
        for val in pooled.iter_mut() {
            *val /= seq_len as f32;
        }

        Ok(pooled)
    }
}
```

### 3.4 Neo4j Graph Storage

```rust
use neo4rs::{Graph, Query, Node};

pub struct GraphStorage {
    graph: Arc<Graph>,
}

impl GraphStorage {
    pub async fn new(config: &Neo4jConfig) -> Result<Self> {
        let graph = Graph::new(&config.uri, &config.user, &config.password).await?;
        
        // Создание индексов
        let index_query = "
            CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.id);
            CREATE INDEX IF NOT EXISTS FOR (c:Context) ON (c.path);
            CREATE VECTOR INDEX IF NOT EXISTS memory_embeddings
            FOR (m:Memory) ON (m.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }};
        ";
        
        graph.run(Query::new(index_query)).await?;

        Ok(Self {
            graph: Arc::new(graph),
        })
    }

    pub async fn store_memory(&self, memory: &MemoryCell, context_path: &str) -> Result<()> {
        let query = "
            MERGE (c:Context {path: $context_path})
            CREATE (m:Memory {
                id: $id,
                content: $content,
                summary: $summary,
                tags: $tags,
                embedding: $embedding,
                memory_type: $memory_type,
                importance: $importance,
                access_frequency: $access_frequency,
                created_at: $created_at,
                last_accessed: $last_accessed
            })
            CREATE (m)-[:BELONGS_TO]->(c)
        ";

        self.graph.run(Query::new(query)
            .param("context_path", context_path)
            .param("id", memory.id.to_string())
            .param("content", &memory.content)
            .param("summary", &memory.summary)
            .param("tags", &memory.tags)
            .param("embedding", &memory.embedding)
            .param("memory_type", format!("{:?}", memory.memory_type))
            .param("importance", memory.importance)
            .param("access_frequency", memory.access_frequency as i64)
            .param("created_at", memory.created_at.to_rfc3339())
            .param("last_accessed", memory.last_accessed.to_rfc3339())
        ).await?;

        Ok(())
    }

    pub async fn semantic_search(&self, embedding: &[f32], limit: usize) -> Result<Vec<MemoryCell>> {
        let query = "
            CALL db.index.vector.queryNodes('memory_embeddings', $limit, $embedding)
            YIELD node, score
            RETURN node
            ORDER BY score DESC
        ";

        let mut result = self.graph.execute(Query::new(query)
            .param("limit", limit)
            .param("embedding", embedding)
        ).await?;

        let mut memories = Vec::new();
        while let Some(row) = result.next().await? {
            let node: Node = row.get("node")?;
            memories.push(self.node_to_memory(node)?);
        }

        Ok(memories)
    }

    pub async fn find_related(&self, memory_id: &Uuid, depth: u8) -> Result<Vec<MemoryCell>> {
        let query = format!("
            MATCH (m:Memory {{id: $memory_id}})
            CALL apoc.path.subgraphNodes(m, {{
                relationshipFilter: 'RELATES_TO|BELONGS_TO|SIMILAR_TO',
                maxLevel: {}
            }})
            YIELD node
            WHERE node.id <> $memory_id
            RETURN node
            ORDER BY node.importance DESC
        ", depth);

        let mut result = self.graph.execute(Query::new(&query)
            .param("memory_id", memory_id.to_string())
        ).await?;

        let mut memories = Vec::new();
        while let Some(row) = result.next().await? {
            let node: Node = row.get("node")?;
            if let Ok(memory) = self.node_to_memory(node) {
                memories.push(memory);
            }
        }

        Ok(memories)
    }

    pub async fn count_memories_in_context(&self, context_path: &str) -> Result<usize> {
        let query = "
            MATCH (c:Context {path: $context_path})<-[:BELONGS_TO]-(m:Memory)
            RETURN count(m) as count
        ";

        let mut result = self.graph.execute(Query::new(query)
            .param("context_path", context_path)
        ).await?;

        if let Some(row) = result.next().await? {
            Ok(row.get::<i64>("count")? as usize)
        } else {
            Ok(0)
        }
    }

    pub async fn get_memory_details(&self, memory_id: &Uuid) -> Result<Vec<MemoryCell>> {
        // Возвращаем связанные детали для этого воспоминания
        self.find_related(memory_id, 1).await
    }

    pub async fn update_context_stats(&self, context_path: &str, count: usize, activity: f32) -> Result<()> {
        let query = "
            MERGE (c:Context {path: $context_path})
            SET c.memory_count = $count, c.activity_level = $activity
        ";

        self.graph.run(Query::new(query)
            .param("context_path", context_path)
            .param("count", count as i64)
            .param("activity", activity)
        ).await?;

        Ok(())
    }

    fn node_to_memory(&self, node: Node) -> Result<MemoryCell> {
        // Преобразование Node в MemoryCell
        let id = Uuid::parse_str(node.get::<String>("id")?)?;
        let content = node.get::<String>("content")?;
        let summary = node.get::<String>("summary")?;
        let tags = node.get::<Vec<String>>("tags")?;
        let embedding = node.get::<Vec<f32>>("embedding")?;
        let importance = node.get::<f64>("importance")? as f32;
        let access_frequency = node.get::<i64>("access_frequency")? as u32;
        let created_at = DateTime::parse_from_rfc3339(&node.get::<String>("created_at")?)?
            .with_timezone(&Utc);
        let last_accessed = DateTime::parse_from_rfc3339(&node.get::<String>("last_accessed")?)?
            .with_timezone(&Utc);

        Ok(MemoryCell {
            id,
            content,
            summary,
            tags,
            embedding,
            memory_type: MemoryType::Semantic { facts: vec![], concepts: vec![] }, // Упрощение
            importance,
            access_frequency,
            created_at,
            last_accessed,
            context_path: "unknown".to_string(), // Можно получить из отдельного запроса
        })
    }
}
```

### 3.5 Simple Cache

```rust
use moka::future::Cache;
use std::time::Duration;

pub struct MemoryCache {
    semantic_cache: Cache<String, Vec<MemoryCell>>,
    contextual_cache: Cache<String, Vec<MemoryCell>>,
}

impl MemoryCache {
    pub fn new(config: &CacheConfig) -> Self {
        let ttl = Duration::from_secs(config.ttl_seconds);
        
        Self {
            semantic_cache: Cache::builder()
                .max_capacity(config.max_size as u64)
                .time_to_live(ttl)
                .build(),
            contextual_cache: Cache::builder()
                .max_capacity((config.max_size / 2) as u64)
                .time_to_live(ttl)
                .build(),
        }
    }

    pub async fn get_semantic(&self, query: &str) -> Option<Vec<MemoryCell>> {
        self.semantic_cache.get(query).await
    }

    pub async fn store_semantic(&self, query: &str, results: &[MemoryCell]) {
        self.semantic_cache.insert(query.to_string(), results.to_vec()).await;
    }
}
```

## 4. API Interfaces

### 4.1 MCP Server Implementation

```rust
use serde_json::{json, Value};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message};

pub struct McpServer {
    memory_service: Arc<MemoryService>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct McpRequest {
    pub id: String,
    pub method: String,
    pub params: Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct McpResponse {
    pub id: String,
    pub result: Option<Value>,
    pub error: Option<Value>,
}

impl McpServer {
    pub fn new(memory_service: Arc<MemoryService>) -> Self {
        Self { memory_service }
    }

    pub async fn start(&self, addr: &str) -> Result<()> {
        let listener = TcpListener::bind(addr).await?;
        println!("MCP Server listening on {}", addr);

        while let Ok((stream, _)) = listener.accept().await {
            let service = self.memory_service.clone();
            tokio::spawn(async move {
                if let Err(e) = Self::handle_connection(stream, service).await {
                    eprintln!("Connection error: {}", e);
                }
            });
        }

        Ok(())
    }

    async fn handle_connection(stream: TcpStream, service: Arc<MemoryService>) -> Result<()> {
        let ws_stream = accept_async(stream).await?;
        let (mut ws_sender, mut ws_receiver) = ws_stream.split();

        while let Some(msg) = ws_receiver.next().await {
            let msg = msg?;
            if let Message::Text(text) = msg {
                let request: McpRequest = serde_json::from_str(&text)?;
                let response = Self::handle_request(&request, &service).await;
                let response_text = serde_json::to_string(&response)?;
                ws_sender.send(Message::Text(response_text)).await?;
            }
        }

        Ok(())
    }

    async fn handle_request(request: &McpRequest, service: &MemoryService) -> McpResponse {
        let result = match request.method.as_str() {
            "memory.store" => {
                Self::handle_store_memory(request, service).await
            }
            "memory.recall" => {
                Self::handle_recall_memory(request, service).await
            }
            "memory.contexts" => {
                Self::handle_list_contexts(request, service).await
            }
            _ => Err(anyhow::anyhow!("Unknown method: {}", request.method))
        };

        match result {
            Ok(value) => McpResponse {
                id: request.id.clone(),
                result: Some(value),
                error: None,
            },
            Err(e) => McpResponse {
                id: request.id.clone(),
                result: None,
                error: Some(json!({
                    "code": -1,
                    "message": e.to_string()
                })),
            },
        }
    }

    async fn handle_store_memory(request: &McpRequest, service: &MemoryService) -> Result<Value> {
        let content = request.params["content"].as_str().unwrap_or("");
        let context = request.params["context"].as_str().unwrap_or("general");
        
        let memory_id = service.store_memory(content.to_string(), context.to_string()).await?;
        
        Ok(json!({
            "memory_id": memory_id.to_string(),
            "success": true
        }))
    }

    async fn handle_recall_memory(request: &McpRequest, service: &MemoryService) -> Result<Value> {
        let query_text = request.params["query"].as_str().unwrap_or("");
        let context_hint = request.params["context"].as_str().map(|s| s.to_string());
        
        let query = MemoryQuery {
            text: query_text.to_string(),
            context_hint,
            memory_types: None,
            limit: Some(10),
            min_importance: Some(0.1),
        };
        
        let recalled = service.recall_memory(query).await?;
        
        Ok(json!({
            "query_id": recalled.query_id.to_string(),
            "semantic_layer": recalled.semantic_layer,
            "contextual_layer": recalled.contextual_layer,
            "detailed_layer": recalled.detailed_layer,
            "reasoning": recalled.reasoning_chain,
            "confidence": recalled.confidence,
            "recall_time_ms": recalled.recall_time_ms
        }))
    }

    async fn handle_list_contexts(request: &McpRequest, service: &MemoryService) -> Result<Value> {
        // Получение списка контекстов
        let contexts = service.list_contexts().await?;
        
        Ok(json!({
            "contexts": contexts
        }))
    }
}
```

### 4.2 REST API

```rust
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};

pub fn create_api_router(service: Arc<MemoryService>) -> Router {
    Router::new()
        .route("/memory", post(store_memory))
        .route("/memory/recall", post(recall_memory))
        .route("/memory/contexts", get(list_contexts))
        .route("/memory/:id", get(get_memory))
        .route("/health", get(health_check))
        .with_state(service)
}

async fn store_memory(
    State(service): State<Arc<MemoryService>>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, StatusCode> {
    let content = payload["content"].as_str().ok_or(StatusCode::BAD_REQUEST)?;
    let context = payload["context"].as_str().unwrap_or("general");

    match service.store_memory(content.to_string(), context.to_string()).await {
        Ok(id) => Ok(Json(json!({
            "memory_id": id.to_string(),
            "success": true
        }))),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn recall_memory(
    State(service): State<Arc<MemoryService>>,
    Json(query): Json<MemoryQuery>,
) -> Result<Json<RecalledMemory>, StatusCode> {
    match service.recall_memory(query).await {
        Ok(result) => Ok(Json(result)),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn list_contexts(
    State(service): State<Arc<MemoryService>>,
) -> Result<Json<Value>, StatusCode> {
    match service.list_contexts().await {
        Ok(contexts) => Ok(Json(json!({ "contexts": contexts }))),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn health_check() -> Json<Value> {
    Json(json!({
        "status": "healthy",
        "timestamp": Utc::now().to_rfc3339()
    }))
}
```

## 5. Конфигурация и запуск

### 5.1 Основной main.rs

```rust
use tokio;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    // Загрузка конфигурации
    let config = load_config("config.toml")?;
    
    // Инициализация сервиса
    let memory_service = Arc::new(MemoryService::new(config.clone()).await?);
    
    // Запуск MCP сервера
    let mcp_server = McpServer::new(memory_service.clone());
    let mcp_handle = tokio::spawn(async move {
        mcp_server.start("127.0.0.1:8080").await
    });
    
    // Запуск REST API
    let api_router = create_api_router(memory_service.clone());
    let api_handle = tokio::spawn(async move {
        axum::Server::bind(&"127.0.0.1:3000".parse().unwrap())
            .serve(api_router.into_make_service())
            .await
    });

    println!("Memory Service started:");
    println!("- MCP Server: ws://127.0.0.1:8080");
    println!("- REST API: http://127.0.0.1:3000");

    // Ожидание завершения
    tokio::select! {
        _ = mcp_handle => {},
        _ = api_handle => {},
    }

    Ok(())
}

fn load_config(path: &str) -> Result<MemoryConfig> {
    let content = std::fs::read_to_string(path)?;
    let config: MemoryConfig = toml::from_str(&content)?;
    Ok(config)
}
```

### 5.2 Конфигурационный файл (config.toml)

```toml
[neo4j]
uri = "neo4j://localhost:7687"
user = "neo4j"
password = "password"

[onnx]
model_path = "./models/embeddinggemma-300m.onnx"
threads = 4

[gpt5_nano]
api_key = "your-openai-api-key"
base_url = "https://api.openai.com/v1"
model = "gpt-5-nano"
max_tokens = 1000

[cache]
ttl_seconds = 3600
max_size = 10000

[memory]
max_semantic_cache = 1000
max_contextual_cache = 500
max_detailed_cache = 100
importance_decay_days = 30.0
access_boost_factor = 1.1
context_depth_limit = 3
```

### 5.3 Docker Compose для разработки

```yaml
version: '3.8'

services:
  memory-service:
    build: .
    ports:
      - "3000:3000"
      - "8080:8080"
    environment:
      - NEO4J_URI=neo4j://neo4j:7687
      - NEO4J_PASSWORD=password
    volumes:
      - ./models:/app/models
    depends_on:
      - neo4j

  neo4j:
    image: neo4j:5.22-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

### 5.4 Dockerfile

```dockerfile
FROM rust:1.75 as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/memory-service /usr/local/bin/
COPY config.toml /app/

WORKDIR /app

EXPOSE 3000 8080

CMD ["memory-service"]
```

## 6. Примеры использования

### 6.1 Использование через MCP

```python
# Python клиент для MCP
import asyncio
import websockets
import json

async def test_mcp_client():
    uri = "ws://localhost:8080"
    
    async with websockets.connect(uri) as websocket:
        # Сохранение памяти
        store_request = {
            "id": "1",
            "method": "memory.store",
            "params": {
                "content": "Rust - это системный язык программирования, который акцентируется на безопасности и производительности",
                "context": "programming/rust"
            }
        }
        
        await websocket.send(json.dumps(store_request))
        response = await websocket.recv()
        print("Store response:", response)
        
        # Поиск памяти
        recall_request = {
            "id": "2", 
            "method": "memory.recall",
            "params": {
                "query": "Что такое Rust?",
                "context": "programming"
            }
        }
        
        await websocket.send(json.dumps(recall_request))
        response = await websocket.recv()
        print("Recall response:", response)

asyncio.run(test_mcp_client())
```

### 6.2 Использование через REST API

```bash
# Сохранение памяти
curl -X POST http://localhost:3000/memory \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Neo4j - это графовая база данных, идеальная для хранения связанной информации",
    "context": "databases/neo4j"
  }'

# Поиск памяти
curl -X POST http://localhost:3000/memory/recall \
  -H "Content-Type: application/json" \
  -d '{
    "text": "графовая база данных",
    "context_hint": "databases",
    "limit": 5
  }'

# Список контекстов
curl http://localhost:3000/memory/contexts
```

Этот упрощенный, но эффективный архитектурный план создает локальную систему памяти для ИИ агентов, которая эмулирует работу человеческого мозга через трехслойную структуру (семантическая → контекстуальная → детальная память) и использует GPT-5-nano как интеллектуальный "мозг" для обработки и структурирования воспоминаний.