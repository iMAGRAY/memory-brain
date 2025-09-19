use crate::{MemoryService, MemoryOrchestrator};
use crate::types::{MemoryCell};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, debug};

#[derive(Debug, Deserialize)]
pub struct PipelineRequest {
    pub query: String,
    pub limit: Option<usize>,
    pub debug: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct PipelineOutput {
    pub query: String,
    pub final_summary: serde_json::Value,
    pub evidence_preview: Vec<EvidenceItem>,
    pub evidence_count: usize,
    pub saved_insights: usize,
    pub pipeline_stats: serde_json::Value,
}

const SYSTEM_PLANNER: &str = "Ты — планировщик поиска. Разбей задачу на подзапросы, критерии и недостающие данные. Верни строгий JSON: {\n  \"subqueries\":[\"...\"],\n  \"constraints\":[\"...\"],\n  \"success_criteria\":[\"...\"]\n}";

const SYSTEM_EVALUATOR: &str = "Ты — проверяющий качество evidence. Верни JSON: {\n  \"coverage\": 0..1,\n  \"missing\":[{\"type\":\"fact|term|context\",\"aspect\":\"...\",\"term\":\"...\",\"context\":\"...\"}],\n  \"bad_evidence_ids\":[\"...\"],\n  \"proceed\": true|false\n}. Не придумывай — опирайся только на evidence.";

const SYSTEM_SYNTHESIZER: &str = "Ты — синтезатор ответа. Верни строгий JSON: {\n  \"summary\": [\"краткие пункты\"],\n  \"key_points\":[{\"text\":\"...\",\"evidence\":[\"memory_id\"],\"confidence\":0..1}],\n  \"gaps\":[\"...\"],\n  \"sources\":[{\"id\":\"...\",\"context_path\":\"...\"}],\n  \"follow_ups\":[\"...\"]\n}. Не добавляй текст вне JSON. Используй только переданные evidence id.";

const SYSTEM_INSIGHT_WRITER: &str = "Ты — генератор инсайтов. По полученному контексту предложи до 20 идей/выводов в JSON: {\n  \"insights\":[{\"text\":\"...\",\"tags\":[\"...\"],\"priority\": 1..5}]\n}. Кратко и по делу.";

#[derive(Debug, Serialize, Clone)]
pub struct EvidenceItem {
    pub id: String,
    pub title: String,
    pub text: String,
    pub context_path: String,
    pub importance: f32,
}

fn slugify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if ch.is_alphanumeric() { out.push(ch.to_ascii_lowercase()); }
        else if ch.is_whitespace() || ch == '/' || ch == '_' || ch == '-' { out.push('-'); }
    }
    while out.contains("--") { out = out.replace("--","-"); }
    out.trim_matches('-').to_string()
}

pub struct MultiOrchestratorPipeline {
    memory: Arc<MemoryService>,
    orchestrator: Arc<MemoryOrchestrator>,
}

impl MultiOrchestratorPipeline {
    pub fn new(memory: Arc<MemoryService>, orchestrator: Arc<MemoryOrchestrator>) -> Self {
        Self { memory, orchestrator }
    }

    pub async fn run(&self, req: PipelineRequest) -> Result<PipelineOutput> {
        let query = req.query.trim().to_string();
        if query.is_empty() { return Err(anyhow!("Empty query")); }
        let k = req.limit.unwrap_or(60).clamp(20, 200);
        let debug_mode = req.debug.unwrap_or(false);

        // 1) Planner: subqueries
        let planner_user = format!("Запрос: {}\nВерни JSON с subqueries/constraints/success_criteria.", query);
        let plan_schema = serde_json::json!({
            "type":"json_schema",
            "json_schema":{
                "name":"planner_output",
                "schema":{
                    "type":"object",
                    "properties":{
                        "subqueries":{"type":"array","items":{"type":"string","maxLength":200},"minItems":1},
                        "constraints":{"type":"array","items":{"type":"string"}},
                        "success_criteria":{"type":"array","items":{"type":"string"}}
                    },
                    "required":["subqueries"],
                    "additionalProperties":false
                },
                "strict": true
            }
        });
        // Budgets: soft limits
        let per_call_max_tokens: usize = std::env::var("PIPELINE_MAX_TOKENS_PER_CALL").ok().and_then(|s| s.parse().ok()).unwrap_or(1500);
        let mut used_in: u64 = 0; let mut used_out: u64 = 0; let mut used_reason: u64 = 0; let mut used_usd: f64 = 0.0;
        let budget_in: Option<u64> = std::env::var("PIPELINE_BUDGET_TOKENS_IN").ok().and_then(|s| s.parse().ok());
        let budget_out: Option<u64> = std::env::var("PIPELINE_BUDGET_TOKENS_OUT").ok().and_then(|s| s.parse().ok());
        let budget_usd: Option<f64> = std::env::var("PIPELINE_BUDGET_USD").ok().and_then(|s| s.parse().ok());

        let plan_call = self.orchestrator.call_json_with_usage(
            "planner", SYSTEM_PLANNER, &planner_user, "medium", Some(plan_schema), Some(per_call_max_tokens)
        ).await?;
        used_in += plan_call.usage.input_tokens; used_out += plan_call.usage.output_tokens; used_reason += plan_call.usage.reasoning_tokens; used_usd += plan_call.usage.cost_usd;
        let plan_json = plan_call.content;
        let plan: serde_json::Value = serde_json::from_str(&extract_json(&plan_json)?)
            .unwrap_or_else(|_| serde_json::json!({"subqueries":[query.clone()]}));
        let subqueries: Vec<String> = plan.get("subqueries").and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
            .filter(|v: &Vec<String>| !v.is_empty()).unwrap_or_else(|| vec![query.clone()]);

        // 2) Retrieval: hybrid search for each subquery
        let mut evidence: Vec<EvidenceItem> = Vec::new();
        for sq in subqueries.iter().take(6) { // до 6 подзапросов
            let items = self.memory.search(sq, k).await.unwrap_or_default();
            for m in items.into_iter().take((k/ subqueries.len().max(1)).max(10)) {
                let title = derive_title(&m);
                let text = derive_text(&m);
                evidence.push(EvidenceItem{
                    id: m.id.to_string(),
                    title,
                    text,
                    context_path: m.context_path.clone(),
                    importance: if m.importance.is_finite() { m.importance } else { 0.0 },
                });
            }
        }
        // de-dup by id
        evidence.sort_by(|a,b| a.id.cmp(&b.id));
        evidence.dedup_by(|a,b| a.id==b.id);
        // limit evidence pack
        if evidence.len() > k { evidence.truncate(k); }

        // 3) Evaluator
        let evidence_json = serde_json::to_string(&evidence).unwrap_or("[]".to_string());
        let evaluator_user = format!("Проверь качество evidence для запроса: {}\nEVIDENCE: {}", query, evidence_json);
        let eval_schema = serde_json::json!({
            "type":"json_schema",
            "json_schema":{
                "name":"evaluator_output",
                "schema":{
                    "type":"object",
                    "properties":{
                        "coverage":{"type":"number","minimum":0.0,"maximum":1.0},
                        "missing":{"type":"array","items":{"type":"string"},"minItems":0},
                        "bad_evidence_ids":{"type":"array","items":{"type":"string"},"minItems":0},
                        "proceed":{"type":"boolean"}
                    },
                    "required":["coverage","proceed"],
                    "additionalProperties":false
                },
                "strict": true
            }
        });
        let remaining_out = budget_out.map(|b| b.saturating_sub(used_out));
        let role_eval_budget: Option<u64> = std::env::var("BUDGET_EVAL_TOKENS_OUT").ok().and_then(|s| s.parse().ok());
        let remaining_eval = role_eval_budget.map(|b| b.saturating_sub(used_out));
        let eval_max_base = remaining_out.map(|r| r.min(per_call_max_tokens as u64) as usize).unwrap_or(per_call_max_tokens);
        let eval_max = remaining_eval.map(|r| r.min(eval_max_base as u64) as usize).unwrap_or(eval_max_base);
        let eval_call = self.orchestrator.call_json_with_usage(
            "evaluator", SYSTEM_EVALUATOR, &evaluator_user, "low", Some(eval_schema.clone()), Some(eval_max)
        ).await?;
        used_in += eval_call.usage.input_tokens; used_out += eval_call.usage.output_tokens; used_reason += eval_call.usage.reasoning_tokens; used_usd += eval_call.usage.cost_usd;
        let eval_json = eval_call.content;
        let eval: serde_json::Value = serde_json::from_str(&extract_json(&eval_json)?)
            .unwrap_or_else(|_| serde_json::json!({"coverage":0.5, "proceed": true, "missing": []}));
        let mut proceed = eval.get("proceed").and_then(|v| v.as_bool()).unwrap_or(true);
        let mut coverage = eval.get("coverage").and_then(|v| v.as_f64()).unwrap_or(0.5);

        // 3.5) Optional second pass evaluator → retrieval: add evidence for missing aspects
        if !proceed || coverage < 0.8 {
            let missing_terms: Vec<String> = extract_missing_terms(&eval);
            if !missing_terms.is_empty() {
                for mt in missing_terms.iter().take(4) {
                    let items = self.memory.search(mt, (k/2).max(10)).await.unwrap_or_default();
                    for m in items.into_iter().take((k/ subqueries.len().max(1)).max(10)) {
                        let title = derive_title(&m);
                        let text = derive_text(&m);
                        if !evidence.iter().any(|e| e.id == m.id.to_string()) {
                            evidence.push(EvidenceItem{
                                id: m.id.to_string(),
                                title,
                                text,
                                context_path: m.context_path.clone(),
                                importance: if m.importance.is_finite() { m.importance } else { 0.0 },
                            });
                        }
                    }
                }
                // Re-evaluate with expanded evidence
                let evidence_json2 = serde_json::to_string(&evidence).unwrap_or_default();
                let evaluator_user2 = format!("Проверь покрытие по запросу: {}\nEVIDENCE: {}", query, evidence_json2);
            let remaining_out2 = budget_out.map(|b| b.saturating_sub(used_out));
            let remaining_eval2 = role_eval_budget.map(|b| b.saturating_sub(used_out));
            let eval2_max_base = remaining_out2.map(|r| r.min(per_call_max_tokens as u64) as usize).unwrap_or(per_call_max_tokens);
            let eval2_max = remaining_eval2.map(|r| r.min(eval2_max_base as u64) as usize).unwrap_or(eval2_max_base);
                let eval2_call = self.orchestrator.call_json_with_usage(
                    "evaluator", SYSTEM_EVALUATOR, &evaluator_user2, "medium", Some(eval_schema), Some(eval2_max)
                ).await?;
                used_in += eval2_call.usage.input_tokens; used_out += eval2_call.usage.output_tokens; used_reason += eval2_call.usage.reasoning_tokens; used_usd += eval2_call.usage.cost_usd;
                let eval_json2 = eval2_call.content;
                let eval2: serde_json::Value = serde_json::from_str(&extract_json(&eval_json2)?)
                    .unwrap_or_else(|_| serde_json::json!({"coverage":0.7, "proceed": true}));
                proceed = eval2.get("proceed").and_then(|v| v.as_bool()).unwrap_or(true);
                coverage = eval2.get("coverage").and_then(|v| v.as_f64()).unwrap_or(coverage);
            }
        }

        // 4) Synthesizer
        let synth_user = format!("Собери ответ по запросу: {}\nEVIDENCE: {}", query, evidence_json);
        let synth_schema = serde_json::json!({
            "type":"json_schema",
            "json_schema":{
                "name":"synth_output",
                "schema":{
                    "type":"object",
                    "properties":{
                        "summary":{"type":"array","items":{"type":"string"},"minItems":1},
                        "key_points":{"type":"array","items":{
                            "type":"object",
                            "properties":{
                                "text":{"type":"string"},
                                "evidence":{"type":"array","items":{"type":"string"}},
                                "confidence":{"type":"number"}
                            },
                            "required":["text","evidence","confidence"],
                            "additionalProperties":false
                        },"minItems":0},
                        "gaps":{"type":"array","items":{"type":"string"}},
                        "sources":{"type":"array","items":{
                            "type":"object",
                            "properties":{
                                "id":{"type":"string"},
                                "context_path":{"type":"string"}
                            },
                            "required":["id"],
                            "additionalProperties":false
                        },"minItems":0},
                        "follow_ups":{"type":"array","items":{"type":"string"}}
                    },
                    "required":["summary","key_points"],
                    "additionalProperties":false
                },
                "strict": true
            }
        });
        let remaining_out3 = budget_out.map(|b| b.saturating_sub(used_out));
        let role_synth_budget: Option<u64> = std::env::var("BUDGET_SYNTH_TOKENS_OUT").ok().and_then(|s| s.parse().ok());
        let remaining_synth = role_synth_budget.map(|b| b.saturating_sub(used_out));
        let synth_max_base = remaining_out3.map(|r| r.min(per_call_max_tokens as u64) as usize).unwrap_or(per_call_max_tokens);
        let synth_max = remaining_synth.map(|r| r.min(synth_max_base as u64) as usize).unwrap_or(synth_max_base);
        let synth_call = self.orchestrator.call_json_with_usage(
            "synthesizer", SYSTEM_SYNTHESIZER, &synth_user, if proceed {"medium"} else {"high"}, Some(synth_schema), Some(synth_max)
        ).await?;
        used_in += synth_call.usage.input_tokens; used_out += synth_call.usage.output_tokens; used_reason += synth_call.usage.reasoning_tokens; used_usd += synth_call.usage.cost_usd;
        let synth_json = synth_call.content;
        let final_summary: serde_json::Value = serde_json::from_str(&extract_json(&synth_json)?)
            .unwrap_or_else(|_| serde_json::json!({"summary":["Нет валидного ответа"],"key_points":[],"sources":[]}));

        // 5) Insight writer (background-like, но синхронно для простоты)
        let iw_user = format!("Создай до 20 инсайтов по запросу: {}\nКОНТЕКСТ: {}", query, serde_json::to_string(&final_summary).unwrap_or_default());
        let iw_schema = serde_json::json!({
            "type":"json_schema",
            "json_schema":{
                "name":"insights_output",
                "schema":{
                    "type":"object",
                    "properties":{
                        "insights":{"type":"array","items":{
                            "type":"object",
                            "properties":{
                                "text":{"type":"string"},
                                "tags":{"type":"array","items":{"type":"string"}},
                                "priority":{"type":"integer"}
                            },
                            "required":["text"],
                            "additionalProperties":false
                        }}
                    },
                    "required":["insights"],
                    "additionalProperties":false
                },
                "strict": true
            }
        });
        let mut budget_exhausted = false;
        if let Some(b) = budget_usd { if used_usd >= b { budget_exhausted = true; } }
        if let Some(b) = budget_out { if used_out >= b { budget_exhausted = true; } }
        let role_ins_budget: Option<u64> = std::env::var("BUDGET_INSIGHTS_TOKENS_OUT").ok().and_then(|s| s.parse().ok());
        if let Some(b) = role_ins_budget { if used_out >= b { budget_exhausted = true; } }
        let iw_json = if budget_exhausted {
            "{\"insights\":[]}".to_string()
        } else {
            let remaining_out4 = budget_out.map(|b| b.saturating_sub(used_out));
            let remaining_ins = role_ins_budget.map(|b| b.saturating_sub(used_out));
            let iw_max_base = remaining_out4.map(|r| r.min(per_call_max_tokens as u64) as usize).unwrap_or(per_call_max_tokens);
            let iw_max = remaining_ins.map(|r| r.min(iw_max_base as u64) as usize).unwrap_or(iw_max_base);
            let iw_call = self.orchestrator.call_json_with_usage(
                "insightwriter", SYSTEM_INSIGHT_WRITER, &iw_user, "low", Some(iw_schema), Some(iw_max)
            ).await?;
            used_in += iw_call.usage.input_tokens; used_out += iw_call.usage.output_tokens; used_reason += iw_call.usage.reasoning_tokens; used_usd += iw_call.usage.cost_usd;
            iw_call.content
        };
        let iw: serde_json::Value = serde_json::from_str(&extract_json(&iw_json)?)
            .unwrap_or_else(|_| serde_json::json!({"insights":[]}));
        let mut saved = 0usize;
        if let Some(arr) = iw.get("insights").and_then(|v| v.as_array()) {
            let ctx = format!("insights/{}", slugify(&query));
            for it in arr.iter().take(20) {
                if let Some(text) = it.get("text").and_then(|v| v.as_str()) {
                    if self.memory.store_memory(text.to_string(), Some(ctx.clone())).await.is_ok() { saved += 1; }
                }
            }
        }

        // Evidence preview (up to 12 items)
        let mut evidence_preview = evidence.clone();
        evidence_preview.truncate(12);

        let out = PipelineOutput{
            query: query.clone(),
            final_summary,
            evidence_preview,
            evidence_count: evidence.len(),
            saved_insights: saved,
            pipeline_stats: serde_json::json!({
                "subqueries": subqueries,
                "coverage": coverage,
                "debug": debug_mode,
                "usage": {"input": used_in, "output": used_out, "reasoning": used_reason, "usd": used_usd},
                "budget": {"in": budget_in, "out": budget_out, "usd": budget_usd},
                "degraded_embedding": !self.memory.embedding_available(),
                "degraded": budget_exhausted || !self.memory.embedding_available()
            }),
        };
        Ok(out)
    }
}

fn derive_title(m: &MemoryCell) -> String {
    // Берем summary или усечённый content как title
    if !m.summary.trim().is_empty() { return m.summary.clone(); }
    let c = m.content.trim();
    let line = c.lines().next().unwrap_or(c);
    line.chars().take(120).collect()
}

fn derive_text(m: &MemoryCell) -> String {
    let c = m.content.trim();
    if c.is_empty() { return String::new(); }
    let mut out = c.to_string();
    if out.len() > 800 { out.truncate(800); }
    out
}

fn extract_json(s: &str) -> Result<String> {
    let start = s.find('{').ok_or_else(|| anyhow!("No JSON start"))?;
    let end = s.rfind('}').ok_or_else(|| anyhow!("No JSON end"))?;
    if end <= start { return Err(anyhow!("Bad JSON bounds")); }
    Ok(s[start..=end].to_string())
}

/// Robust extractor for evaluator.missing: accepts strings or typed objects
fn extract_missing_terms(eval: &serde_json::Value) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if let Some(arr) = eval.get("missing").and_then(|v| v.as_array()) {
        for it in arr {
            if let Some(s) = it.as_str() {
                if !s.trim().is_empty() { out.push(s.trim().to_string()); }
                continue;
            }
            if let Some(obj) = it.as_object() {
                let term = obj.get("term").and_then(|v| v.as_str()).unwrap_or("");
                let aspect = obj.get("aspect").and_then(|v| v.as_str()).unwrap_or("");
                let context = obj.get("context").and_then(|v| v.as_str()).unwrap_or("");
                let mut s = String::new();
                if !term.is_empty() { s.push_str(term); }
                if s.is_empty() && !aspect.is_empty() { s.push_str(aspect); }
                if s.is_empty() && !context.is_empty() { s.push_str(context); }
                if !s.is_empty() { out.push(s); }
            }
        }
    }
    out
}
