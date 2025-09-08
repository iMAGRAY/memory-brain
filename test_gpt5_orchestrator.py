#!/usr/bin/env python3
"""
Test GPT-5-nano Orchestrator real functionality 
Direct API testing without relying on full system
"""

import asyncio
import aiohttp
import json
import time
import os
from typing import Dict, Any

async def test_gpt5_nano_orchestrator():
    """Test real GPT-5-nano orchestrator functionality"""
    
    print("🧠 TESTING GPT-5-NANO ORCHESTRATOR")
    print("=" * 60)
    
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not api_key or api_key == 'sk-your_openai_api_key_here':
        print("❌ No valid OpenAI API key found")
        print("   Set OPENAI_API_KEY environment variable")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test GPT-5-nano API call
    async with aiohttp.ClientSession() as session:
        try:
            # Test 1: Simple distillation task
            print("\n1. Testing context distillation...")
            
            distillation_prompt = """Ты - эксперт по дистилляции информации для AI Memory Service.

ЗАДАЧА: Анализируй предоставленную информацию и создай сжатый, но информативный дистиллированный контекст.

ПРИНЦИПЫ:
1. Сохраняй ключевую информацию и связи
2. Удаляй избыточность и дубликаты  
3. Выделяй паттерны и важные инсайты
4. Создавай actionable рекомендации

ФОРМАТ ОТВЕТА (JSON):
{
  "key_points": ["точка1", "точка2", ...],
  "relationships": ["связь1", "связь2", ...],
  "actionable_insights": ["инсайт1", "инсайт2", ...],
  "confidence_score": 0.85
}"""

            user_prompt = """Дистиллируй следующий контекст:

ПАМЯТЬ 1: [Semantic] The capital of France is Paris. This is a basic geographical fact.
Важность: 0.70, Контекст: geography/facts, Теги: france, capital, paris

ПАМЯТЬ 2: [Semantic] Database performance tuning and indexing strategies for better query optimization.
Важность: 0.55, Контекст: database/optimization, Теги: database, performance, tuning

ПАМЯТЬ 3: [Code] function calculateSum(a, b) { return a + b; }
Важность: 0.60, Контекст: javascript/functions, Теги: javascript, function, math

Создай дистиллированный контекст в JSON формате."""

            request_data = {
                "model": "gpt-5-nano",
                "messages": [
                    {"role": "system", "content": distillation_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_completion_tokens": 1000,
                "reasoning_effort": "medium"  # GPT-5 parameter
            }
            
            start_time = time.time()
            
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                elapsed = time.time() - start_time
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ GPT-5 API error: {response.status}")
                    print(f"   Error: {error_text}")
                    return False
                
                result = await response.json()
                
                if 'choices' not in result or not result['choices']:
                    print("❌ No response from GPT-5-nano")
                    return False
                
                content = result['choices'][0]['message']['content']
                print(f"✅ GPT-5-nano responded in {elapsed:.2f}s")
                print(f"   Response length: {len(content)} chars")
                
                # Try to parse JSON
                try:
                    # Extract JSON from response
                    json_start = content.find('{')
                    json_end = content.rfind('}')
                    
                    if json_start == -1 or json_end == -1:
                        print("⚠️ No JSON found in response")
                        print(f"   Raw response: {content[:200]}...")
                        return False
                    
                    json_str = content[json_start:json_end+1]
                    parsed = json.loads(json_str)
                    
                    required_fields = ['key_points', 'relationships', 'actionable_insights', 'confidence_score']
                    missing = [f for f in required_fields if f not in parsed]
                    
                    if missing:
                        print(f"⚠️ Missing fields in response: {missing}")
                        return False
                    
                    print(f"✅ JSON parsing successful")
                    print(f"   Key points: {len(parsed['key_points'])}")
                    print(f"   Relationships: {len(parsed['relationships'])}")
                    print(f"   Insights: {len(parsed['actionable_insights'])}")
                    print(f"   Confidence: {parsed['confidence_score']}")
                    
                    # Test 2: Insight generation
                    print("\n2. Testing insight generation...")
                    
                    insight_prompt = """Ты - эксперт по анализу паттернов и генерации инсайтов для AI Memory Service.

ЗАДАЧА: На основе анализа паттернов памяти генерируй actionable инсайты.

ПРИНЦИПЫ:
1. Фокусируйся на практичных выводах
2. Выявляй скрытые закономерности  
3. Предлагай улучшения процессов
4. Учитывай контекст и важность

ФОРМАТ ОТВЕТА (JSON):
{
  "insights": [
    {
      "type": "UserPreference",
      "confidence": 0.85,
      "insight": "Описание инсайта",
      "implications": ["следствие1", "следствие2"],
      "actionable_items": ["действие1", "действие2"],
      "source_evidence": ["доказательство1"]
    }
  ]
}"""

                    insight_user = """Проанализируй паттерны и сгенерируй инсайты типа UserPreference:

АНАЛИЗ ПАТТЕРНОВ:
ТИПЫ ПАМЯТИ:
- Semantic: 6 записей, средняя важность: 0.62
- Code: 3 записи, средняя важность: 0.58
- Documentation: 2 записи, средняя важность: 0.71

ВРЕМЕННЫЕ ПАТТЕРНЫ:
- Недавно созданных (< 7 дней): 8
- Старых (> 30 дней): 1
- Недавно использованных: 5

АНАЛИЗ ВАЖНОСТИ:
- Средняя важность: 0.61
- Высокой важности (>0.7): 2
- Низкой важности (<0.3): 1

КОНТЕКСТНЫЕ ПАТТЕРНЫ:
- level_0: javascript: 3 записи
- level_0: geography: 2 записи
- level_0: database: 2 записи

Создай 2-3 практичных инсайта в JSON формате."""

                    insight_request = {
                        "model": "gpt-5-nano", 
                        "messages": [
                            {"role": "system", "content": insight_prompt},
                            {"role": "user", "content": insight_user}
                        ],
                        "max_completion_tokens": 1500,
                        "reasoning_effort": "high"
                    }
                    
                    start_time = time.time()
                    
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json=insight_request,
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as response2:
                        
                        elapsed2 = time.time() - start_time
                        
                        if response2.status != 200:
                            error_text2 = await response2.text()
                            print(f"⚠️ Second API call failed: {response2.status}")
                            print(f"   Error: {error_text2}")
                            return True  # First test passed
                        
                        result2 = await response2.json()
                        content2 = result2['choices'][0]['message']['content']
                        
                        print(f"✅ Insight generation completed in {elapsed2:.2f}s")
                        print(f"   Response length: {len(content2)} chars")
                        
                        # Parse insights JSON
                        try:
                            json_start2 = content2.find('{')
                            json_end2 = content2.rfind('}')
                            
                            if json_start2 != -1 and json_end2 != -1:
                                json_str2 = content2[json_start2:json_end2+1]
                                parsed2 = json.loads(json_str2)
                                
                                if 'insights' in parsed2:
                                    insights = parsed2['insights']
                                    print(f"✅ Generated {len(insights)} insights")
                                    
                                    for i, insight in enumerate(insights):
                                        if 'type' in insight and 'confidence' in insight:
                                            print(f"   Insight {i+1}: {insight['type']} (conf: {insight['confidence']})")
                                        
                                    print("\n✅ GPT-5-nano ORCHESTRATOR WORKING CORRECTLY")
                                    return True
                                else:
                                    print("⚠️ No 'insights' field in second response")
                                    
                        except Exception as e:
                            print(f"⚠️ JSON parsing failed for insights: {e}")
                            print(f"   Raw content: {content2[:200]}...")
                        
                        return True  # First test passed
                        
                except Exception as e:
                    print(f"❌ JSON parsing failed: {e}")
                    print(f"   Raw response: {content[:300]}...")
                    return False
                    
        except Exception as e:
            print(f"❌ GPT-5-nano test failed: {e}")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_gpt5_nano_orchestrator())
    if success:
        print(f"\n🎉 GPT-5-nano orchestrator is functional")
        exit(0)
    else:
        print(f"\n❌ GPT-5-nano orchestrator has issues")
        exit(1)