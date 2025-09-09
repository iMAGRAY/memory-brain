#!/usr/bin/env python3
"""
Быстрый тест системы AI Memory Service
"""
import asyncio
import aiohttp
import time
import json
from typing import Dict, List, Any


class AIMemoryTester:
    def __init__(self):
        self.memory_url = "http://127.0.0.1:8080"
        self.embedding_url = "http://localhost:8090"
        self.results = {}

    async def test_health_checks(self) -> Dict[str, Any]:
        """Проверка состояния сервисов"""
        results = {}

        async with aiohttp.ClientSession() as session:
            # Memory server health
            try:
                start = time.time()
                async with session.get(
                    f"{self.memory_url}/health", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    results["memory_server"] = {
                        "status": resp.status,
                        "response_time_ms": int((time.time() - start) * 1000),
                        "healthy": resp.status == 200,
                    }
            except Exception as e:
                results["memory_server"] = {"error": str(e), "healthy": False}

            # Embedding server health
            try:
                start = time.time()
                async with session.get(
                    f"{self.embedding_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    results["embedding_server"] = {
                        "status": resp.status,
                        "response_time_ms": int((time.time() - start) * 1000),
                        "healthy": resp.status == 200,
                    }
            except Exception as e:
                results["embedding_server"] = {"error": str(e), "healthy": False}

        return results

    async def test_search_performance(self) -> Dict[str, Any]:
        """Тестирование производительности поиска"""
        test_queries = [
            "python programming",
            "machine learning",
            "neural networks",
            "memory optimization",
            "database queries",
        ]

        results = []

        async with aiohttp.ClientSession() as session:
            for query in test_queries:
                try:
                    start = time.time()
                    url = f"{self.memory_url}/search?query={query}&limit=5"
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        response_time = int((time.time() - start) * 1000)

                        if resp.status == 200:
                            data = await resp.json()
                            memories = data.get("memories", [])

                            results.append(
                                {
                                    "query": query,
                                    "status": "success",
                                    "response_time_ms": response_time,
                                    "memories_found": len(memories),
                                    "server_time_ms": (
                                        data.get("metrics", {}).get("query_time_ms", 0)
                                    ),
                                }
                            )
                        else:
                            results.append(
                                {
                                    "query": query,
                                    "status": "error",
                                    "response_time_ms": response_time,
                                    "error_code": resp.status,
                                }
                            )

                except Exception as e:
                    results.append(
                        {"query": query, "status": "exception", "error": str(e)}
                    )

                # Пауза между запросами
                await asyncio.sleep(0.5)

        return results

    async def test_memory_stats(self) -> Dict[str, Any]:
        """Получение статистики памяти"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.memory_url}/stats", timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        return {"error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"error": str(e)}

    def calculate_metrics(self, search_results: List[Dict]) -> Dict[str, Any]:
        """Вычисление метрик производительности"""
        if not search_results:
            return {"error": "No search results to analyze"}

        successful = [r for r in search_results if r.get("status") == "success"]

        if not successful:
            return {"error": "No successful queries"}

        response_times = [r["response_time_ms"] for r in successful]
        memories_found = [r["memories_found"] for r in successful]

        return {
            "total_queries": len(search_results),
            "successful_queries": len(successful),
            "success_rate": len(successful) / len(search_results) * 100,
            "avg_response_time_ms": sum(response_times) / len(response_times),
            "min_response_time_ms": min(response_times),
            "max_response_time_ms": max(response_times),
            "avg_memories_found": sum(memories_found) / len(memories_found),
            "total_memories_found": sum(memories_found),
        }

    async def run_full_test(self) -> Dict[str, Any]:
        """Запуск полного тестирования"""
        print("🔍 Запуск тестирования AI Memory Service...")

        # Health checks
        print("📡 Проверка состояния сервисов...")
        health_results = await self.test_health_checks()

        # Search performance
        print("🚀 Тестирование производительности поиска...")
        search_results = await self.test_search_performance()

        # Memory stats
        print("📊 Получение статистики системы...")
        stats_results = await self.test_memory_stats()

        # Calculate metrics
        metrics = self.calculate_metrics(search_results)

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "health_checks": health_results,
            "search_performance": search_results,
            "system_stats": stats_results,
            "performance_metrics": metrics,
        }


async def main():
    tester = AIMemoryTester()
    results = await tester.run_full_test()

    print("\n" + "=" * 60)
    print("🎯 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ AI MEMORY SERVICE")
    print("=" * 60)

    # Health status
    print("\n📡 СОСТОЯНИЕ СЕРВИСОВ:")
    for service, status in results["health_checks"].items():
        health = "✅" if status.get("healthy") else "❌"
        time_ms = status.get("response_time_ms", "N/A")
        print(f"  {health} {service}: {time_ms}ms")

    # Performance metrics
    if "performance_metrics" in results:
        metrics = results["performance_metrics"]
        if "error" not in metrics:
            print("\n🚀 МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ:")
            print(
                f"  • Успешных запросов: {metrics['successful_queries']}/{metrics['total_queries']} ({metrics['success_rate']:.1f}%)"
            )
            print(f"  • Среднее время ответа: {metrics['avg_response_time_ms']:.0f}ms")
            print(
                f"  • Диапазон времени: {metrics['min_response_time_ms']:.0f}-{metrics['max_response_time_ms']:.0f}ms"
            )
            print(
                f"  • Среднее кол-во результатов: {metrics['avg_memories_found']:.1f}"
            )
            print(f"  • Всего найдено воспоминаний: {metrics['total_memories_found']}")
        else:
            print(f"\n❌ Ошибка расчета метрик: {metrics['error']}")

    # System stats
    if "system_stats" in results and "error" not in results["system_stats"]:
        stats = results["system_stats"]
        print("\n📊 СТАТИСТИКА СИСТЕМЫ:")
        print(f"  • Всего воспоминаний: {stats.get('total_memories', 'N/A')}")
        print(f"  • Индексированных: {stats.get('indexed_memories', 'N/A')}")

    print("\n" + "=" * 60)

    # Сохранение детальных результатов
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("💾 Детальные результаты сохранены в test_results.json")


if __name__ == "__main__":
    asyncio.run(main())
