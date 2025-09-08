#!/usr/bin/env python3
"""
Comprehensive System Quality Audit
Тестирует реальное качество каждого компонента AI Memory Service
"""

import sys
import os
import time
import json
import asyncio
import aiohttp
import logging
import subprocess
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests
import psutil

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ComponentTestResult:
    """Результат теста компонента"""
    name: str
    score: float  # 0.0-1.0
    details: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    execution_time: float

@dataclass
class SystemAuditReport:
    """Итоговый отчет аудита системы"""
    overall_score: float
    component_results: List[ComponentTestResult]
    critical_issues: List[str]
    recommendations: List[str]
    execution_time: float

class ConfigurationAuditor:
    """Аудитор конфигурации"""
    
    def audit(self) -> ComponentTestResult:
        logger.info("=== CONFIGURATION AUDIT ===")
        start_time = time.perf_counter()
        score = 1.0
        details = {}
        errors = []
        warnings = []
        
        try:
            # Проверка основных конфигурационных файлов
            config_files = {
                "config.toml": "C:/Models/ai-memory-service/config.toml",
                "Cargo.toml": "C:/Models/ai-memory-service/Cargo.toml",
                ".env": "C:/Models/ai-memory-service/.env"
            }
            
            missing_configs = []
            for name, path in config_files.items():
                if not Path(path).exists():
                    missing_configs.append(name)
                    score -= 0.2
                    
            details["missing_configs"] = missing_configs
            details["config_files_found"] = len(config_files) - len(missing_configs)
            
            # Проверка переменных окружения
            env_vars = ["EMBEDDING_MODEL_PATH", "NEO4J_URI", "NEO4J_USER"]
            missing_env = []
            for var in env_vars:
                if not os.environ.get(var):
                    missing_env.append(var)
                    warnings.append(f"Environment variable {var} not set")
                    score -= 0.1
                    
            details["missing_env_vars"] = missing_env
            details["env_vars_set"] = len(env_vars) - len(missing_env)
            
            # Проверка модели
            model_paths = [
                "C:/Models/ai-memory-service/models/embeddinggemma-300m",
                os.environ.get("EMBEDDING_MODEL_PATH", "")
            ]
            
            model_found = False
            for path in model_paths:
                if path and Path(path).exists():
                    model_found = True
                    details["model_path"] = path
                    break
                    
            if not model_found:
                errors.append("EmbeddingGemma model not found")
                score -= 0.3
                
            details["model_available"] = model_found
            
        except Exception as e:
            errors.append(f"Configuration audit failed: {str(e)}")
            score = 0.0
            
        execution_time = time.perf_counter() - start_time
        
        logger.info(f"Configuration audit completed: score={score:.3f}")
        return ComponentTestResult("Configuration", score, details, errors, warnings, execution_time)

class EmbeddingServiceAuditor:
    """Аудитор сервиса эмбеддингов"""
    
    def audit(self) -> ComponentTestResult:
        logger.info("=== EMBEDDING SERVICE AUDIT ===")
        start_time = time.perf_counter()
        score = 1.0
        details = {}
        errors = []
        warnings = []
        
        try:
            # Импортируем и тестируем embedding server
            from embedding_server import EmbeddingService
            
            # Находим модель
            model_path = self._get_model_path()
            if not model_path:
                errors.append("Model path not found")
                return ComponentTestResult("EmbeddingService", 0.0, {}, errors, warnings, 0)
                
            # Инициализация сервиса
            init_start = time.perf_counter()
            service = EmbeddingService(model_path=model_path, cache_size=10, default_dimension=512)
            init_time = time.perf_counter() - init_start
            details["initialization_time"] = init_time
            
            if init_time > 5.0:
                warnings.append(f"Slow initialization: {init_time:.2f}s")
                score -= 0.1
                
            # Тест базовой функциональности
            test_texts = [
                "This is a test sentence",
                "Another test for embedding",
                "Machine learning algorithm"
            ]
            
            # Query encoding test
            query_start = time.perf_counter()
            query_embeddings = []
            for text in test_texts:
                emb = service.encode_query(text)
                query_embeddings.append(emb)
            query_time = time.perf_counter() - query_start
            details["query_encoding_time"] = query_time
            details["query_avg_time"] = query_time / len(test_texts)
            
            # Document encoding test
            doc_start = time.perf_counter()
            doc_embeddings = [service.encode_document(text) for text in test_texts]
            doc_time = time.perf_counter() - doc_start
            details["batch_encoding_time"] = doc_time
            details["doc_avg_time"] = doc_time / len(test_texts)
            
            # Проверка размерностей
            expected_dim = service.default_dimension
            for i, emb in enumerate(query_embeddings):
                if emb.shape[0] != expected_dim:
                    errors.append(f"Wrong embedding dimension: got {emb.shape[0]}, expected {expected_dim}")
                    score -= 0.2
                    
            details["embedding_dimension"] = expected_dim
            details["embeddings_count"] = len(query_embeddings)
            
            # Проверка нормализации
            norms = [np.linalg.norm(emb) for emb in query_embeddings]
            avg_norm = np.mean(norms)
            details["average_norm"] = avg_norm
            
            if abs(avg_norm - 1.0) > 0.01:
                warnings.append(f"Embeddings not normalized: avg norm = {avg_norm:.4f}")
                score -= 0.1
                
            # Тест производительности
            if details["query_avg_time"] > 0.1:  # 100ms per query is too slow
                warnings.append(f"Slow query encoding: {details['query_avg_time']:.3f}s per query")
                score -= 0.15
                
            if details["doc_avg_time"] > 0.05:  # 50ms per document is too slow for batch
                warnings.append(f"Slow document encoding: {details['doc_avg_time']:.3f}s per document")
                score -= 0.15
                
            # Тест консистентности
            test_text = "Consistency test"
            emb1 = service.encode_query(test_text)
            emb2 = service.encode_query(test_text)
            similarity = np.dot(emb1, emb2)
            details["consistency_similarity"] = similarity
            
            if similarity < 0.999:  # Should be almost identical
                errors.append(f"Inconsistent embeddings: similarity = {similarity:.6f}")
                score -= 0.3
                
            # Очистка ресурсов
            service.cleanup()
            
        except Exception as e:
            errors.append(f"Embedding service test failed: {str(e)}")
            logger.error(f"Embedding service error: {traceback.format_exc()}")
            score = 0.0
            
        execution_time = time.perf_counter() - start_time
        logger.info(f"Embedding service audit completed: score={score:.3f}")
        return ComponentTestResult("EmbeddingService", score, details, errors, warnings, execution_time)
    
    def _get_model_path(self) -> str:
        """Get model path"""
        model_path = os.environ.get('EMBEDDING_MODEL_PATH')
        if model_path and Path(model_path).exists():
            return model_path
        
        possible_paths = [
            r'C:\Models\ai-memory-service\models\embeddinggemma-300m',
            r'models\embeddinggemma-300m'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        return None

class APIAuditor:
    """Аудитор API endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:3001"):
        self.base_url = base_url
        
    def audit(self) -> ComponentTestResult:
        logger.info("=== API AUDIT ===")
        start_time = time.perf_counter()
        score = 1.0
        details = {}
        errors = []
        warnings = []
        
        try:
            # Проверяем что сервер запущен
            if not self._is_server_running():
                errors.append("API server is not running")
                return ComponentTestResult("API", 0.0, {"server_running": False}, errors, warnings, 0)
                
            details["server_running"] = True
            
            # Тест health endpoint
            health_result = self._test_health_endpoint()
            details.update(health_result)
            if not health_result.get("health_ok", False):
                score -= 0.3
                
            # Тест embedding endpoints
            embedding_result = self._test_embedding_endpoints()
            details.update(embedding_result)
            score *= embedding_result.get("embedding_score", 0.0)
            
            # Тест memory endpoints
            memory_result = self._test_memory_endpoints()
            details.update(memory_result)
            score *= memory_result.get("memory_score", 1.0)
            
        except Exception as e:
            errors.append(f"API audit failed: {str(e)}")
            logger.error(f"API audit error: {traceback.format_exc()}")
            score = 0.0
            
        execution_time = time.perf_counter() - start_time
        logger.info(f"API audit completed: score={score:.3f}")
        return ComponentTestResult("API", score, details, errors, warnings, execution_time)
    
    def _is_server_running(self) -> bool:
        """Check if server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def _test_health_endpoint(self) -> Dict[str, Any]:
        """Test health endpoint"""
        result = {"health_ok": False}
        
        try:
            start_time = time.perf_counter()
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response_time = time.perf_counter() - start_time
            
            result["health_response_time"] = response_time
            result["health_status_code"] = response.status_code
            
            if response.status_code == 200:
                result["health_ok"] = True
                try:
                    health_data = response.json()
                    result["health_data"] = health_data
                except:
                    result["health_data"] = response.text
                    
        except Exception as e:
            result["health_error"] = str(e)
            
        return result
    
    def _test_embedding_endpoints(self) -> Dict[str, Any]:
        """Test embedding endpoints"""
        result = {"embedding_score": 0.0}
        
        try:
            # Test encode endpoint
            test_data = {
                "texts": ["Test sentence 1", "Test sentence 2"],
                "task": "query"
            }
            
            start_time = time.perf_counter()
            response = requests.post(f"{self.base_url}/embed", json=test_data, timeout=30)
            response_time = time.perf_counter() - start_time
            
            result["embed_response_time"] = response_time
            result["embed_status_code"] = response.status_code
            
            if response.status_code == 200:
                try:
                    embed_data = response.json()
                    embeddings = embed_data.get("embeddings", [])
                    
                    if embeddings and len(embeddings) == 2:
                        result["embed_count"] = len(embeddings)
                        result["embed_dimension"] = len(embeddings[0]) if embeddings[0] else 0
                        result["embedding_score"] = 1.0
                    else:
                        result["embed_error"] = "Invalid embeddings response"
                        result["embedding_score"] = 0.3
                        
                except Exception as e:
                    result["embed_parse_error"] = str(e)
                    result["embedding_score"] = 0.1
            else:
                result["embed_error"] = f"HTTP {response.status_code}"
                result["embedding_score"] = 0.0
                
        except Exception as e:
            result["embedding_test_error"] = str(e)
            result["embedding_score"] = 0.0
            
        return result
    
    def _test_memory_endpoints(self) -> Dict[str, Any]:
        """Test memory endpoints"""
        result = {"memory_score": 1.0}
        
        try:
            # Test store memory endpoint
            memory_data = {
                "text": "This is a test memory for audit",
                "metadata": {"source": "audit_test", "timestamp": int(time.time())}
            }
            
            start_time = time.perf_counter()
            response = requests.post(f"{self.base_url}/memory", json=memory_data, timeout=30)
            response_time = time.perf_counter() - start_time
            
            result["store_response_time"] = response_time
            result["store_status_code"] = response.status_code
            
            if response.status_code != 200:
                result["store_error"] = f"HTTP {response.status_code}"
                result["memory_score"] *= 0.5
                
            # Test search memory endpoint
            search_data = {"query": "test memory", "limit": 5}
            
            start_time = time.perf_counter()
            response = requests.post(f"{self.base_url}/search", json=search_data, timeout=30)
            search_response_time = time.perf_counter() - start_time
            
            result["search_response_time"] = search_response_time
            result["search_status_code"] = response.status_code
            
            if response.status_code != 200:
                result["search_error"] = f"HTTP {response.status_code}"
                result["memory_score"] *= 0.5
                
        except Exception as e:
            result["memory_test_error"] = str(e)
            result["memory_score"] = 0.0
            
        return result

class PerformanceAuditor:
    """Аудитор производительности"""
    
    def audit(self) -> ComponentTestResult:
        logger.info("=== PERFORMANCE AUDIT ===")
        start_time = time.perf_counter()
        score = 1.0
        details = {}
        errors = []
        warnings = []
        
        try:
            # Системные ресурсы
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            details["cpu_usage_percent"] = cpu_percent
            details["memory_total_gb"] = memory.total / (1024**3)
            details["memory_available_gb"] = memory.available / (1024**3)
            details["memory_usage_percent"] = memory.percent
            details["disk_free_gb"] = disk.free / (1024**3)
            details["disk_usage_percent"] = (disk.used / disk.total) * 100
            
            # Проверка ресурсов
            if cpu_percent > 80:
                warnings.append(f"High CPU usage: {cpu_percent}%")
                score -= 0.1
                
            if memory.percent > 80:
                warnings.append(f"High memory usage: {memory.percent}%")
                score -= 0.1
                
            if details["disk_usage_percent"] > 90:
                warnings.append(f"Low disk space: {details['disk_usage_percent']:.1f}% used")
                score -= 0.1
                
            # Тест файловой системы (скорость чтения/записи)
            io_result = self._test_io_performance()
            details.update(io_result)
            
            if io_result.get("write_speed_mb", 0) < 50:  # Less than 50 MB/s is slow
                warnings.append(f"Slow write speed: {io_result.get('write_speed_mb', 0):.1f} MB/s")
                score -= 0.15
                
            if io_result.get("read_speed_mb", 0) < 100:  # Less than 100 MB/s is slow
                warnings.append(f"Slow read speed: {io_result.get('read_speed_mb', 0):.1f} MB/s")
                score -= 0.15
                
            # Тест многопоточности
            threading_result = self._test_threading_performance()
            details.update(threading_result)
            
        except Exception as e:
            errors.append(f"Performance audit failed: {str(e)}")
            logger.error(f"Performance audit error: {traceback.format_exc()}")
            score = 0.0
            
        execution_time = time.perf_counter() - start_time
        logger.info(f"Performance audit completed: score={score:.3f}")
        return ComponentTestResult("Performance", score, details, errors, warnings, execution_time)
    
    def _test_io_performance(self) -> Dict[str, float]:
        """Test I/O performance"""
        result = {}
        
        try:
            test_file = Path("temp_io_test.dat")
            test_data = b"0" * (10 * 1024 * 1024)  # 10MB
            
            # Write test
            write_start = time.perf_counter()
            test_file.write_bytes(test_data)
            write_time = time.perf_counter() - write_start
            result["write_speed_mb"] = len(test_data) / (1024 * 1024) / write_time
            
            # Read test
            read_start = time.perf_counter()
            read_data = test_file.read_bytes()
            read_time = time.perf_counter() - read_start
            result["read_speed_mb"] = len(read_data) / (1024 * 1024) / read_time
            
            # Cleanup
            test_file.unlink(missing_ok=True)
            
        except Exception as e:
            result["io_test_error"] = str(e)
            
        return result
    
    def _test_threading_performance(self) -> Dict[str, Any]:
        """Test threading performance"""
        result = {}
        
        try:
            def cpu_task(n: int) -> int:
                """Simple CPU-intensive task"""
                return sum(i * i for i in range(n))
            
            # Single-threaded test
            single_start = time.perf_counter()
            single_result = sum(cpu_task(1000) for _ in range(100))
            single_time = time.perf_counter() - single_start
            
            # Multi-threaded test
            multi_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(cpu_task, 1000) for _ in range(100)]
                multi_result = sum(f.result() for f in futures)
            multi_time = time.perf_counter() - multi_start
            
            result["single_thread_time"] = single_time
            result["multi_thread_time"] = multi_time
            result["threading_speedup"] = single_time / multi_time if multi_time > 0 else 0
            result["results_match"] = single_result == multi_result
            
        except Exception as e:
            result["threading_test_error"] = str(e)
            
        return result

class ComprehensiveSystemAuditor:
    """Главный аудитор системы"""
    
    def __init__(self):
        self.auditors = [
            ConfigurationAuditor(),
            EmbeddingServiceAuditor(),
            APIAuditor(),
            PerformanceAuditor()
        ]
    
    def audit(self) -> SystemAuditReport:
        """Проводит полный аудит системы"""
        logger.info("🔍 STARTING COMPREHENSIVE SYSTEM AUDIT")
        logger.info("=" * 80)
        
        start_time = time.perf_counter()
        results = []
        
        for auditor in self.auditors:
            try:
                result = auditor.audit()
                results.append(result)
                
                # Логирование результата
                status = "✅" if result.score > 0.7 else "⚠️" if result.score > 0.3 else "❌"
                logger.info(f"{status} {result.name}: {result.score:.1%} ({result.execution_time:.2f}s)")
                
                if result.errors:
                    for error in result.errors:
                        logger.error(f"  ERROR: {error}")
                        
                if result.warnings:
                    for warning in result.warnings:
                        logger.warning(f"  WARN: {warning}")
                        
            except Exception as e:
                logger.error(f"❌ {auditor.__class__.__name__} failed: {e}")
                results.append(ComponentTestResult(
                    name=auditor.__class__.__name__,
                    score=0.0,
                    details={},
                    errors=[str(e)],
                    warnings=[],
                    execution_time=0.0
                ))
        
        # Вычисляем общую оценку
        if results:
            overall_score = np.mean([r.score for r in results])
        else:
            overall_score = 0.0
        
        # Собираем критические проблемы
        critical_issues = []
        for result in results:
            if result.score < 0.3:
                critical_issues.append(f"{result.name}: Score {result.score:.1%}")
            critical_issues.extend(result.errors)
        
        # Генерируем рекомендации
        recommendations = self._generate_recommendations(results)
        
        execution_time = time.perf_counter() - start_time
        
        report = SystemAuditReport(
            overall_score=overall_score,
            component_results=results,
            critical_issues=critical_issues,
            recommendations=recommendations,
            execution_time=execution_time
        )
        
        self._print_final_report(report)
        return report
    
    def _generate_recommendations(self, results: List[ComponentTestResult]) -> List[str]:
        """Генерирует рекомендации на основе результатов"""
        recommendations = []
        
        for result in results:
            if result.name == "Configuration" and result.score < 0.8:
                recommendations.append("Fix configuration issues: check missing config files and environment variables")
                
            if result.name == "EmbeddingService" and result.score < 0.8:
                if "Slow" in " ".join(result.warnings):
                    recommendations.append("Optimize embedding service performance: consider GPU acceleration or model quantization")
                if result.errors:
                    recommendations.append("Fix embedding service errors: check model path and dependencies")
                    
            if result.name == "API" and result.score < 0.8:
                if not result.details.get("server_running", True):
                    recommendations.append("Start the API server")
                else:
                    recommendations.append("Fix API endpoints: check server logs for errors")
                    
            if result.name == "Performance" and result.score < 0.8:
                recommendations.append("Improve system performance: consider upgrading hardware or optimizing resource usage")
        
        # Общие рекомендации
        overall_score = np.mean([r.score for r in results])
        if overall_score < 0.5:
            recommendations.append("CRITICAL: System requires immediate attention - multiple components are failing")
        elif overall_score < 0.7:
            recommendations.append("System needs optimization - several components have issues")
            
        return recommendations
    
    def _print_final_report(self, report: SystemAuditReport):
        """Выводит финальный отчет"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPREHENSIVE SYSTEM AUDIT REPORT")
        logger.info("=" * 80)
        
        # Общая оценка
        if report.overall_score >= 0.8:
            verdict = "✅ SYSTEM QUALITY: EXCELLENT"
        elif report.overall_score >= 0.6:
            verdict = "⚠️ SYSTEM QUALITY: GOOD (needs minor improvements)"
        elif report.overall_score >= 0.4:
            verdict = "🟡 SYSTEM QUALITY: POOR (needs significant improvements)"
        else:
            verdict = "❌ SYSTEM QUALITY: CRITICAL (multiple failures)"
            
        logger.info(f"{verdict}")
        logger.info(f"Overall Score: {report.overall_score:.1%}")
        logger.info(f"Audit Time: {report.execution_time:.1f}s")
        
        # Детали по компонентам
        logger.info("\n📋 COMPONENT BREAKDOWN:")
        for result in report.component_results:
            status = "✅" if result.score > 0.7 else "⚠️" if result.score > 0.3 else "❌"
            logger.info(f"  {status} {result.name}: {result.score:.1%}")
            
        # Критические проблемы
        if report.critical_issues:
            logger.info(f"\n🚨 CRITICAL ISSUES ({len(report.critical_issues)}):")
            for issue in report.critical_issues:
                logger.info(f"  • {issue}")
        
        # Рекомендации
        if report.recommendations:
            logger.info(f"\n💡 RECOMMENDATIONS ({len(report.recommendations)}):")
            for i, rec in enumerate(report.recommendations, 1):
                logger.info(f"  {i}. {rec}")
        
        logger.info("\n" + "=" * 80)

def main():
    """Главная функция"""
    auditor = ComprehensiveSystemAuditor()
    
    try:
        report = auditor.audit()
        
        # Сохраняем отчет в файл
        report_file = Path("system_audit_report.json")
        report_data = {
            "timestamp": time.time(),
            "overall_score": report.overall_score,
            "execution_time": report.execution_time,
            "components": {
                result.name: {
                    "score": result.score,
                    "details": result.details,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "execution_time": result.execution_time
                }
                for result in report.component_results
            },
            "critical_issues": report.critical_issues,
            "recommendations": report.recommendations
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📄 Report saved to {report_file}")
        
        # Возвращаем код завершения на основе качества
        if report.overall_score >= 0.6:
            return 0  # Success
        else:
            return 1  # Failure
            
    except Exception as e:
        logger.error(f"Audit failed: {e}")
        logger.error(traceback.format_exc())
        return 2  # Error

if __name__ == "__main__":
    sys.exit(main())