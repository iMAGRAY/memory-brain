#!/usr/bin/env python3
"""
🔍 FORENSIC INVESTIGATION x200000% MAGNIFICATION
AI Memory Service Deception Detection System
========================================

Mission: Detect and expose any deception in the memory system
Approach: Direct embedding and search testing with cross-validation
Magnification: x200000% - ultimate precision investigation
"""

import asyncio
import json
import time
import requests
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import sys
from contextlib import asynccontextmanager
from requests.exceptions import RequestException, Timeout, ConnectionError
import traceback

# Configure forensic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='🔬 %(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('forensic_investigation_x200000.log', mode='w')
    ]
)

logger = logging.getLogger('ForensicInvestigator')

class ForensicInvestigationError(Exception):
    """Base exception for forensic investigation errors"""
    pass

class EmbeddingServiceError(ForensicInvestigationError):
    """Exception for embedding service related errors"""
    pass

class MemorySearchError(ForensicInvestigationError):
    """Exception for memory search related errors"""
    pass

@dataclass
class DeceptionEvidence:
    test_name: str
    is_deceptive: bool
    confidence: float
    evidence: List[str]
    raw_data: Dict[str, Any]

class ForensicInvestigatorX200000:
    """Ultimate precision forensic investigator for AI Memory Service"""
    
    def __init__(self):
        self.embedding_url = "http://localhost:8090/embed"
        self.memory_url = "http://localhost:8080"
        self.evidence = []
        self.magnification_level = 200000
        self.request_timeout = 15
        self.max_retries = 3
        
    async def investigate_deception(self) -> Dict[str, Any]:
        """Comprehensive deception investigation at x200000% magnification"""
        
        logger.info(f"🚨 STARTING FORENSIC INVESTIGATION x{self.magnification_level}%")
        logger.info("=" * 80)
        logger.info("🎯 MISSION: Detect and expose deception in AI Memory Service")
        logger.info("🔬 APPROACH: Multi-layer validation with cross-verification")
        logger.info("⚡ PRECISION: Maximum possible - leaving no stone unturned")
        logger.info("=" * 80)
        
        investigation_results = []
        
        try:
            # Phase 1: Embedding Service Investigation
            logger.info("📍 PHASE 1: Embedding Service Deception Detection")
            embedding_evidence = await self.investigate_embedding_service()
            investigation_results.append(embedding_evidence)
            
        except EmbeddingServiceError as e:
            logger.error(f"❌ Embedding service investigation failed: {e}")
            investigation_results.append(DeceptionEvidence(
                test_name="Embedding Service Authenticity",
                is_deceptive=True,
                confidence=0.9,
                evidence=[f"🚨 CRITICAL: Embedding service investigation failed: {str(e)}"],
                raw_data={"error": str(e)}
            ))
            
        try:
            # Phase 2: Memory Search Investigation  
            logger.info("\n📍 PHASE 2: Memory Search Result Manipulation Detection")
            search_evidence = await self.investigate_search_manipulation()
            investigation_results.append(search_evidence)
            
        except MemorySearchError as e:
            logger.error(f"❌ Memory search investigation failed: {e}")
            investigation_results.append(DeceptionEvidence(
                test_name="Memory Search Manipulation",
                is_deceptive=True,
                confidence=0.9,
                evidence=[f"🚨 CRITICAL: Memory search investigation failed: {str(e)}"],
                raw_data={"error": str(e)}
            ))
        
        try:
            # Phase 3: Vector Similarity Investigation
            logger.info("\n📍 PHASE 3: Vector Similarity Calculation Authenticity")
            similarity_evidence = await self.investigate_similarity_authenticity()
            investigation_results.append(similarity_evidence)
            
        except ForensicInvestigationError as e:
            logger.error(f"❌ Similarity investigation failed: {e}")
            investigation_results.append(DeceptionEvidence(
                test_name="Vector Similarity Authenticity",
                is_deceptive=True,
                confidence=0.8,
                evidence=[f"🚨 CRITICAL: Similarity investigation failed: {str(e)}"],
                raw_data={"error": str(e)}
            ))
            
        try:
            # Phase 4: Cross-validation Investigation
            logger.info("\n📍 PHASE 4: Cross-Validation and Consistency Checks")
            consistency_evidence = await self.investigate_consistency()
            investigation_results.append(consistency_evidence)
            
        except ForensicInvestigationError as e:
            logger.error(f"❌ Consistency investigation failed: {e}")
            investigation_results.append(DeceptionEvidence(
                test_name="Consistency and Cross-validation",
                is_deceptive=True,
                confidence=0.7,
                evidence=[f"🚨 CRITICAL: Consistency investigation failed: {str(e)}"],
                raw_data={"error": str(e)}
            ))
        
        # Compile final investigation report
        return self.compile_investigation_report(investigation_results)
        
    async def investigate_embedding_service(self) -> DeceptionEvidence:
        """Investigate embedding service for deception patterns"""
        logger.info("🔍 Testing embedding service authenticity...")
        
        test_queries = [
            "machine learning algorithms neural networks",
            "machine learning algorithms neural networks",  # Duplicate for consistency
            "completely different topic about cooking recipes",
            "random nonsense xyz123abc456def789",
            "еще один запрос на русском языке",
            "",  # Empty string test
        ]
        
        embeddings = []
        evidence = []
        
        for i, query in enumerate(test_queries):
            logger.info(f"🧪 Testing query {i+1}: '{query}'")
            
            try:
                embedding = await self.get_embedding(query)
                if embedding:
                    embeddings.append((query, embedding))
                    logger.debug(f"✅ Got embedding vector of dimension {len(embedding)}")
                else:
                    evidence.append(f"🚨 DECEPTION: Failed to get embedding for query: '{query}'")
                    
            except EmbeddingServiceError as e:
                evidence.append(f"🚨 DECEPTION: Embedding service error for query '{query}': {str(e)}")
                logger.error(f"Embedding service error for query '{query}': {e}")
                
        # Analyze embeddings for deception patterns
        if len(embeddings) >= 2:
            try:
                # Check duplicate query consistency
                if len(embeddings) >= 2:
                    sim = self.cosine_similarity(embeddings[0][1], embeddings[1][1])
                    logger.info(f"🔬 Identical queries similarity: {sim:.6f}")
                    
                    if sim < 0.999:
                        evidence.append(f"🚨 DECEPTION: Identical queries have similarity {sim:.6f} < 0.999")
                
                # Check if different topics have suspiciously high similarity
                if len(embeddings) >= 3:
                    ml_embedding = embeddings[0][1]  # Machine learning query
                    cooking_embedding = embeddings[2][1]  # Cooking query
                    cross_sim = self.cosine_similarity(ml_embedding, cooking_embedding)
                    
                    logger.info(f"🔬 Cross-topic similarity (ML vs Cooking): {cross_sim:.6f}")
                    
                    if cross_sim > 0.7:  # Suspiciously high
                        evidence.append(f"🚨 DECEPTION: Unrelated topics have suspiciously high similarity: {cross_sim:.6f}")
                
                # Check embedding distribution
                all_embeddings = [emb[1] for emb in embeddings]
                norms = [np.linalg.norm(emb) for emb in all_embeddings]
                avg_norm = np.mean(norms)
                std_norm = np.std(norms)
                
                logger.info(f"🔬 Embedding norms - avg: {avg_norm:.6f}, std: {std_norm:.6f}")
                
                if std_norm < 0.001:  # All embeddings suspiciously similar norms
                    evidence.append(f"🚨 DECEPTION: All embeddings have suspiciously similar norms (std: {std_norm:.6f})")
                
            except (ValueError, TypeError, ZeroDivisionError) as e:
                evidence.append(f"🚨 DECEPTION: Mathematical error during embedding analysis: {str(e)}")
                logger.error(f"Mathematical error during embedding analysis: {e}")
                
        is_deceptive = len(evidence) > 0
        confidence = min(len(evidence) * 0.3, 1.0) if is_deceptive else 0.9
        
        return DeceptionEvidence(
            test_name="Embedding Service Authenticity",
            is_deceptive=is_deceptive,
            confidence=confidence,
            evidence=evidence,
            raw_data={"embeddings": [(q, len(e) if e else 0) for q, e in embeddings]}
        )
        
    async def investigate_search_manipulation(self) -> DeceptionEvidence:
        """Investigate memory search for result manipulation"""
        logger.info("🕵️ Investigating memory search result manipulation...")
        
        test_queries = [
            ("relevant", "machine learning neural networks artificial intelligence"),
            ("irrelevant", "cooking recipes pasta italian food"),
            ("nonsense", "xyzabc123def456ghi789jkl"),
            ("empty", ""),
            ("special", "!@#$%^&*()_+-=[]{}|;:,.<>?"),
            ("russian", "машинное обучение нейронные сети"),
        ]
        
        evidence = []
        search_results = []
        
        for label, query in test_queries:
            logger.info(f"🎯 Testing search query '{label}': '{query}'")
            
            try:
                # Test embedding generation first
                embedding = await self.get_embedding(query)
                if not embedding:
                    evidence.append(f"🚨 DECEPTION: Cannot generate embedding for '{label}' query")
                    continue
                    
                # Test memory search via API
                results = await self.search_memories(query)
                
                result_count = len(results) if results else 0
                search_results.append((label, query, result_count, results))
                
                logger.info(f"📊 Query '{label}' returned {result_count} results")
                
                # Log first few results for analysis
                if results and len(results) > 0:
                    for i, result in enumerate(results[:3]):
                        if isinstance(result, dict):
                            content = result.get('content', 'N/A')[:100]
                            score = result.get('relevance_score', result.get('similarity', 'N/A'))
                            logger.debug(f"  Result {i+1}: score={score}, content='{content}...'")
                            
            except (EmbeddingServiceError, MemorySearchError) as e:
                evidence.append(f"🚨 DECEPTION: Service error for '{label}' query: {str(e)}")
                logger.error(f"Service error for '{label}' query: {e}")
                
        # Analyze search patterns for manipulation
        if search_results:
            try:
                result_counts = [r[2] for r in search_results if r[2] is not None]
                
                if result_counts:
                    unique_counts = set(result_counts)
                    logger.info(f"🔬 Search result count analysis:")
                    for label, query, count, _ in search_results:
                        logger.info(f"  '{label}' query: {count} results")
                    
                    # Check if all queries return same count (suspicious)
                    if len(unique_counts) == 1 and len(result_counts) > 3 and result_counts[0] > 0:
                        evidence.append(f"🚨 MANIPULATION: All {len(result_counts)} queries return exactly {result_counts[0]} results")
                    
                    # Check if nonsense queries return results (suspicious)
                    for label, query, count, results in search_results:
                        if label == "nonsense" and count > 0:
                            evidence.append(f"🚨 MANIPULATION: Nonsense query '{query}' returned {count} results")
                        if label == "empty" and count > 0:
                            evidence.append(f"🚨 MANIPULATION: Empty query returned {count} results")
                    
                    # Check result overlap between completely different queries
                    if len(search_results) >= 2:
                        ml_results = next((r[3] for r in search_results if r[0] == "relevant"), [])
                        cooking_results = next((r[3] for r in search_results if r[0] == "irrelevant"), [])
                        
                        if ml_results and cooking_results:
                            ml_ids = {r.get('id', str(i)) for i, r in enumerate(ml_results) if isinstance(r, dict)}
                            cooking_ids = {r.get('id', str(i)) for i, r in enumerate(cooking_results) if isinstance(r, dict)}
                            
                            overlap = len(ml_ids.intersection(cooking_ids))
                            total_unique = len(ml_ids.union(cooking_ids))
                            
                            if total_unique > 0:
                                overlap_ratio = overlap / total_unique
                                logger.info(f"🔬 Result overlap ratio: {overlap_ratio:.3f}")
                                
                                if overlap_ratio > 0.5:
                                    evidence.append(f"🚨 MANIPULATION: High result overlap ({overlap_ratio:.3f}) between unrelated queries")
                                    
            except (ValueError, TypeError, ZeroDivisionError) as e:
                evidence.append(f"🚨 DECEPTION: Mathematical error during search analysis: {str(e)}")
                logger.error(f"Mathematical error during search analysis: {e}")
        
        is_deceptive = len(evidence) > 0
        confidence = min(len(evidence) * 0.25, 1.0) if is_deceptive else 0.8
        
        return DeceptionEvidence(
            test_name="Memory Search Manipulation",
            is_deceptive=is_deceptive,
            confidence=confidence,
            evidence=evidence,
            raw_data={"search_results": [(r[0], r[1], r[2]) for r in search_results]}
        )
        
    async def investigate_similarity_authenticity(self) -> DeceptionEvidence:
        """Investigate vector similarity calculation authenticity"""
        logger.info("📐 Investigating vector similarity calculation authenticity...")
        
        evidence = []
        
        # Test with known similarity patterns
        test_cases = [
            ("identical", "test query", "test query"),
            ("similar", "machine learning AI", "artificial intelligence ML"),
            ("different", "machine learning", "cooking recipes"),
            ("opposite", "good positive", "bad negative")
        ]
        
        similarities = []
        
        for case_name, query1, query2 in test_cases:
            logger.info(f"🧮 Testing similarity case: '{case_name}'")
            
            try:
                emb1 = await self.get_embedding(query1)
                emb2 = await self.get_embedding(query2)
                
                if emb1 and emb2:
                    similarity = self.cosine_similarity(emb1, emb2)
                    similarities.append((case_name, query1, query2, similarity))
                    
                    logger.info(f"  '{query1}' vs '{query2}': {similarity:.6f}")
                    
                    # Check for deceptive patterns
                    if case_name == "identical" and similarity < 0.99:
                        evidence.append(f"🚨 DECEPTION: Identical strings have similarity {similarity:.6f} < 0.99")
                    elif case_name == "different" and similarity > 0.8:
                        evidence.append(f"🚨 DECEPTION: Very different topics have high similarity {similarity:.6f}")
                else:
                    evidence.append(f"🚨 DECEPTION: Could not get embeddings for similarity test '{case_name}'")
                    
            except (EmbeddingServiceError, ValueError, TypeError) as e:
                evidence.append(f"🚨 DECEPTION: Error during similarity test '{case_name}': {str(e)}")
                logger.error(f"Error during similarity test '{case_name}': {e}")
                
        is_deceptive = len(evidence) > 0
        confidence = min(len(evidence) * 0.4, 1.0) if is_deceptive else 0.7
        
        return DeceptionEvidence(
            test_name="Vector Similarity Authenticity", 
            is_deceptive=is_deceptive,
            confidence=confidence,
            evidence=evidence,
            raw_data={"similarities": similarities}
        )
        
    async def investigate_consistency(self) -> DeceptionEvidence:
        """Cross-validation and consistency investigation"""
        logger.info("🔄 Performing cross-validation consistency checks...")
        
        evidence = []
        
        # Test embedding service consistency
        test_query = "machine learning neural networks"
        
        logger.info(f"🔁 Testing consistency with repeated calls for: '{test_query}'")
        
        embeddings = []
        
        for i in range(3):
            try:
                emb = await self.get_embedding(test_query)
                if emb:
                    embeddings.append(emb)
                    await asyncio.sleep(0.1)  # Small delay between calls
                else:
                    evidence.append(f"🚨 INCONSISTENCY: Failed to get embedding on attempt {i+1}")
                    
            except EmbeddingServiceError as e:
                evidence.append(f"🚨 INCONSISTENCY: Embedding service error on attempt {i+1}: {str(e)}")
                logger.error(f"Embedding service error on attempt {i+1}: {e}")
                
        if len(embeddings) >= 2:
            try:
                # Check if repeated calls return identical results
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = self.cosine_similarity(embeddings[i], embeddings[i+1])
                    similarities.append(sim)
                    
                avg_similarity = np.mean(similarities)
                logger.info(f"🔬 Average similarity between repeated calls: {avg_similarity:.6f}")
                
                if avg_similarity < 0.999:
                    evidence.append(f"🚨 INCONSISTENCY: Repeated embedding calls have similarity {avg_similarity:.6f} < 0.999")
                    
            except (ValueError, TypeError, ZeroDivisionError) as e:
                evidence.append(f"🚨 INCONSISTENCY: Mathematical error during consistency analysis: {str(e)}")
                logger.error(f"Mathematical error during consistency analysis: {e}")
                
        is_deceptive = len(evidence) > 0
        confidence = min(len(evidence) * 0.3, 1.0) if is_deceptive else 0.6
        
        return DeceptionEvidence(
            test_name="Consistency and Cross-validation",
            is_deceptive=is_deceptive,
            confidence=confidence,
            evidence=evidence,
            raw_data={"repeated_calls": len(embeddings)}
        )
        
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from embedding service with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.embedding_url,
                    json={"text": text, "dimension": 512},
                    timeout=self.request_timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get("embedding", [])
                    if not embedding:
                        raise EmbeddingServiceError(f"Empty embedding returned for text: '{text}'")
                    return embedding
                else:
                    raise EmbeddingServiceError(f"HTTP {response.status_code}: {response.text}")
                    
            except Timeout as e:
                if attempt == self.max_retries - 1:
                    raise EmbeddingServiceError(f"Timeout after {self.max_retries} attempts: {str(e)}")
                logger.warning(f"⚠️ Embedding timeout attempt {attempt + 1}/{self.max_retries}")
                await asyncio.sleep(1)
                
            except ConnectionError as e:
                if attempt == self.max_retries - 1:
                    raise EmbeddingServiceError(f"Connection error after {self.max_retries} attempts: {str(e)}")
                logger.warning(f"⚠️ Embedding connection error attempt {attempt + 1}/{self.max_retries}")
                await asyncio.sleep(2)
                
            except RequestException as e:
                if attempt == self.max_retries - 1:
                    raise EmbeddingServiceError(f"Request error after {self.max_retries} attempts: {str(e)}")
                logger.warning(f"⚠️ Embedding request error attempt {attempt + 1}/{self.max_retries}: {e}")
                await asyncio.sleep(1)
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                raise EmbeddingServiceError(f"Invalid response format: {str(e)}")
                
        return None
        
    async def search_memories(self, query: str, limit: int = 10) -> Optional[List[Dict]]:
        """Search memories via API with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.memory_url}/memories/search",
                    json={"query": query, "limit": limit},
                    timeout=self.request_timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("memories", [])
                else:
                    raise MemorySearchError(f"HTTP {response.status_code}: {response.text}")
                    
            except Timeout as e:
                if attempt == self.max_retries - 1:
                    raise MemorySearchError(f"Search timeout after {self.max_retries} attempts: {str(e)}")
                logger.warning(f"⚠️ Search timeout attempt {attempt + 1}/{self.max_retries}")
                await asyncio.sleep(1)
                
            except ConnectionError as e:
                if attempt == self.max_retries - 1:
                    raise MemorySearchError(f"Search connection error after {self.max_retries} attempts: {str(e)}")
                logger.warning(f"⚠️ Search connection error attempt {attempt + 1}/{self.max_retries}")
                await asyncio.sleep(2)
                
            except RequestException as e:
                if attempt == self.max_retries - 1:
                    raise MemorySearchError(f"Search request error after {self.max_retries} attempts: {str(e)}")
                logger.warning(f"⚠️ Search request error attempt {attempt + 1}/{self.max_retries}: {e}")
                await asyncio.sleep(1)
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                raise MemorySearchError(f"Invalid search response format: {str(e)}")
                
        return []
        
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not a or not b or len(a) != len(b):
            raise ValueError(f"Invalid vectors for similarity calculation: len(a)={len(a) if a else 0}, len(b)={len(b) if b else 0}")
            
        try:
            a_np = np.array(a, dtype=np.float64)
            b_np = np.array(b, dtype=np.float64)
            
            dot_product = np.dot(a_np, b_np)
            norm_a = np.linalg.norm(a_np) 
            norm_b = np.linalg.norm(b_np)
            
            if norm_a == 0 or norm_b == 0:
                raise ValueError("Zero-norm vector encountered")
                
            similarity = dot_product / (norm_a * norm_b)
            
            # Clamp to [-1, 1] to handle numerical precision issues
            return float(np.clip(similarity, -1.0, 1.0))
            
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as e:
            raise ValueError(f"Mathematical error in cosine similarity calculation: {str(e)}")
        
    def compile_investigation_report(self, evidence_list: List[DeceptionEvidence]) -> Dict[str, Any]:
        """Compile comprehensive investigation report"""
        
        if not evidence_list:
            raise ForensicInvestigationError("No evidence collected - investigation incomplete")
        
        try:
            total_deceptions = sum(1 for e in evidence_list if e.is_deceptive)
            confidence_values = [e.confidence for e in evidence_list if 0 <= e.confidence <= 1]
            
            if not confidence_values:
                raise ForensicInvestigationError("No valid confidence values found")
                
            overall_confidence = float(np.mean(confidence_values))
            all_evidence = []
            
            for evidence in evidence_list:
                if evidence.evidence:
                    all_evidence.extend(evidence.evidence)
                    
            report = {
                "investigation_timestamp": time.time(),
                "magnification_level": f"x{self.magnification_level}%",
                "overall_status": "DECEPTION_DETECTED" if total_deceptions > 0 else "SYSTEM_AUTHENTIC",
                "deception_count": total_deceptions,
                "total_tests": len(evidence_list),
                "overall_confidence": overall_confidence,
                "evidence_summary": {
                    "total_evidence_items": len(all_evidence),
                    "critical_issues": [e for e in all_evidence if "CRITICAL" in e],
                    "deception_issues": [e for e in all_evidence if "DECEPTION" in e], 
                    "manipulation_issues": [e for e in all_evidence if "MANIPULATION" in e],
                },
                "test_results": [
                    {
                        "test_name": e.test_name,
                        "is_deceptive": e.is_deceptive,
                        "confidence": e.confidence,
                        "evidence_count": len(e.evidence),
                        "evidence": e.evidence,
                        "raw_data": e.raw_data
                    }
                    for e in evidence_list
                ]
            }
            
            return report
            
        except (ValueError, TypeError, ArithmeticError) as e:
            raise ForensicInvestigationError(f"Error compiling investigation report: {str(e)}")
        
    def print_investigation_report(self, report: Dict[str, Any]):
        """Print detailed investigation report"""
        
        print("\n" + "="*80)
        print(f"🔍 FORENSIC INVESTIGATION REPORT x{self.magnification_level}%")
        print("="*80)
        print(f"📅 Timestamp: {time.ctime(report['investigation_timestamp'])}")
        print(f"🔬 Magnification Level: {report['magnification_level']}")
        print(f"🎯 Overall Status: {report['overall_status']}")
        print(f"⚠️  Deceptions Detected: {report['deception_count']} / {report['total_tests']}")
        print(f"📊 Overall Confidence: {report['overall_confidence']:.3f}")
        print()
        
        # Evidence Summary
        evidence_summary = report["evidence_summary"]
        print("📋 EVIDENCE SUMMARY:")
        print(f"  Total Evidence Items: {evidence_summary['total_evidence_items']}")
        print(f"  Critical Issues: {len(evidence_summary['critical_issues'])}")
        print(f"  Deception Issues: {len(evidence_summary['deception_issues'])}")
        print(f"  Manipulation Issues: {len(evidence_summary['manipulation_issues'])}")
        print()
        
        # Detailed Test Results
        print("🔍 DETAILED TEST RESULTS:")
        print("-" * 80)
        
        for i, test in enumerate(report["test_results"], 1):
            status = "❌ DECEPTIVE" if test["is_deceptive"] else "✅ AUTHENTIC"
            print(f"{i}. {test['test_name']}: {status} (confidence: {test['confidence']:.3f})")
            
            if test["evidence"]:
                print(f"   Evidence found ({len(test['evidence'])} items):")
                for evidence in test["evidence"]:
                    print(f"   • {evidence}")
                print()
        
        # Final Verdict
        print("=" * 80)
        if report['overall_status'] == "DECEPTION_DETECTED":
            print("🚨 FINAL VERDICT: DECEPTION DETECTED IN AI MEMORY SERVICE")
            print(f"   {report['deception_count']} out of {report['total_tests']} tests found deceptive behavior")
            print("   System requires immediate investigation and fixes")
        else:
            print("✅ FINAL VERDICT: SYSTEM APPEARS AUTHENTIC")
            print("   No significant deception patterns detected")
            print("   System functioning within expected parameters")
        print("=" * 80)


async def main():
    """Main forensic investigation entry point"""
    
    investigator = ForensicInvestigatorX200000()
    exit_code = 0
    
    try:
        # Check if embedding server is running
        logger.info("🔌 Checking embedding server availability...")
        try:
            response = requests.get("http://localhost:8090/health", timeout=5)
            if response.status_code != 200:
                logger.warning("⚠️  Embedding server health check failed, but continuing investigation...")
        except RequestException:
            logger.warning("⚠️  Cannot reach embedding server at localhost:8090, but continuing investigation...")
        
        # Perform the investigation
        report = await investigator.investigate_deception()
        
        # Print detailed report
        investigator.print_investigation_report(report)
        
        # Save report to file
        report_file = f"forensic_report_x{investigator.magnification_level}_percent.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"📄 Full report saved to: {report_file}")
        
        # Set appropriate exit code
        if report['overall_status'] == "DECEPTION_DETECTED":
            exit_code = 1  # Indicate deception found
        else:
            exit_code = 0  # System appears authentic
            
    except ForensicInvestigationError as e:
        logger.error(f"💥 FORENSIC INVESTIGATION ERROR: {e}")
        print(f"\n🚨 INVESTIGATION ERROR: {e}")
        exit_code = 2
        
    except RequestException as e:
        logger.error(f"💥 NETWORK ERROR DURING INVESTIGATION: {e}")
        print(f"\n🚨 NETWORK ERROR: {e}")
        exit_code = 3
        
    except (OSError, IOError) as e:
        logger.error(f"💥 FILE SYSTEM ERROR DURING INVESTIGATION: {e}")
        print(f"\n🚨 FILE SYSTEM ERROR: {e}")
        exit_code = 4
        
    except KeyboardInterrupt:
        logger.info("Investigation interrupted by user")
        print("\n🛑 Investigation interrupted by user")
        exit_code = 130
        
    except Exception as e:
        logger.error(f"💥 UNEXPECTED ERROR DURING INVESTIGATION: {e}")
        logger.debug("Full traceback:", exc_info=True)
        print(f"\n🚨 UNEXPECTED ERROR: {e}")
        print("Check the log file for detailed traceback information.")
        exit_code = 5
        
    finally:
        sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())