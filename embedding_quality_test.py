#!/usr/bin/env python3
"""
Comprehensive Embedding Quality Test
Проверка реального качества и точности embeddings от EmbeddingGemma-300M
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import json

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent))

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingQualityTest:
    def __init__(self):
        self.service = None
        self.results = {
            'semantic_similarity_tests': {},
            'clustering_quality': {},
            'retrieval_accuracy': {},
            'benchmark_comparisons': {},
            'edge_case_handling': {},
            'performance_metrics': {}
        }
        
    def setup_service(self):
        """Initialize embedding service"""
        try:
            from embedding_server import EmbeddingService
            model_path = self._get_model_path()
            
            logger.info(f"Initializing EmbeddingGemma-300M from {model_path}")
            self.service = EmbeddingService(model_path=model_path, cache_size=100)
            
            # Warmup
            self.service.encode_query("warmup test")
            logger.info("Service initialized and warmed up")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            return False
    
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
        
        raise FileNotFoundError("EmbeddingGemma-300M model not found")
    
    def test_semantic_similarity_pairs(self) -> Dict[str, Any]:
        """Test 1: Semantic similarity between text pairs"""
        logger.info("=== TEST 1: SEMANTIC SIMILARITY PAIRS ===")
        
        # High similarity pairs (should have cosine similarity > 0.7)
        high_similarity_pairs = [
            ("The cat is sleeping on the couch", "A cat rests on the sofa"),
            ("I love programming in Python", "Python is my favorite programming language"), 
            ("The weather is very cold today", "It's extremely chilly outside today"),
            ("She bought a new car", "She purchased a new automobile"),
            ("The movie was fantastic", "The film was excellent"),
            ("I need to go to the doctor", "I should visit the physician"),
            ("The book is on the table", "There's a book sitting on the table"),
            ("He's learning machine learning", "He studies artificial intelligence")
        ]
        
        # Medium similarity pairs (should have cosine similarity 0.3-0.7)
        medium_similarity_pairs = [
            ("The cat is sleeping", "The dog is running"),
            ("I like pizza", "She enjoys pasta"),
            ("Programming is fun", "Reading books is enjoyable"),
            ("The weather is cold", "The sun is shining"),
            ("Cars are expensive", "Houses cost a lot"),
            ("I work in technology", "She teaches mathematics")
        ]
        
        # Low similarity pairs (should have cosine similarity < 0.3)
        low_similarity_pairs = [
            ("The cat is sleeping", "Quantum physics equations"),
            ("I love pizza", "Database optimization techniques"),
            ("Beautiful sunset today", "Binary search algorithm"),
            ("Playing guitar music", "Chemical reaction formula"),
            ("Traveling to Paris", "Protein synthesis process"),
            ("Cooking dinner tonight", "Stock market analysis")
        ]
        
        def evaluate_pairs(pairs: List[Tuple[str, str]], expected_range: Tuple[float, float], category: str):
            similarities = []
            correct_predictions = 0
            
            for text1, text2 in pairs:
                emb1 = self.service.encode_query(text1)
                emb2 = self.service.encode_query(text2)
                
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                similarities.append(similarity)
                
                # Check if similarity is in expected range
                if expected_range[0] <= similarity <= expected_range[1]:
                    correct_predictions += 1
                    
                logger.info(f"{category}: '{text1[:30]}...' vs '{text2[:30]}...' = {similarity:.3f}")
            
            accuracy = correct_predictions / len(pairs)
            avg_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            
            return {
                'accuracy': accuracy,
                'avg_similarity': avg_similarity,
                'std_similarity': std_similarity,
                'similarities': similarities,
                'correct_predictions': correct_predictions,
                'total_pairs': len(pairs)
            }
        
        # Evaluate all categories
        high_results = evaluate_pairs(high_similarity_pairs, (0.7, 1.0), "HIGH")
        medium_results = evaluate_pairs(medium_similarity_pairs, (0.3, 0.7), "MEDIUM") 
        low_results = evaluate_pairs(low_similarity_pairs, (0.0, 0.3), "LOW")
        
        # Overall assessment
        overall_accuracy = (
            high_results['correct_predictions'] + 
            medium_results['correct_predictions'] + 
            low_results['correct_predictions']
        ) / (
            high_results['total_pairs'] + 
            medium_results['total_pairs'] + 
            low_results['total_pairs']
        )
        
        results = {
            'high_similarity': high_results,
            'medium_similarity': medium_results,
            'low_similarity': low_results,
            'overall_accuracy': overall_accuracy
        }
        
        logger.info(f"HIGH similarity accuracy: {high_results['accuracy']:.1%}")
        logger.info(f"MEDIUM similarity accuracy: {medium_results['accuracy']:.1%}")
        logger.info(f"LOW similarity accuracy: {low_results['accuracy']:.1%}")
        logger.info(f"OVERALL accuracy: {overall_accuracy:.1%}")
        
        return results
    
    def test_clustering_quality(self) -> Dict[str, Any]:
        """Test 2: Clustering quality - do similar texts cluster together?"""
        logger.info("=== TEST 2: CLUSTERING QUALITY ===")
        
        # Define text clusters by topic
        clusters = {
            'technology': [
                "Artificial intelligence is advancing rapidly",
                "Machine learning models are becoming more sophisticated", 
                "Neural networks process complex data patterns",
                "Deep learning revolutionizes computer vision",
                "AI algorithms solve complex problems"
            ],
            'cooking': [
                "Baking bread requires precise timing",
                "Fresh ingredients make delicious meals",
                "Cooking pasta al dente is an art",
                "Grilling vegetables brings out natural flavors", 
                "Homemade pizza beats restaurant quality"
            ],
            'travel': [
                "Exploring new cultures enriches the soul",
                "Mountain hiking offers breathtaking views",
                "Beach vacations provide perfect relaxation",
                "City tours reveal hidden historical gems",
                "Adventure travel creates lifelong memories"
            ],
            'health': [
                "Regular exercise improves cardiovascular health",
                "Balanced nutrition supports immune system",
                "Adequate sleep enhances cognitive function",
                "Meditation reduces stress and anxiety",
                "Hydration is essential for body function"
            ]
        }
        
        # Generate embeddings for all texts
        all_embeddings = []
        all_labels = []
        
        for cluster_name, texts in clusters.items():
            for text in texts:
                embedding = self.service.encode_document(text)
                all_embeddings.append(embedding)
                all_labels.append(cluster_name)
        
        all_embeddings = np.array(all_embeddings)
        
        # Calculate intra-cluster vs inter-cluster similarities
        cluster_results = {}
        
        for i, cluster_name in enumerate(clusters.keys()):
            cluster_indices = [j for j, label in enumerate(all_labels) if label == cluster_name]
            cluster_embeddings = all_embeddings[cluster_indices]
            
            # Intra-cluster similarity (within cluster)
            intra_similarities = []
            for i in range(len(cluster_embeddings)):
                for j in range(i+1, len(cluster_embeddings)):
                    sim = cosine_similarity([cluster_embeddings[i]], [cluster_embeddings[j]])[0][0]
                    intra_similarities.append(sim)
            
            # Inter-cluster similarity (with other clusters)
            inter_similarities = []
            other_indices = [j for j, label in enumerate(all_labels) if label != cluster_name]
            other_embeddings = all_embeddings[other_indices]
            
            for cluster_emb in cluster_embeddings:
                for other_emb in other_embeddings:
                    sim = cosine_similarity([cluster_emb], [other_emb])[0][0]
                    inter_similarities.append(sim)
            
            avg_intra = np.mean(intra_similarities) if intra_similarities else 0
            avg_inter = np.mean(inter_similarities) if inter_similarities else 0
            separation = avg_intra - avg_inter  # Good clustering = positive separation
            
            cluster_results[cluster_name] = {
                'avg_intra_similarity': avg_intra,
                'avg_inter_similarity': avg_inter,
                'separation': separation
            }
            
            logger.info(f"{cluster_name.upper()}: intra={avg_intra:.3f}, inter={avg_inter:.3f}, sep={separation:.3f}")
        
        # Overall clustering quality
        overall_separation = np.mean([result['separation'] for result in cluster_results.values()])
        
        results = {
            'cluster_results': cluster_results,
            'overall_separation': overall_separation,
            'quality_score': min(1.0, max(0.0, overall_separation * 2))  # Scale to 0-1
        }
        
        logger.info(f"Overall clustering separation: {overall_separation:.3f}")
        logger.info(f"Clustering quality score: {results['quality_score']:.1%}")
        
        return results
    
    def test_retrieval_accuracy(self) -> Dict[str, Any]:
        """Test 3: Information retrieval accuracy"""
        logger.info("=== TEST 3: RETRIEVAL ACCURACY ===")
        
        # Document collection
        documents = [
            "Python is a high-level programming language known for its simplicity and readability",
            "JavaScript enables dynamic web development and interactive user interfaces", 
            "Machine learning algorithms can identify patterns in large datasets automatically",
            "Deep neural networks consist of multiple layers that process information hierarchically",
            "Natural language processing helps computers understand and generate human language",
            "Computer vision systems can analyze and interpret visual information from images",
            "Cloud computing provides scalable infrastructure for modern applications",
            "Blockchain technology ensures secure and transparent transaction records",
            "Quantum computing leverages quantum mechanics for exponentially faster calculations",
            "Cybersecurity protects digital systems from unauthorized access and attacks"
        ]
        
        # Queries with expected relevant documents
        queries_and_expected = [
            ("programming language syntax", [0, 1]),  # Python, JavaScript
            ("artificial intelligence algorithms", [2, 3, 4, 5]),  # ML, DL, NLP, CV
            ("secure data systems", [7, 9]),  # Blockchain, Cybersecurity
            ("computational performance", [8, 6]),  # Quantum, Cloud
            ("web development tools", [1]),  # JavaScript
        ]
        
        # Generate document embeddings
        doc_embeddings = []
        for doc in documents:
            doc_emb = self.service.encode_document(doc)
            doc_embeddings.append(doc_emb)
        
        doc_embeddings = np.array(doc_embeddings)
        
        retrieval_results = []
        
        for query, expected_indices in queries_and_expected:
            # Get query embedding
            query_emb = self.service.encode_query(query)
            
            # Calculate similarities with all documents
            similarities = []
            for doc_emb in doc_embeddings:
                sim = cosine_similarity([query_emb], [doc_emb])[0][0]
                similarities.append(sim)
            
            # Get top-k retrieved documents
            top_k = len(expected_indices) + 2  # Retrieve a few extra
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Calculate precision and recall
            retrieved_relevant = len(set(top_indices) & set(expected_indices))
            precision = retrieved_relevant / min(top_k, len(top_indices))
            recall = retrieved_relevant / len(expected_indices)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            result = {
                'query': query,
                'expected_indices': expected_indices,
                'top_indices': top_indices.tolist(),
                'similarities': [similarities[i] for i in top_indices],
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            retrieval_results.append(result)
            
            logger.info(f"Query: '{query}'")
            top_sims = [similarities[i] for i in top_indices[:3]]
            logger.info(f"  Top docs: {top_indices[:3]} (sim: {[f'{s:.3f}' for s in top_sims]})")
            logger.info(f"  Expected: {expected_indices}")
            logger.info(f"  P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
        
        # Overall metrics
        avg_precision = np.mean([r['precision'] for r in retrieval_results])
        avg_recall = np.mean([r['recall'] for r in retrieval_results])
        avg_f1 = np.mean([r['f1'] for r in retrieval_results])
        
        results = {
            'retrieval_results': retrieval_results,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1
        }
        
        logger.info(f"Average Precision: {avg_precision:.1%}")
        logger.info(f"Average Recall: {avg_recall:.1%}")
        logger.info(f"Average F1: {avg_f1:.1%}")
        
        return results
    
    def test_multilingual_quality(self) -> Dict[str, Any]:
        """Test 4: Multilingual embedding quality"""
        logger.info("=== TEST 4: MULTILINGUAL QUALITY ===")
        
        # Parallel sentences in different languages
        parallel_sentences = [
            {
                'english': "The weather is beautiful today",
                'russian': "Погода сегодня прекрасная",
                'spanish': "El clima está hermoso hoy",
                'french': "Le temps est magnifique aujourd'hui"
            },
            {
                'english': "I love reading books",
                'russian': "Я люблю читать книги", 
                'spanish': "Me encanta leer libros",
                'french': "J'adore lire des livres"
            },
            {
                'english': "Technology is advancing rapidly",
                'russian': "Технологии быстро развиваются",
                'spanish': "La tecnología avanza rápidamente", 
                'french': "La technologie progresse rapidement"
            }
        ]
        
        multilingual_results = []
        
        for sentence_group in parallel_sentences:
            group_results = {}
            embeddings = {}
            
            # Generate embeddings for each language
            for lang, text in sentence_group.items():
                embeddings[lang] = self.service.encode_query(text)
            
            # Calculate cross-language similarities
            languages = list(sentence_group.keys())
            similarities = {}
            
            for i, lang1 in enumerate(languages):
                for j, lang2 in enumerate(languages):
                    if i < j:  # Avoid duplicates
                        sim = cosine_similarity([embeddings[lang1]], [embeddings[lang2]])[0][0]
                        pair = f"{lang1}-{lang2}"
                        similarities[pair] = sim
                        group_results[pair] = sim
            
            multilingual_results.append({
                'sentences': sentence_group,
                'similarities': group_results
            })
            
            logger.info(f"Sentence: '{sentence_group['english']}'")
            for pair, sim in group_results.items():
                logger.info(f"  {pair}: {sim:.3f}")
        
        # Calculate average cross-language similarity
        all_similarities = []
        for result in multilingual_results:
            all_similarities.extend(result['similarities'].values())
        
        avg_multilingual_similarity = np.mean(all_similarities)
        
        results = {
            'multilingual_results': multilingual_results,
            'avg_cross_language_similarity': avg_multilingual_similarity
        }
        
        logger.info(f"Average cross-language similarity: {avg_multilingual_similarity:.3f}")
        
        return results
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test 5: Edge cases and robustness"""
        logger.info("=== TEST 5: EDGE CASES AND ROBUSTNESS ===")
        
        edge_cases = {
            'empty_string': "",
            'single_word': "hello",
            'numbers_only': "123 456 789",
            'special_chars': "!@#$%^&*()_+-=[]{}|;:,.<>?",
            'very_short': "Hi",
            'repeated_word': "test test test test test",
            'mixed_languages': "Hello привет bonjour hola",
            'technical_terms': "API REST HTTP JSON SQL database schema normalization",
            'long_sentence': "This is a very long sentence that contains many different words and concepts to test how well the embedding model handles longer text inputs with complex semantic relationships and technical terminology throughout the entire passage.",
            'code_snippet': "def function(x): return x * 2 if x > 0 else 0"
        }
        
        edge_case_results = {}
        
        for case_name, text in edge_cases.items():
            try:
                start_time = time.perf_counter()
                embedding = self.service.encode_query(text)
                encoding_time = time.perf_counter() - start_time
                
                # Basic checks
                is_valid = embedding is not None and len(embedding) == 512
                has_nan = np.isnan(embedding).any() if is_valid else True
                has_inf = np.isinf(embedding).any() if is_valid else True
                norm = np.linalg.norm(embedding) if is_valid else 0
                
                edge_case_results[case_name] = {
                    'success': True,
                    'is_valid': is_valid,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'norm': norm,
                    'encoding_time': encoding_time,
                    'text_length': len(text)
                }
                
                status = "✅" if is_valid and not has_nan and not has_inf else "❌"
                logger.info(f"{status} {case_name}: norm={norm:.3f}, time={encoding_time*1000:.1f}ms")
                
            except Exception as e:
                edge_case_results[case_name] = {
                    'success': False,
                    'error': str(e),
                    'text_length': len(text)
                }
                logger.error(f"❌ {case_name}: ERROR - {e}")
        
        # Summary
        successful_cases = sum(1 for r in edge_case_results.values() if r['success'])
        total_cases = len(edge_case_results)
        success_rate = successful_cases / total_cases
        
        results = {
            'edge_case_results': edge_case_results,
            'success_rate': success_rate,
            'successful_cases': successful_cases,
            'total_cases': total_cases
        }
        
        logger.info(f"Edge cases success rate: {success_rate:.1%} ({successful_cases}/{total_cases})")
        
        return results
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality assessment report"""
        logger.info("=== GENERATING QUALITY REPORT ===")
        
        # Run all tests
        semantic_results = self.test_semantic_similarity_pairs()
        clustering_results = self.test_clustering_quality()
        retrieval_results = self.test_retrieval_accuracy()
        multilingual_results = self.test_multilingual_quality()
        edge_case_results = self.test_edge_cases()
        
        # Calculate overall quality scores
        scores = {
            'semantic_accuracy': semantic_results['overall_accuracy'],
            'clustering_quality': clustering_results['quality_score'],
            'retrieval_f1': retrieval_results['avg_f1'],
            'multilingual_similarity': min(1.0, multilingual_results['avg_cross_language_similarity']),
            'robustness': edge_case_results['success_rate']
        }
        
        # Weighted overall score
        weights = {
            'semantic_accuracy': 0.3,
            'clustering_quality': 0.25,
            'retrieval_f1': 0.25,
            'multilingual_similarity': 0.1,
            'robustness': 0.1
        }
        
        overall_score = sum(scores[metric] * weight for metric, weight in weights.items())
        
        # Quality assessment
        if overall_score >= 0.8:
            quality_rating = "EXCELLENT"
        elif overall_score >= 0.6:
            quality_rating = "GOOD"
        elif overall_score >= 0.4:
            quality_rating = "FAIR"
        else:
            quality_rating = "POOR"
        
        quality_report = {
            'overall_score': overall_score,
            'quality_rating': quality_rating,
            'component_scores': scores,
            'detailed_results': {
                'semantic_similarity': semantic_results,
                'clustering_quality': clustering_results, 
                'retrieval_accuracy': retrieval_results,
                'multilingual_quality': multilingual_results,
                'edge_case_robustness': edge_case_results
            }
        }
        
        return quality_report

def main():
    """Main test execution"""
    logger.info("🔍 STARTING COMPREHENSIVE EMBEDDING QUALITY TEST")
    logger.info("=" * 80)
    
    # Initialize test
    test = EmbeddingQualityTest()
    
    if not test.setup_service():
        logger.error("Failed to initialize embedding service")
        return False
    
    # Generate quality report
    start_time = time.perf_counter()
    
    try:
        quality_report = test.generate_quality_report()
        
        test_time = time.perf_counter() - start_time
        
        # Print final report
        logger.info("\n" + "=" * 80)
        logger.info("📊 FINAL EMBEDDING QUALITY REPORT")
        logger.info("=" * 80)
        
        logger.info(f"Overall Quality Score: {quality_report['overall_score']:.1%}")
        logger.info(f"Quality Rating: {quality_report['quality_rating']}")
        logger.info(f"Test Execution Time: {test_time:.1f}s")
        
        logger.info("\n📈 COMPONENT SCORES:")
        for metric, score in quality_report['component_scores'].items():
            logger.info(f"  {metric.replace('_', ' ').title()}: {score:.1%}")
        
        # Verdict
        logger.info("\n" + "=" * 80)
        if quality_report['overall_score'] >= 0.6:
            logger.info("✅ VERDICT: EMBEDDING QUALITY IS ACCEPTABLE FOR PRODUCTION USE")
            if quality_report['overall_score'] >= 0.8:
                logger.info("🌟 The embeddings show EXCELLENT semantic understanding")
            else:
                logger.info("👍 The embeddings show GOOD semantic understanding")
        else:
            logger.info("❌ VERDICT: EMBEDDING QUALITY NEEDS IMPROVEMENT")
            logger.info("⚠️ Consider using a different model or fine-tuning")
        
        logger.info("=" * 80)
        
        return quality_report['overall_score'] >= 0.6
        
    except Exception as e:
        logger.error(f"Quality test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)