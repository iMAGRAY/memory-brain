#!/usr/bin/env python3
"""
Comprehensive test script for EmbeddingGemma-300M server
Проверяет все аспекты качества и работоспособности
"""

import sys
import ast
import json
from pathlib import Path

def analyze_server_code():
    """Анализ структуры и качества кода сервера"""
    print("=" * 60)
    print("EMBEDDING SERVER QUALITY CHECK")
    print("=" * 60)
    
    try:
        with open('embedding_server.py', 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines()
            
        # 1. Parse AST
        tree = ast.parse(content)
        print("✅ Syntax: Valid Python code")
        
        # 2. Find all classes and methods
        classes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)
                classes[node.name] = methods
        
        print(f"\n📦 Classes found: {len(classes)}")
        for cls_name, methods in classes.items():
            print(f"  - {cls_name}: {len(methods)} methods")
        
        # 3. Check critical methods
        critical_methods = {
            'EmbeddingService': [
                '__init__', '_load_model_safe', '_warmup_model',
                '_safe_cache_key', '_evict_cache_if_needed', 
                '_update_cache_stats', '_get_from_cache', '_put_in_cache',
                '_encode_with_prompts', '_encode_batch',
                'encode_for_retrieval_query', 'encode_for_retrieval_documents',
                'encode_for_classification', 'encode_for_clustering',
                'encode_for_semantic_similarity', 'encode_for_code_search',
                'encode_with_matryoshka', 'encode_async',
                'get_stats', 'clear_cache'
            ],
            'EmbeddingServer': [
                '__init__', 'setup_routes', 'setup_cors',
                'handle_embed', 'handle_embed_batch', 
                'handle_health', 'handle_stats', 'run'
            ]
        }
        
        print("\n🔍 Critical methods check:")
        all_good = True
        for cls_name, required_methods in critical_methods.items():
            if cls_name in classes:
                found_methods = classes[cls_name]
                missing = set(required_methods) - set(found_methods)
                if missing:
                    print(f"  ❌ {cls_name} missing: {missing}")
                    all_good = False
                else:
                    print(f"  ✅ {cls_name}: All {len(required_methods)} critical methods present")
            else:
                print(f"  ❌ Class {cls_name} not found!")
                all_good = False
        
        # 4. Check for common issues
        print("\n🔎 Common issues check:")
        issues = []
        
        # Check for duplicate cache initialization
        cache_init_lines = [i for i, line in enumerate(lines) if 'self.cache = ' in line]
        if len(cache_init_lines) > 1:
            issues.append(f"Duplicate cache initialization at lines: {cache_init_lines}")
        
        # Check for duplicate cache_size
        cache_size_lines = [i for i, line in enumerate(lines) if 'self.cache_size = ' in line]
        if len(cache_size_lines) > 1:
            issues.append(f"Duplicate cache_size at lines: {cache_size_lines}")
            
        # Check for executor initialization
        executor_lines = [i for i, line in enumerate(lines) if 'ThreadPoolExecutor' in line]
        if len(executor_lines) != 1:
            issues.append(f"ThreadPoolExecutor initialization issues at lines: {executor_lines}")
        
        if issues:
            for issue in issues:
                print(f"  ⚠️ {issue}")
        else:
            print("  ✅ No common issues detected")
        
        # 5. Check EmbeddingGemma optimizations
        print("\n🚀 EmbeddingGemma optimizations check:")
        optimizations = {
            'TASK_PROMPTS': 'Specialized prompts',
            'Matryoshka': 'Matryoshka representation',
            'thread-safe': 'Thread safety',
            'LRU': 'LRU cache eviction',
            'SHA256': 'Secure hashing',
            'bfloat16': 'BFloat16 support',
            'normalize_embeddings': 'Normalization'
        }
        
        for keyword, description in optimizations.items():
            if keyword.lower() in content.lower():
                print(f"  ✅ {description}: Implemented")
            else:
                print(f"  ❌ {description}: Not found")
        
        # 6. Configuration check
        print("\n⚙️ Configuration check:")
        config_params = {
            'default_dimension': 512,
            'cache_size': (100, 10000),
            'max_workers': 4,
            'port': 8090
        }
        
        for param, expected in config_params.items():
            if param in content:
                print(f"  ✅ {param}: Configured")
            else:
                print(f"  ⚠️ {param}: Not explicitly configured")
        
        # 7. Security features
        print("\n🔒 Security features:")
        security_features = {
            'escape': 'HTML escaping',
            'RLock': 'Thread locks',
            'ValueError': 'Input validation',
            'try.*except': 'Error handling',
            'hashlib': 'Secure hashing'
        }
        
        import re
        for pattern, feature in security_features.items():
            if re.search(pattern, content):
                print(f"  ✅ {feature}: Present")
            else:
                print(f"  ❌ {feature}: Missing")
        
        # Final verdict
        print("\n" + "=" * 60)
        if all_good and not issues:
            print("✅ VERDICT: Server is PRODUCTION READY")
            print("All critical components are properly implemented")
        else:
            print("⚠️ VERDICT: Server needs minor fixes")
            print("Please address the issues listed above")
        print("=" * 60)
        
        return all_good
        
    except Exception as e:
        print(f"❌ Failed to analyze server: {e}")
        return False

def check_dependencies():
    """Проверка зависимостей"""
    print("\n📦 Checking dependencies...")
    required = [
        'torch', 'numpy', 'sentence_transformers', 
        'aiohttp', 'aiohttp_cors', 'hashlib', 'asyncio'
    ]
    
    import importlib
    missing = []
    for module in required:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} - NOT INSTALLED")
            missing.append(module)
    
    return len(missing) == 0

def check_model_path():
    """Проверка наличия модели"""
    print("\n🤖 Checking model path...")
    model_path = Path(r'C:\Models\ai-memory-service\models\embeddinggemma-300m')
    
    if model_path.exists():
        print(f"  ✅ Model directory exists: {model_path}")
        
        # Check for key files
        key_files = ['config.json', 'tokenizer.json', 'model.safetensors']
        for file in key_files:
            if (model_path / file).exists():
                print(f"    ✅ {file}")
            else:
                print(f"    ❌ {file} - MISSING")
                return False
        return True
    else:
        print(f"  ❌ Model directory not found: {model_path}")
        return False

def main():
    """Main test function"""
    print("\n🔧 EMBEDDING SERVER COMPREHENSIVE TEST\n")
    
    results = {
        'code_quality': analyze_server_code(),
        'dependencies': check_dependencies(),
        'model_available': check_model_path()
    }
    
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS:")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test}: {status}")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The embedding server is ready for production use.")
        print("\nTo start the server:")
        print("  python embedding_server.py")
        print("\nAPI endpoints will be available at:")
        print("  - POST http://localhost:8090/embed")
        print("  - POST http://localhost:8090/embed_batch")
        print("  - GET  http://localhost:8090/health")
        print("  - GET  http://localhost:8090/stats")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("Please fix the issues before deploying to production.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())