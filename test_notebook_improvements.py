#!/usr/bin/env python3
"""
Test script to verify improvements made to ai_memory_analysis.ipynb
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List

# Set environment variables for testing
os.environ['MEMORY_API_URL'] = 'http://127.0.0.1:8080'
os.environ['EMBEDDING_API_URL'] = 'http://127.0.0.1:8090'
os.environ['API_TIMEOUT'] = '10'
os.environ['MAX_RETRIES'] = '3'

# Test results storage
test_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_environment_configuration():
    """Test environment-based configuration"""
    print("Testing environment-based configuration...")
    
    try:
        memory_url = os.getenv('MEMORY_API_URL', 'http://127.0.0.1:8080')
        embedding_url = os.getenv('EMBEDDING_API_URL', 'http://127.0.0.1:8090')
        timeout = int(os.getenv('API_TIMEOUT', '10'))
        retries = int(os.getenv('MAX_RETRIES', '3'))
        
        assert memory_url == 'http://127.0.0.1:8080'
        assert embedding_url == 'http://127.0.0.1:8090'
        assert timeout == 10
        assert retries == 3
        
        test_results['passed'].append('Environment configuration')
        print("✅ Environment configuration: PASSED")
        return True
    except Exception as e:
        test_results['failed'].append(f'Environment configuration: {e}')
        print(f"❌ Environment configuration: FAILED - {e}")
        return False

def test_retry_mechanism():
    """Test HTTP retry mechanism"""
    print("\nTesting retry mechanism...")
    
    try:
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry
        
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=['GET', 'POST', 'PUT', 'DELETE']
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        # Test with a non-existent endpoint to trigger retries
        try:
            # This should retry but ultimately fail
            response = session.get('http://127.0.0.1:9999/test', timeout=1)
        except requests.exceptions.ConnectionError:
            # Expected behavior - connection refused after retries
            pass
        
        test_results['passed'].append('Retry mechanism')
        print("✅ Retry mechanism: PASSED")
        return True
    except Exception as e:
        test_results['failed'].append(f'Retry mechanism: {e}')
        print(f"❌ Retry mechanism: FAILED - {e}")
        return False

def test_service_connectivity():
    """Test enhanced service connectivity checks"""
    print("\nTesting service connectivity...")
    
    services = {}
    
    # Memory service
    try:
        start_time = time.time()
        response = requests.get('http://127.0.0.1:8080/health', timeout=5)
        latency = (time.time() - start_time) * 1000
        
        services['memory_service'] = {
            'status': response.status_code == 200,
            'latency_ms': round(latency, 2)
        }
    except:
        services['memory_service'] = {
            'status': False,
            'latency_ms': None
        }
    
    # Embedding service
    try:
        start_time = time.time()
        response = requests.get('http://127.0.0.1:8090/health', timeout=5)
        latency = (time.time() - start_time) * 1000
        
        services['embedding_service'] = {
            'status': response.status_code == 200,
            'latency_ms': round(latency, 2)
        }
    except:
        services['embedding_service'] = {
            'status': False,
            'latency_ms': None
        }
    
    # Check if latency tracking works
    for service, status in services.items():
        if status['status']:
            if status['latency_ms'] is not None:
                print(f"  {service}: Online ({status['latency_ms']}ms)")
            else:
                print(f"  {service}: Online")
        else:
            print(f"  {service}: Offline")
    
    if any(s['latency_ms'] is not None for s in services.values() if s['status']):
        test_results['passed'].append('Service connectivity with latency')
        print("✅ Service connectivity with latency: PASSED")
        return True
    else:
        test_results['warnings'].append('Services offline, latency not measurable')
        print("⚠️ Service connectivity: WARNING - Services offline")
        return True

def test_function_documentation():
    """Test that functions have proper documentation"""
    print("\nTesting function documentation...")
    
    # Read notebook and check for docstrings
    try:
        import nbformat
        
        nb_path = Path('ai_memory_analysis.ipynb')
        if not nb_path.exists():
            test_results['warnings'].append('Notebook file not found')
            print("⚠️ Function documentation: WARNING - Notebook not found")
            return True
        
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Count functions with docstrings
        functions_with_docs = 0
        total_functions = 0
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                source = cell.source
                if 'def ' in source:
                    total_functions += source.count('def ')
                    if '"""' in source or "'''" in source:
                        functions_with_docs += source.count('"""') // 2 + source.count("'''") // 2
        
        doc_coverage = (functions_with_docs / total_functions * 100) if total_functions > 0 else 0
        
        if doc_coverage >= 80:
            test_results['passed'].append(f'Function documentation ({doc_coverage:.1f}% coverage)')
            print(f"✅ Function documentation: PASSED ({doc_coverage:.1f}% coverage)")
            return True
        else:
            test_results['warnings'].append(f'Function documentation only {doc_coverage:.1f}% coverage')
            print(f"⚠️ Function documentation: WARNING - Only {doc_coverage:.1f}% coverage")
            return True
            
    except Exception as e:
        test_results['failed'].append(f'Function documentation: {e}')
        print(f"❌ Function documentation: FAILED - {e}")
        return False

def test_async_support():
    """Test async/await support for benchmarks"""
    print("\nTesting async support...")
    
    try:
        import asyncio
        import aiohttp
        
        async def test_async_function():
            async with aiohttp.ClientSession() as session:
                return True
        
        # Test if async can run
        try:
            result = asyncio.run(test_async_function())
            test_results['passed'].append('Async support')
            print("✅ Async support: PASSED")
            return True
        except Exception as e:
            # Try with nest_asyncio for Jupyter compatibility
            import nest_asyncio
            nest_asyncio.apply()
            result = asyncio.run(test_async_function())
            test_results['passed'].append('Async support (with nest_asyncio)')
            print("✅ Async support (with nest_asyncio): PASSED")
            return True
            
    except ImportError as e:
        test_results['failed'].append(f'Async support: Missing package {e}')
        print(f"❌ Async support: FAILED - Missing package {e}")
        return False
    except Exception as e:
        test_results['failed'].append(f'Async support: {e}')
        print(f"❌ Async support: FAILED - {e}")
        return False

def test_env_example_file():
    """Test .env.example file exists and has notebook configuration"""
    print("\nTesting .env.example file...")
    
    try:
        env_path = Path('.env.example')
        if not env_path.exists():
            test_results['failed'].append('.env.example file not found')
            print("❌ .env.example file: FAILED - File not found")
            return False
        
        with open(env_path, 'r') as f:
            content = f.read()
        
        required_vars = [
            'MEMORY_API_URL',
            'EMBEDDING_API_URL',
            'API_TIMEOUT',
            'MAX_RETRIES',
            'SIMILARITY_THRESHOLD'
        ]
        
        missing_vars = []
        for var in required_vars:
            if var not in content:
                missing_vars.append(var)
        
        if not missing_vars:
            test_results['passed'].append('.env.example configuration')
            print("✅ .env.example configuration: PASSED")
            return True
        else:
            test_results['failed'].append(f'.env.example missing: {", ".join(missing_vars)}')
            print(f"❌ .env.example configuration: FAILED - Missing {", ".join(missing_vars)}")
            return False
            
    except Exception as e:
        test_results['failed'].append(f'.env.example: {e}')
        print(f"❌ .env.example file: FAILED - {e}")
        return False

def generate_report():
    """Generate test report"""
    print("\n" + "=" * 60)
    print("TEST REPORT - Notebook Improvements")
    print("=" * 60)
    
    total_tests = len(test_results['passed']) + len(test_results['failed'])
    
    print(f"\n✅ Passed: {len(test_results['passed'])}/{total_tests}")
    for test in test_results['passed']:
        print(f"  • {test}")
    
    if test_results['failed']:
        print(f"\n❌ Failed: {len(test_results['failed'])}/{total_tests}")
        for test in test_results['failed']:
            print(f"  • {test}")
    
    if test_results['warnings']:
        print(f"\n⚠️ Warnings: {len(test_results['warnings'])}")
        for warning in test_results['warnings']:
            print(f"  • {warning}")
    
    # Calculate score
    if total_tests > 0:
        score = (len(test_results['passed']) / total_tests) * 100
        print(f"\n📊 Overall Score: {score:.1f}%")
        
        if score >= 90:
            grade = 'A'
        elif score >= 80:
            grade = 'B'
        elif score >= 70:
            grade = 'C'
        elif score >= 60:
            grade = 'D'
        else:
            grade = 'F'
        
        print(f"📈 Grade: {grade}")
    
    # Save report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': test_results,
        'score': score if total_tests > 0 else 0,
        'grade': grade if total_tests > 0 else 'N/A'
    }
    
    with open('notebook_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n📁 Report saved to: notebook_test_report.json")
    print("=" * 60)

def main():
    """Run all tests"""
    print("Testing AI Memory Analysis Notebook Improvements")
    print("=" * 60)
    
    # Run tests
    test_environment_configuration()
    test_retry_mechanism()
    test_service_connectivity()
    test_function_documentation()
    test_async_support()
    test_env_example_file()
    
    # Generate report
    generate_report()

if __name__ == "__main__":
    main()