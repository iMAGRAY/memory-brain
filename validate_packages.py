#!/usr/bin/env python3
"""
AI Memory Service - Python Package Validation Script
Validates installed Python packages and their core functionality
"""

import sys
import traceback

def validate_package(name, import_func, test_func=None):
    """Validate a package import and optionally test functionality"""
    try:
        package = import_func()
        print(f"✅ {name}: Successfully imported")
        
        if hasattr(package, '__version__'):
            print(f"   Version: {package.__version__}")
        
        if test_func:
            test_func(package)
            print(f"   ✅ Functionality test passed")
            
        return True
    except ImportError as e:
        print(f"❌ {name}: Import failed - {e}")
        return False
    except Exception as e:
        print(f"❌ {name}: Test failed - {e}")
        return False

def test_torch(torch):
    """Test PyTorch basic functionality"""
    import torch.nn as nn
    model = nn.Linear(10, 1)
    test_input = torch.randn(1, 10)
    output = model(test_input)
    assert output.shape == (1, 1), f"Expected shape (1, 1), got {output.shape}"

def test_transformers(transformers):
    """Test transformers basic functionality"""
    from transformers import AutoTokenizer
    # Just test that we can import the class
    assert AutoTokenizer is not None

def main():
    """Main validation routine"""
    print("🔍 AI Memory Service - Package Validation")
    print("=" * 50)
    
    success_count = 0
    total_tests = 0
    
    # Core packages to validate
    packages = [
        ("PyTorch", lambda: __import__('torch'), test_torch),
        ("Transformers", lambda: __import__('transformers'), test_transformers),
        ("SentenceTransformers", lambda: __import__('sentence_transformers'), None),
        ("NumPy", lambda: __import__('numpy'), None),
        ("Flask", lambda: __import__('flask'), None),
        ("Requests", lambda: __import__('requests'), None),
        ("Gunicorn", lambda: __import__('gunicorn'), None),
    ]
    
    for name, import_func, test_func in packages:
        total_tests += 1
        if validate_package(name, import_func, test_func):
            success_count += 1
        print()
    
    print("=" * 50)
    print(f"📊 Validation Results: {success_count}/{total_tests} packages validated")
    
    if success_count == total_tests:
        print("🎉 All packages validated successfully!")
        sys.exit(0)
    else:
        print("❌ Some packages failed validation")
        sys.exit(1)

if __name__ == "__main__":
    main()