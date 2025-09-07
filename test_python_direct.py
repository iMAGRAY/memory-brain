#!/usr/bin/env python3
"""
Direct test of Python EmbeddingGemma functionality
This bypasses Rust/PyO3 to test if the Python environment is working correctly
"""
import sys
import os

def main():
    print("=== ПРЯМОЙ ТЕСТ PYTHON EMBEDDINGGEMMA ===")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    
    try:
        # Test torch
        import torch
        print(f"✓ torch {torch.__version__} - CUDA: {torch.cuda.is_available()}")
        
        # Test numpy
        import numpy as np
        print(f"✓ numpy {np.__version__}")
        
        # Test sentence_transformers
        import sentence_transformers
        print(f"✓ sentence_transformers loaded")
        
        # Test model loading
        model_path = "C:/Models/ai-memory-service/models/embeddinggemma-300m"
        if os.path.exists(model_path):
            print(f"✓ Local model found: {model_path}")
            
            print("Loading EmbeddingGemma model...")
            model = sentence_transformers.SentenceTransformer(model_path)
            print("✓ Model loaded successfully!")
            
            # Test embedding generation
            test_text = "task: search result | query: test embedding"
            embedding = model.encode([test_text], normalize_embeddings=True)
            print(f"✓ Embedding generated: shape {embedding.shape}")
            print(f"✓ First 5 values: {embedding[0][:5]}")
            
            print("\n🎉 PYTHON EMBEDDINGGEMMA ПОЛНОСТЬЮ РАБОТАЕТ!")
            return True
            
        else:
            print(f"⚠ Local model not found at {model_path}")
            print("Testing with online model...")
            model = sentence_transformers.SentenceTransformer("google/embeddinggemma-300m")
            embedding = model.encode(["test"], normalize_embeddings=True)
            print(f"✓ Online model works: shape {embedding.shape}")
            return True
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)