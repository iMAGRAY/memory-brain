#!/usr/bin/env python3
"""
AI Memory Service - Python Embedding Service
Provides text embedding functionality using Sentence Transformers
"""
import os
import logging
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model instance
model = None

def detect_best_device():
    """Detect the best available device with GPU+CPU > GPU > CPU priority"""
    import torch
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🎮 GPU detected: {gpu_name} (devices: {gpu_count})")
        
        # Check if we can use GPU+CPU hybrid (MPS on Apple Silicon or CUDA)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🚀 Using Apple Metal Performance Shaders (GPU+CPU hybrid)")
            return 'mps'
        else:
            logger.info("🚀 Using CUDA GPU acceleration")
            return 'cuda'
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("🍎 Using Apple Silicon GPU (MPS)")
        return 'mps'
    else:
        logger.info("💻 Using CPU (no GPU acceleration available)")
        return 'cpu'

def load_model():
    """Load the sentence transformer model with optimal device selection"""
    global model
    
    # Get model configuration from environment
    model_name = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    trust_remote = os.environ.get('TRUST_REMOTE_CODE', 'false').lower() == 'true'
    
    try:
        # Detect best available device
        device = detect_best_device()
        
        logger.info(f"🎯 Target device: {device}")
        logger.info(f"🔒 Trust remote code: {trust_remote}")
        logger.info(f"📥 Loading model: {model_name}")
        
        # Load model with optimized settings
        model = SentenceTransformer(
            model_name, 
            device=device,
            trust_remote_code=trust_remote
        )
        
        # Validate model loaded correctly
        test_embedding = model.encode(["test"], show_progress_bar=False)
        embedding_dim = len(test_embedding[0])
        
        logger.info(f"✅ Model loaded successfully: {model_name}")
        logger.info(f"📊 Embedding dimension: {embedding_dim}")
        logger.info(f"🎯 Active device: {model.device}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(f"📋 Model: {model_name}, Device: {device}, Trust remote: {trust_remote}")
        
        # Try fallback to CPU if GPU failed
        if device != 'cpu':
            logger.warning("🔄 Attempting fallback to CPU...")
            try:
                model = SentenceTransformer(
                    model_name, 
                    device='cpu',
                    trust_remote_code=trust_remote
                )
                logger.info("✅ Model loaded successfully on CPU fallback")
                return True
            except Exception as fallback_error:
                logger.error(f"❌ CPU fallback also failed: {fallback_error}")
        
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "model not loaded"}), 503
    return jsonify({"status": "healthy", "model": "loaded"})

@app.route('/embed', methods=['POST'])
def embed_text():
    """Embed single text"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        task_type = data.get('task', 'search')  # Default to search
        doc_title = data.get('title', None)
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Apply task-specific prompts for EmbeddingGemma
        if task_type == 'document':
            # Document format
            title_part = doc_title if doc_title else "none"
            formatted_text = f"title: {title_part} | text: {text}"
        else:
            # Query format with various tasks
            task_descriptions = {
                'search': 'search result',
                'qa': 'question answering', 
                'fact': 'fact checking',
                'classification': 'classification',
                'clustering': 'clustering',
                'similarity': 'sentence similarity',
                'code': 'code retrieval'
            }
            task_desc = task_descriptions.get(task_type, 'search result')
            formatted_text = f"task: {task_desc} | query: {text}"
            
        # Generate embedding
        embedding = model.encode([formatted_text])[0]
        
        return jsonify({
            "embedding": embedding.tolist(),
            "dimension": len(embedding)
        })
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/embed/batch', methods=['POST'])
def embed_batch():
    """Embed multiple texts"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        task_type = data.get('task', 'search')  # Default to search
        doc_titles = data.get('titles', None)  # Optional list of titles for documents
        
        if not texts or not isinstance(texts, list):
            return jsonify({"error": "No texts provided or invalid format"}), 400
        
        # Apply task-specific prompts for EmbeddingGemma
        formatted_texts = []
        for i, text in enumerate(texts):
            if task_type == 'document':
                # Document format
                title = doc_titles[i] if doc_titles and i < len(doc_titles) else "none"
                formatted_text = f"title: {title} | text: {text}"
            else:
                # Query format with various tasks
                task_descriptions = {
                    'search': 'search result',
                    'qa': 'question answering',
                    'fact': 'fact checking',
                    'classification': 'classification',
                    'clustering': 'clustering',
                    'similarity': 'sentence similarity',
                    'code': 'code retrieval'
                }
                task_desc = task_descriptions.get(task_type, 'search result')
                formatted_text = f"task: {task_desc} | query: {text}"
            formatted_texts.append(formatted_text)
            
        # Generate embeddings
        embeddings = model.encode(formatted_texts)
        
        return jsonify({
            "embeddings": [emb.tolist() for emb in embeddings],
            "count": len(embeddings),
            "dimension": len(embeddings[0]) if len(embeddings) > 0 else 0
        })
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        exit(1)
    
    # Start server
    port = int(os.environ.get('PORT', 8001))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)