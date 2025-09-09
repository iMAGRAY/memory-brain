# AI Memory Service 🧠

Advanced intelligent memory system for AI agents with cognitive science-inspired architecture, GPT-5-nano orchestration, and Claude Code integration.

## 🌟 Features

### Core Architecture
- **Cognitive Memory Types**: Semantic, Episodic, Procedural, Working, Code, Documentation, and Conversation memory
- **Multi-layer Recall System**: Three-layer progressive retrieval (Semantic → Contextual → Detailed)
- **GPT-5-nano Orchestrator**: Intelligent memory management with 400K token context
- **EmbeddingGemma-300M**: State-of-the-art multilingual embeddings (768-dimensional)

### Performance & Integration
- **SIMD-optimized Vector Search**: Hardware-accelerated similarity calculations
- **Multi-level Caching**: L1 (DashMap), L2 (Moka with TTL), L3 (compressed)
- **MCP Server**: Native Claude Code integration via Model Context Protocol
- **REST API**: Comprehensive HTTP endpoints for all operations
- **Session Isolation**: Cross-session knowledge extraction with privacy

### Storage & Scalability
- **Dual Storage Backend**: RocksDB for persistence, in-memory for speed
- **Graph Relationships**: Neo4j for complex memory interconnections
- **Compression Support**: Automatic data compression for efficiency
- **Parallel Processing**: Tokio async runtime with thread-safe operations

## 📋 Prerequisites

- **Rust**: 1.75+ (for async traits)
- **Python**: 3.11+ (for embedding service)
- **Docker**: For Neo4j and containerized deployment
- **OS**: Windows 10/11, Linux, macOS
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional (improves embedding performance)

## 🚀 Quick Start

### 1. Clone and Configure

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-memory-service.git
cd ai-memory-service

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your configuration:
# - Set OPENAI_API_KEY for GPT-5-nano
# - Configure NEO4J_PASSWORD
# - Adjust other settings as needed
```

### 2. Install Dependencies

```bash
# Install Rust dependencies
cargo build --release

# Install Python dependencies for embedding service
pip install -r requirements.txt
```

### 3. Start Services

```bash
# Start Neo4j database (optional, if using graph storage)
docker-compose up -d neo4j

# Start embedding service
python embedding_service.py

# Start the main memory service
cargo run --release --bin memory-server
```

### 4. Verify Installation

```bash
# Check service health
curl http://localhost:8080/health

# Test memory storage
curl -X POST http://localhost:8080/api/v1/memory \
  -H "Content-Type: application/json" \
  -d '{"content": "Test memory", "context_hint": "testing"}'
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code / AI Agent               │
└─────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   MCP Server      │
                    │  (Claude Code)    │
                    └─────────┬─────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                      REST API Layer                        │
│  (Axum Web Framework with Tower Middleware)               │
└─────────────────────────────┬─────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                 GPT-5-nano Orchestrator                    │
│     (Memory Optimization, Insights, Distillation)         │
└─────────────────────────────┬─────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                    Memory Service Core                     │
│        (Cognitive Architecture Implementation)             │
├────────────────────┬────────────────┬─────────────────────┤
│   Storage Layer    │  Cache Layer   │   Embedding Layer   │
│  (RocksDB/Neo4j)   │  (Multi-level) │  (EmbeddingGemma)   │
└────────────────────┴────────────────┴─────────────────────┘
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```env
# GPT-5-nano Orchestrator
OPENAI_API_KEY=sk-...
ORCHESTRATOR_MODEL=gpt-5-nano
MAX_INPUT_TOKENS=400000
MAX_OUTPUT_TOKENS=12000

# Memory Service
MAX_MEMORY_SIZE=1000  # MB
CACHE_SIZE=100
SESSION_TIMEOUT=3600

# Embedding Service
EMBEDDING_MODEL=onnx-community/embeddinggemma-300m-ONNX
EMBEDDING_DIMENSIONS=768
MAX_SEQUENCE_LENGTH=2048

# Storage
STORAGE_BACKEND=rocksdb  # or 'memory'
STORAGE_PATH=./data/memory_store
```

### Configuration File

Advanced configuration via `config.toml`:

```toml
[memory]
max_memories = 100000
importance_threshold = 0.3

[orchestrator]
enable = true
model = "gpt-5-nano"
reasoning_effort = "medium"

[api]
host = "0.0.0.0"
port = 8080
cors_enabled = true
```

## 📚 API Documentation

### Memory Operations

```bash
# Store a memory
POST /api/v1/memory
{
  "content": "Important information",
  "context_hint": "project/context",
  "memory_type": "semantic"
}

# Search memories
POST /api/v1/memory/search
{
  "query": "search text",
  "limit": 10,
  "min_importance": 0.5
}

# Advanced recall with orchestrator
POST /api/v1/memory/search/advanced
{
  "query": "complex query",
  "context": "specific/context",
  "include_related": true
}
```

### Maintenance Endpoints

```bash
# Trigger importance decay (single tick). Returns count of updated nodes
POST /maintenance/decay
{}

# Emulate N daily decay ticks (virtual days). Returns ticks and total updates
POST /maintenance/tick
{
  "ticks": 7
}

# Trigger duplicate consolidation (context/threshold/max_items)
POST /maintenance/consolidate
{
  "context": "optional/context",
  "similarity_threshold": 0.92,
  "max_items": 120
}

# Aliases (versioned /api/v1 and compatibility routes)
POST /api/v1/maintenance/decay                # alias
POST /api/v1/maintenance/tick                 # alias
POST /api/v1/maintenance/consolidate          # alias
POST /api/memory/consolidate                  # alias to /maintenance/consolidate
GET|POST /api/search                          # alias to /api/memory/search
```

### Compatibility & Stability

- Aliases preserved for compatibility:
  - `POST /memories`, `POST /api/memories`, `POST /api/memory` → store memory
  - `POST /memories/search`, `POST /api/memories/search`, `POST /api/memory/search`, `GET /search` → search
  - `POST /api/memory/consolidate` → alias to `/maintenance/consolidate`
  - Versioned aliases: `/api/v1/memory`, `/api/v1/memory/search`, `/api/v1/maintenance/*`
- Determinism: repeatable search ordering and stable assembled context hash under identical state.
- Maintenance operations require explicit trigger and are idempotent-safe at API layer.

### Orchestrator Operations

```bash
# Generate insights
POST /api/v1/orchestrator/insights
{
  "context": "project/analysis",
  "limit": 20
}

# Distill context
POST /api/v1/orchestrator/distill
{
  "context": "session/context",
  "max_tokens": 2000
}
```

## 🧪 Testing

```bash
# Run unit tests
cargo test

# Run integration tests
cargo test --test integration

# Test with coverage
cargo tarpaulin --out Html

# Benchmark performance
cargo bench
```

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t ai-memory-service .

# Run with Docker Compose
docker-compose up

# Or run standalone
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v ./data:/app/data \
  ai-memory-service
```

## 📊 Performance

- **Embedding Speed**: ~22ms per text (on-device)
- **Vector Search**: <5ms for 10K vectors (SIMD-optimized)
- **Memory Storage**: ~1ms write latency
- **Cache Hit Rate**: >90% for frequent queries
- **Concurrent Connections**: 10K+ with Tokio

## 🔒 Security

- API key authentication for OpenAI
- Environment-based configuration
- Input validation and sanitization
- Rate limiting support
- CORS configuration

## 📖 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [Development Guide](docs/development.md)
- [MCP Integration](docs/mcp-integration.md)

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-5-nano API
- Google for EmbeddingGemma model
- Anthropic for Claude Code and MCP protocol
- Rust community for excellent libraries

## 📞 Support

- Issues: [GitHub Issues](https://github.com/yourusername/ai-memory-service/issues)
- Documentation: [Wiki](https://github.com/yourusername/ai-memory-service/wiki)
- Discord: [Join our server](https://discord.gg/yourinvite)

---

Built with ❤️ using Rust, designed for the future of AI memory systems.
