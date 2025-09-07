# 🚀 AI Memory Service - Docker Quick Start

> **Self-contained deployment requiring only Docker**  
> No need to install Rust, Python, Neo4j, or any models manually!

## ⚡ Super Quick Start

```bash
# Clone and start in 3 commands
git clone <your-repo-url> ai-memory-service
cd ai-memory-service
./scripts/quick-start.sh
```

That's it! The service will be available at `http://localhost:8080`

## 📋 What You Get

✅ **Complete AI Memory Service** with vector search and graph storage  
✅ **EmbeddingGemma-300M model** automatically downloaded and ready  
✅ **Neo4j graph database** with web interface at `http://localhost:7474`  
✅ **Python embedding service** for text processing  
✅ **Secure credentials** automatically generated  
✅ **Health monitoring** and logging included  

## 🛠️ Requirements

- **Docker Desktop** or **Docker Engine** (latest version)
- **4GB+ RAM** (8GB+ recommended)
- **2GB+ free disk space**
- **Internet connection** (for initial model download)

## 🎯 Service Endpoints

| Service | URL | Purpose |
|---------|-----|---------|
| **Main API** | `http://localhost:8080` | Memory storage and search |
| **Admin Panel** | `http://localhost:8081` | Service administration |
| **Python Embeddings** | `http://localhost:8001` | Text embedding generation |
| **Neo4j Browser** | `http://localhost:7474` | Graph database interface |

## 🧪 Test the Service

```bash
# Health check
curl http://localhost:8080/health

# Store a memory
curl -X POST http://localhost:8080/memory \
  -H "Content-Type: application/json" \
  -d '{"content": "Docker makes deployment easy!", "tags": ["tech", "deployment"]}'

# Search memories
curl "http://localhost:8080/memory/search?q=deployment&limit=10"
```

## 🔧 Management Commands

```bash
# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Update and restart
docker-compose pull && docker-compose up -d

# Enable monitoring (Grafana + Prometheus)
docker-compose --profile monitoring up -d

# Clean restart (removes data)
docker-compose down -v && ./scripts/quick-start.sh
```

## 📊 Optional Monitoring

Enable Grafana dashboards and Prometheus metrics:

```bash
docker-compose --profile monitoring up -d
```

- **Grafana**: `http://localhost:3000` (admin/generated_password)
- **Prometheus**: `http://localhost:9090`

## 🔐 Security

All passwords are automatically generated and stored in `.env`:
- Neo4j database password
- Redis cache password  
- Grafana admin password
- API administration key

**Keep the `.env` file secure and never commit it to version control!**

## 📁 Data Persistence

Your data is automatically persisted in Docker volumes:
- `neo4j_data` - Graph database storage
- `memory_data` - Application data
- `model_cache` - Downloaded AI models

## 🚑 Troubleshooting

### Service won't start?
```bash
# Check Docker is running
docker info

# Check logs for errors
docker-compose logs -f ai-memory-service

# Verify system resources
docker system df
```

### Out of memory?
```bash
# Reduce memory limits in docker-compose.yml
# Or increase Docker Desktop memory limit
```

### Can't connect to service?
```bash
# Check if ports are available
docker-compose ps

# Test each service individually
curl http://localhost:8080/health
curl http://localhost:8001/health
```

### Models not downloading?
```bash
# Check internet connection and try rebuilding
docker-compose build --no-cache python-embeddings
```

## 🎨 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │────│  AI Memory API  │────│ Python Embedding│
│                 │    │  (Rust Service) │    │    Service      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │                        │
                       ┌───────▼───────┐        ┌───────▼───────┐
                       │ Neo4j Database│        │EmbeddingGemma │
                       │  (Graph Store)│        │  300M Model   │
                       └───────────────┘        └───────────────┘
```

## 🔄 Development vs Production

### Development Mode (Default)
- Relaxed security settings
- Debug logging enabled
- Development volumes mounted

### Production Mode
```bash
# Use production environment file
cp .env.example .env.production
# Edit .env.production with secure settings
docker-compose --env-file .env.production up -d
```

## 📚 Next Steps

1. **Read the API docs**: `docs/api.md`
2. **Try the examples**: `examples/`
3. **Configure for production**: `DEPLOYMENT.md`
4. **Monitor performance**: Enable Grafana monitoring
5. **Scale up**: See `docker-compose.override.yml` for scaling

## 🆘 Support

- **Logs**: `docker-compose logs -f`
- **Health**: `curl http://localhost:8080/health`
- **Status**: `docker-compose ps`
- **Resources**: `docker system df`

---

**🎉 Enjoy your self-contained AI Memory Service!**

*Everything you need is included - just add Docker!*