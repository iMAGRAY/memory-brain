# AI Memory Service - Deployment Options

## Overview

The AI Memory Service provides multiple deployment options to suit different needs and environments:

1. **Native Installation** - Direct installation without Docker
2. **Embedded Database** - SQLite fallback for Neo4j
3. **Docker Compose** - Complete containerized setup
4. **Production Installer** - Automated setup with security

## 1. Native Neo4j Installation (Windows)

For users who prefer not to use Docker, run the PowerShell script:

```powershell
# Run as Administrator
.\setup_neo4j_native.ps1
```

**Features:**
- Downloads and installs Neo4j Community Edition
- Configures Java 17+ automatically
- Sets up Windows service
- Creates management scripts
- Configures for AI Memory Service

**Requirements:**
- Windows 10+ with PowerShell 5.1+
- Administrator privileges
- Internet connection for downloads

## 2. Embedded Database (SQLite)

When Neo4j is not available, the service automatically falls back to SQLite:

```rust
// Automatic fallback in storage layer
let storage = if neo4j_available {
    GraphStorage::new(&config.neo4j).await?
} else {
    EmbeddedStorage::new(EmbeddedDbConfig::default()).await?
};
```

**Features:**
- Full-text search with FTS5
- Vector similarity search
- In-memory caching
- Automatic schema management
- WAL mode for concurrency

**Limitations:**
- Single-node only
- Limited graph capabilities
- No real-time clustering

## 3. Docker Compose Deployment

### Simple Development Setup

```bash
# Start basic services
docker-compose up -d neo4j

# Check Neo4j browser
open http://localhost:7474
# Username: neo4j, Password: set in .env file
```

### Full Stack Development

```bash
# Create environment file
cp .env.example .env
# Edit NEO4J_PASSWORD and other settings

# Start all services
docker-compose up -d

# Services available:
# - Neo4j: http://localhost:7474
# - Memory API: http://localhost:8080
# - Admin: http://localhost:8081
# - WebSocket: ws://localhost:8082
```

### Environment Variables

Required in `.env` file:

```bash
# Database
NEO4J_PASSWORD=your-secure-password-here
NEO4J_USER=neo4j

# Service Configuration  
RUST_LOG=info
MEMORY_CACHE_MB=512

# Optional Monitoring
ENABLE_PROMETHEUS=true
ENABLE_GRAFANA=true
```

**Security Notes:**
- Never commit `.env` files to version control
- Use strong passwords (minimum 16 characters)
- Restrict CORS origins in production
- Enable TLS/SSL for external access

## 4. Production Deployment

### Security-First Configuration

```bash
# Generate secure passwords
export NEO4J_PASSWORD=$(openssl rand -base64 32)
export REDIS_PASSWORD=$(openssl rand -base64 32)
export GRAFANA_PASSWORD=$(openssl rand -base64 32)

# Restrict network access
export CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"

# Use production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Resource Limits

Services are configured with appropriate resource limits:

- **Neo4j**: 2GB RAM, optimized for graph operations
- **Memory Service**: 2GB RAM limit, 1GB reserved
- **Python Embeddings**: 4GB RAM limit, 2GB reserved
- **Redis**: 512MB RAM limit
- **Monitoring**: Combined 1.5GB limit

### Health Monitoring

All services include health checks:

```bash
# Check service health
docker-compose ps

# View service logs
docker-compose logs ai-memory-service

# Monitor resource usage
docker stats
```

## 5. Installation Script

For fully automated deployment:

```powershell
# Windows
.\install.ps1 -Environment Production -DataDir "D:\AIMemory"

# Options:
# -Environment: Development, Production
# -DataDir: Custom data directory
# -InstallNeo4j: true/false
# -UseDocker: true/false
```

The installer:
- Detects system requirements
- Generates secure passwords
- Configures networking
- Sets up monitoring
- Creates backup scripts
- Configures Windows services

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │────│   Nginx Proxy    │────│  Load Balancer  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
┌────────▼────────┐    ┌────────▼────────┐    ┌───────▼───────┐
│ AI Memory API   │    │ Python Embeddings│    │   Monitoring  │
│ (Rust Service)  │    │  (Transformers)  │    │ (Prometheus)  │
└─────────────────┘    └─────────────────┘    └───────────────┘
         │                      │                      │
         └──────────┬───────────┘                      │
                    │                                  │
         ┌──────────▼──────────┐              ┌───────▼───────┐
         │      Neo4j          │              │    Grafana    │
         │   Graph Database    │              │   Dashboards  │
         └─────────────────────┘              └───────────────┘
                    │
         ┌──────────▼──────────┐
         │      Redis          │
         │     (Optional)      │
         └─────────────────────┘
```

## Backup and Recovery

### Automated Backups

```bash
# Database backup
docker-compose exec neo4j neo4j-admin dump --database=neo4j --to=/backups/

# Full system backup
./scripts/backup.sh --full

# Restore from backup
./scripts/restore.sh --from=backup-2024-01-01.tar.gz
```

### Disaster Recovery

1. **Data Persistence**: All data stored in named volumes
2. **Configuration Backup**: Environment files and configs
3. **State Recovery**: Redis and in-memory cache rebuild
4. **Health Validation**: Automated health checks post-recovery

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   ```bash
   # Check Neo4j logs
   docker-compose logs neo4j
   
   # Verify network connectivity
   docker network ls
   docker network inspect ai-memory-network
   ```

2. **Python Embedding Service Timeout**
   ```bash
   # Check model download progress
   docker-compose logs python-embeddings
   
   # Increase timeout in environment
   export EMBEDDING_TIMEOUT_SECONDS=60
   ```

3. **Memory Service Startup Issues**
   ```bash
   # Check configuration
   docker-compose exec ai-memory-service cat /app/config/config.toml
   
   # Validate environment variables
   docker-compose config
   ```

### Performance Tuning

1. **Neo4j Memory**: Adjust based on dataset size
2. **SIMD Threads**: Set to CPU core count
3. **Batch Sizes**: Tune for embedding performance
4. **Cache Size**: Balance memory vs. speed

For detailed configuration options, see `config/production.toml`.