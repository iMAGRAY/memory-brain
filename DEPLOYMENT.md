# 🚀 AI Memory Service - Deployment Guide

## Quick Start Options

The AI Memory Service provides **4 deployment options** to handle Neo4j setup without requiring users to configure it manually:

### Option 1: Automated Installer (Recommended)
```powershell
# Windows - Detects system and chooses best option automatically
.\install.ps1 -Environment Development

# For production
.\install.ps1 -Environment Production -EnableMonitoring $true
```

### Option 2: Docker Compose (If Docker Available)
```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with secure passwords

# Quick start with setup script
chmod +x scripts/setup-docker.sh
./scripts/setup-docker.sh

# Or manual start
docker-compose up -d
```

### Option 3: Native Installation (No Docker)
```powershell
# Windows - Installs Neo4j Community Edition natively
.\setup_neo4j_native.ps1
```

### Option 4: Embedded Database (SQLite Fallback)
If Neo4j is unavailable, the service automatically uses SQLite with full functionality:
- Vector similarity search
- Full-text search (FTS5)
- In-memory caching
- Automatic schema management

## 🎯 Deployment Decision Matrix

| Scenario | Recommended Option | Command |
|----------|-------------------|---------|
| **First time user** | Automated Installer | `.\install.ps1` |
| **Docker available** | Docker Compose | `./scripts/setup-docker.sh` |
| **No Docker, want Neo4j** | Native Installation | `.\setup_neo4j_native.ps1` |
| **Quick testing** | Embedded SQLite | Just run the service |
| **Production** | Docker + Monitoring | `.\install.ps1 -Environment Production` |

## 📋 System Requirements

### Minimum Requirements
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **CPU**: Dual-core 2GHz
- **OS**: Windows 10+ / Linux / macOS

### Optimal Performance
- **RAM**: 16GB+ for large datasets
- **Storage**: SSD recommended
- **CPU**: 4+ cores with AVX2 support
- **Network**: Low latency for distributed setups

## 🔧 Configuration

### Environment Variables (.env file)
```bash
# Database
NEO4J_PASSWORD=your-secure-password-here  # REQUIRED
NEO4J_USER=neo4j

# Security
API_KEY_REQUIRED=true
CORS_ORIGINS=http://localhost:3000  # Restrict in production

# Performance
MEMORY_CACHE_MB=512
TOKIO_THREADS=4
RUST_LOG=info,ai_memory_service=debug
```

### Generate Secure Passwords
```bash
# Linux/macOS
openssl rand -base64 32

# Windows PowerShell
[System.Web.Security.Membership]::GeneratePassword(24, 8)
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Client    │────│      Nginx       │────│ Load Balancer   │
│   (Frontend)    │    │   (Optional)     │    │   (Optional)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
┌────────▼────────┐    ┌────────▼────────┐    ┌───────▼───────┐
│ AI Memory API   │    │ Python Embeddings│    │  Monitoring   │
│ (Rust Service)  │    │ (Transformers)   │    │ (Prometheus)  │
│                 │    │                  │    │               │
│ • SIMD Search   │    │ • Text Encoding  │    │ • Metrics     │
│ • Cache Layer   │    │ • Model Serving  │    │ • Alerting    │
│ • WebSocket     │    │ • Batch Process  │    │ • Dashboards  │
└─────────────────┘    └─────────────────┘    └───────────────┘
         │                      │                      │
         └──────────┬───────────┘                      │
                    │                                  │
         ┌──────────▼──────────┐              ┌───────▼───────┐
         │      Database       │              │   Grafana     │
         │                     │              │  Dashboards   │
         │ Neo4j (Preferred)   │              │               │
         │    OR               │              │ • Performance │
         │ SQLite (Fallback)   │              │ • Usage Stats │
         │                     │              │ • Alerts      │
         └─────────────────────┘              └───────────────┘
```

## 🔐 Security Configuration

### Development (Default)
```bash
API_KEY_REQUIRED=false
CORS_ORIGINS=*
NEO4J_PASSWORD=generated-secure-password
```

### Production (Required Changes)
```bash
API_KEY_REQUIRED=true
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
NEO4J_PASSWORD=ultra-secure-password-32-chars
ADMIN_API_KEY=secure-api-key-for-admin-access

# Enable HTTPS
USE_TLS=true
TLS_CERT_PATH=/path/to/cert.pem
TLS_KEY_PATH=/path/to/key.pem
```

### Firewall Rules
```bash
# Allow only required ports
ufw allow 8080/tcp  # API
ufw allow 8081/tcp  # Admin (restrict to admin IPs)
ufw deny 7474/tcp   # Neo4j HTTP (internal only)
ufw deny 7687/tcp   # Neo4j Bolt (internal only)
```

## 📊 Monitoring & Observability

### Health Checks
```bash
# Service health
curl http://localhost:8080/health

# Database connectivity
curl http://localhost:8081/health

# Metrics endpoint
curl http://localhost:8081/metrics
```

### Log Locations
```
Docker:     docker-compose logs -f ai-memory-service
Native:     ./logs/ai-memory-service.log
Windows:    %PROGRAMDATA%\AIMemoryService\logs\
```

### Performance Metrics
- **Memory Usage**: Monitor heap and cache utilization
- **Request Latency**: Track API response times
- **Database Performance**: Neo4j query times
- **SIMD Operations**: Vector search performance

## 🔄 Backup & Recovery

### Automated Backups
```bash
# Neo4j backup (Docker)
docker-compose exec neo4j neo4j-admin dump --database=neo4j --to=/backups/

# Full system backup
./scripts/backup.sh --full --output=/backup/$(date +%Y%m%d).tar.gz
```

### Disaster Recovery
1. **Data**: All data in persistent Docker volumes
2. **Config**: Environment files and secrets
3. **State**: Redis cache rebuilds automatically
4. **Recovery**: `./scripts/restore.sh --from=backup.tar.gz`

## 🚦 Troubleshooting

### Common Issues

#### 1. Neo4j Connection Failed
```bash
# Check Neo4j status
docker-compose logs neo4j

# Verify connectivity
telnet localhost 7687

# Check credentials
docker-compose exec neo4j cypher-shell -u neo4j -p yourpassword
```

#### 2. Python Dependencies Missing
```bash
# Install Python packages
pip install sentence-transformers torch numpy

# Check PyO3 integration
cargo run --bin test_simple_pyo3
```

#### 3. SIMD Performance Issues
```bash
# Check CPU features
cargo run --bin test_working_components

# Verify AVX2 support
cat /proc/cpuinfo | grep avx2

# Windows
wmic cpu get name,family,model,stepping
```

#### 4. Memory/Performance Issues
```bash
# Monitor resource usage
docker stats

# Check service logs
docker-compose logs -f ai-memory-service

# Adjust memory limits in docker-compose.yml
```

### Performance Tuning

#### SIMD Optimization
```rust
// Automatic detection in config
simd_threads: 0,  // Uses all CPU cores
batch_size: 1000, // Optimal for most systems
```

#### Database Tuning
```yaml
# docker-compose.yml
environment:
  - NEO4J_server_memory_heap_max__size=4G      # Increase for large datasets
  - NEO4J_server_memory_pagecache_size=4G       # Cache for graph traversals
  - NEO4J_server_bolt_thread__pool__max__size=400
```

#### Caching Strategy
```bash
# Environment configuration
MEMORY_CACHE_MB=1024          # In-memory cache size
L1_CACHE_SIZE=10000          # Frequently accessed items
L2_CACHE_SIZE=100000         # Extended cache
CACHE_TTL_SECONDS=7200       # 2 hours default TTL
```

## 🌐 Production Deployment

### Load Balancing
```nginx
# nginx.conf
upstream ai_memory_backend {
    server ai-memory-service-1:8080;
    server ai-memory-service-2:8080;
    server ai-memory-service-3:8080;
}

server {
    listen 80;
    location / {
        proxy_pass http://ai_memory_backend;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### Scaling Considerations
- **Horizontal**: Multiple service instances with shared Neo4j
- **Vertical**: Increase memory/CPU for single instance
- **Database**: Neo4j clustering for high availability
- **Cache**: Redis cluster for distributed caching

## 📝 Quick Reference

### Essential Commands
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Restart service
docker-compose restart ai-memory-service

# Update images
docker-compose pull && docker-compose up -d

# Backup data
./scripts/backup.sh

# Check health
curl http://localhost:8080/health
```

### API Endpoints
- **Health**: `GET /health`
- **Metrics**: `GET /metrics`
- **Store Memory**: `POST /memory`
- **Search**: `GET /memory/search?q=query`
- **Similar**: `POST /memory/similar`

### Default Ports
- **8080**: Main API
- **8081**: Admin interface
- **8082**: WebSocket
- **7474**: Neo4j browser
- **7687**: Neo4j bolt
- **3000**: Grafana (if enabled)
- **9090**: Prometheus (if enabled)

---

## 🆘 Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Verify health: `curl http://localhost:8080/health`
3. Review configuration in `.env` file
4. See troubleshooting section above

**The service is designed to work out-of-the-box with minimal configuration required!**