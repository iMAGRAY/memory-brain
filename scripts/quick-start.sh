#!/bin/bash
# AI Memory Service - Quick Start Script
# Self-contained deployment requiring only Docker

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"
ENV_EXAMPLE="$PROJECT_DIR/.env.example"

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check system requirements
check_requirements() {
    log "🔍 Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker Desktop or Docker Engine."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose."
        exit 1
    fi
    
    # Check available resources
    local available_memory
    if command -v free &> /dev/null; then
        available_memory=$(free -m | awk '/^Mem:/{print $2}')
        if [[ $available_memory -lt 4096 ]]; then
            warning "System has less than 4GB RAM ($available_memory MB). Performance may be limited."
        fi
    fi
    
    success "✅ System requirements satisfied"
}

# Generate secure credentials
generate_secure_env() {
    log "🔐 Generating secure environment configuration..."
    
    if [[ -f "$ENV_FILE" ]]; then
        warning ".env file already exists"
        read -p "Do you want to regenerate credentials? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Using existing .env file"
            return 0
        fi
    fi
    
    if [[ ! -f "$ENV_EXAMPLE" ]]; then
        error ".env.example file not found"
        exit 1
    fi
    
    # Generate secure passwords
    local neo4j_password redis_password grafana_password admin_api_key
    neo4j_password=$(openssl rand -base64 24 | tr -d "=+/" | cut -c1-24)
    redis_password=$(openssl rand -base64 20 | tr -d "=+/" | cut -c1-20)
    grafana_password=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-16)
    admin_api_key=$(openssl rand -hex 32)
    
    # Create .env file with replacements
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    
    # Replace placeholders
    sed -i.bak \
        -e "s/NEO4J_PASSWORD_PLACEHOLDER/$neo4j_password/g" \
        -e "s/REDIS_PASSWORD_PLACEHOLDER/$redis_password/g" \
        -e "s/GRAFANA_PASSWORD_PLACEHOLDER/$grafana_password/g" \
        -e "s/ADMIN_API_KEY_PLACEHOLDER/$admin_api_key/g" \
        "$ENV_FILE"
    
    rm -f "$ENV_FILE.bak"
    chmod 600 "$ENV_FILE"
    
    success "✅ Secure credentials generated"
    
    echo
    echo "🔐 Generated Credentials (SAVE THESE SECURELY):"
    echo "================================================="
    echo "Neo4j Password: $neo4j_password"
    echo "Redis Password: $redis_password"
    echo "Grafana Password: $grafana_password"
    echo "Admin API Key: $admin_api_key"
    echo "================================================="
    echo "⚠️  These credentials are saved in .env file"
    echo
}

# Build and start services
start_services() {
    log "🚀 Building and starting AI Memory Service..."
    
    cd "$PROJECT_DIR"
    
    # Validate docker-compose configuration
    if ! docker-compose config >/dev/null 2>&1; then
        error "Docker Compose configuration is invalid"
        docker-compose config
        exit 1
    fi
    
    # Build images
    log "📦 Building container images (this may take 10-15 minutes)..."
    docker-compose build --parallel
    
    # Start core services
    log "🎯 Starting core services..."
    docker-compose up -d neo4j python-embeddings ai-memory-service
    
    # Wait for services to be healthy
    log "⏳ Waiting for services to be ready..."
    local timeout=600  # 10 minutes
    local elapsed=0
    
    while [[ $elapsed -lt $timeout ]]; do
        local healthy_count
        healthy_count=$(docker-compose ps --format json | jq -r 'select(.Health == "healthy")' | wc -l)
        
        if [[ $healthy_count -ge 3 ]]; then
            success "✅ All core services are healthy!"
            break
        fi
        
        sleep 15
        elapsed=$((elapsed + 15))
        log "Waiting... ($elapsed/${timeout}s) - Healthy services: $healthy_count/3"
    done
    
    if [[ $elapsed -ge $timeout ]]; then
        error "❌ Services failed to start within timeout"
        log "Service status:"
        docker-compose ps
        log "Logs:"
        docker-compose logs --tail=20
        exit 1
    fi
}

# Display service information
show_service_info() {
    log "📋 Service Information"
    echo "======================"
    
    # Load environment for port info
    if [[ -f "$ENV_FILE" ]]; then
        # shellcheck source=/dev/null
        source "$ENV_FILE"
    fi
    
    echo "🌐 Access Points:"
    echo "  • Memory API:      http://localhost:${API_PORT:-8080}"
    echo "  • Admin Panel:     http://localhost:${ADMIN_PORT:-8081}"
    echo "  • Python Embeddings: http://localhost:8001"
    echo "  • Neo4j Browser:   http://localhost:${NEO4J_HTTP_PORT:-7474}"
    echo
    echo "🧪 Quick API Test:"
    echo "  curl -X GET http://localhost:8080/health"
    echo
    echo "🛠️  Management Commands:"
    echo "  • View logs:       docker-compose logs -f"
    echo "  • Stop services:   docker-compose down"
    echo "  • Start monitoring: docker-compose --profile monitoring up -d"
    echo "  • Update services: docker-compose pull && docker-compose up -d"
    echo
    echo "📊 Current Status:"
    docker-compose ps
    echo
}

# Test service connectivity
test_services() {
    log "🧪 Testing service connectivity..."
    
    # Test main API
    if curl -f -s http://localhost:8080/health >/dev/null; then
        success "✅ Memory API: Healthy"
    else
        warning "⚠️  Memory API: Not responding"
    fi
    
    # Test Python embeddings
    if curl -f -s http://localhost:8001/health >/dev/null; then
        success "✅ Python Embeddings: Healthy"
    else
        warning "⚠️  Python Embeddings: Not responding"
    fi
    
    # Test Neo4j
    if curl -f -s http://localhost:7474 >/dev/null; then
        success "✅ Neo4j Browser: Accessible"
    else
        warning "⚠️  Neo4j Browser: Not accessible"
    fi
}

# Optional monitoring services
setup_monitoring() {
    read -p "Do you want to enable monitoring (Prometheus + Grafana)? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "📊 Starting monitoring services..."
        docker-compose --profile monitoring up -d
        
        echo "📊 Monitoring Access:"
        echo "  • Grafana Dashboard: http://localhost:${GRAFANA_PORT:-3000}"
        echo "  • Prometheus:        http://localhost:${PROMETHEUS_PORT:-9090}"
        echo "  • Health Monitor:    docker-compose logs -f healthcheck"
    fi
}

# Cleanup function
cleanup() {
    if [[ ${1:-0} -ne 0 ]]; then
        log "🧹 Cleaning up due to error..."
        docker-compose down 2>/dev/null || true
    fi
}

# Main execution
main() {
    echo "🚀 AI Memory Service - Quick Start"
    echo "=================================="
    echo "Self-contained deployment with everything included!"
    echo
    
    # Set trap for cleanup on error
    trap 'cleanup $?' EXIT
    
    # Run setup steps
    check_requirements
    generate_secure_env
    start_services
    test_services
    show_service_info
    setup_monitoring
    
    success "🎉 AI Memory Service is running!"
    
    echo
    echo "🔗 Next Steps:"
    echo "  1. Visit http://localhost:8080/health to verify the service"
    echo "  2. Check the API documentation in docs/api.md"
    echo "  3. Try the example requests in examples/"
    echo
    echo "📖 For troubleshooting, see DEPLOYMENT.md"
    
    # Reset trap
    trap - EXIT
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "AI Memory Service Quick Start"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --help, -h     Show this help message"
            echo "  --monitoring   Enable monitoring services automatically"
            echo "  --no-test      Skip service connectivity tests"
            echo ""
            echo "This script sets up a complete self-contained AI Memory Service"
            echo "requiring only Docker to be installed on your system."
            echo ""
            exit 0
            ;;
        --monitoring)
            ENABLE_MONITORING=1
            shift
            ;;
        --no-test)
            SKIP_TESTS=1
            shift
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"