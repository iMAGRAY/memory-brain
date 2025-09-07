#!/bin/bash
# AI Memory Service - Docker Setup Script
# Secure setup with validation and error handling

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

# Logging function
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

# Check if Docker is installed and running
check_docker() {
    log "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker Desktop or Docker Engine."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose."
        exit 1
    fi
    
    success "Docker and Docker Compose are available"
}

# Generate secure password
generate_password() {
    local length=${1:-24}
    openssl rand -base64 "$length" | tr -d "=+/" | cut -c1-"$length"
}

# Generate API key
generate_api_key() {
    openssl rand -hex 32
}

# Validate password strength
validate_password() {
    local password="$1"
    local min_length=16
    
    if [[ ${#password} -lt $min_length ]]; then
        return 1
    fi
    
    # Check for at least one uppercase, lowercase, digit
    if [[ ! "$password" =~ [A-Z] ]] || [[ ! "$password" =~ [a-z] ]] || [[ ! "$password" =~ [0-9] ]]; then
        return 1
    fi
    
    return 0
}

# Create environment file
create_env_file() {
    log "Creating environment configuration..."
    
    if [[ -f "$ENV_FILE" ]]; then
        warning ".env file already exists"
        read -p "Do you want to overwrite it? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Using existing .env file"
            return 0
        fi
    fi
    
    if [[ ! -f "$ENV_EXAMPLE" ]]; then
        error ".env.example file not found. Cannot create environment configuration."
        exit 1
    fi
    
    # Generate secure passwords
    log "Generating secure credentials..."
    NEO4J_PASSWORD=$(generate_password 24)
    REDIS_PASSWORD=$(generate_password 20)
    GRAFANA_PASSWORD=$(generate_password 16)
    ADMIN_API_KEY=$(generate_api_key)
    
    # Validate generated passwords
    for password in "$NEO4J_PASSWORD" "$REDIS_PASSWORD" "$GRAFANA_PASSWORD"; do
        if ! validate_password "$password"; then
            error "Generated password does not meet security requirements"
            exit 1
        fi
    done
    
    # Copy template and replace placeholders
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    
    # Use sed to replace unique placeholders securely
    sed -i.bak \
        -e "s/NEO4J_PASSWORD_PLACEHOLDER/$NEO4J_PASSWORD/g" \
        -e "s/REDIS_PASSWORD_PLACEHOLDER/$REDIS_PASSWORD/g" \
        -e "s/GRAFANA_PASSWORD_PLACEHOLDER/$GRAFANA_PASSWORD/g" \
        -e "s/ADMIN_API_KEY_PLACEHOLDER/$ADMIN_API_KEY/g" \
        "$ENV_FILE"
    
    # Remove backup file
    rm -f "$ENV_FILE.bak"
    
    # Set restrictive permissions
    chmod 600 "$ENV_FILE"
    
    success "Environment file created with secure credentials"
    
    # Display credentials (user should save them securely)
    echo
    echo "🔐 Generated Credentials (SAVE THESE SECURELY):"
    echo "================================================="
    echo "Neo4j Password: $NEO4J_PASSWORD"
    echo "Redis Password: $REDIS_PASSWORD"
    echo "Grafana Password: $GRAFANA_PASSWORD"
    echo "Admin API Key: $ADMIN_API_KEY"
    echo "================================================="
    echo "⚠️  Store these credentials in your password manager!"
    echo
}

# Create required directories
create_directories() {
    log "Creating required directories..."
    
    local dirs=(
        "$PROJECT_DIR/data/neo4j"
        "$PROJECT_DIR/data/memory"
        "$PROJECT_DIR/logs"
        "$PROJECT_DIR/backup"
        "$PROJECT_DIR/docker/prometheus"
        "$PROJECT_DIR/docker/grafana/datasources"
        "$PROJECT_DIR/docker/grafana/dashboards"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        log "Created directory: $dir"
    done
    
    success "All directories created"
}

# Validate Docker Compose configuration
validate_compose() {
    log "Validating Docker Compose configuration..."
    
    cd "$PROJECT_DIR"
    
    if ! docker-compose config >/dev/null 2>&1; then
        error "Docker Compose configuration is invalid"
        docker-compose config
        exit 1
    fi
    
    success "Docker Compose configuration is valid"
}

# Pull required Docker images
pull_images() {
    log "Pulling required Docker images..."
    
    cd "$PROJECT_DIR"
    docker-compose pull
    
    success "All images pulled successfully"
}

# Start services
start_services() {
    log "Starting AI Memory Service..."
    
    cd "$PROJECT_DIR"
    
    # Start core services first
    log "Starting core services (Neo4j, AI Memory Service)..."
    docker-compose up -d neo4j ai-memory-service
    
    # Wait for core services to be healthy
    log "Waiting for services to be healthy..."
    local timeout=300
    local elapsed=0
    
    while [[ $elapsed -lt $timeout ]]; do
        if docker-compose ps | grep -E "(healthy|Up)" | wc -l | grep -q "2"; then
            success "Core services are healthy"
            break
        fi
        
        sleep 10
        elapsed=$((elapsed + 10))
        log "Waiting for services... (${elapsed}/${timeout}s)"
    done
    
    if [[ $elapsed -ge $timeout ]]; then
        error "Services did not become healthy within timeout"
        docker-compose logs
        exit 1
    fi
    
    # Optionally start monitoring services
    read -p "Do you want to start monitoring services (Prometheus, Grafana)? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Starting monitoring services..."
        docker-compose --profile monitoring up -d
    fi
    
    success "AI Memory Service is running!"
}

# Display service information
show_service_info() {
    log "Service Information"
    echo "==================="
    
    # Load environment variables
    if [[ -f "$ENV_FILE" ]]; then
        # shellcheck source=/dev/null
        source "$ENV_FILE"
    fi
    
    echo "🌐 Web Interfaces:"
    echo "  • Memory API: http://localhost:${API_PORT:-8080}"
    echo "  • Admin Panel: http://localhost:${ADMIN_PORT:-8081}"
    echo "  • Neo4j Browser: http://localhost:${NEO4J_HTTP_PORT:-7474}"
    echo "  • Grafana (if enabled): http://localhost:${GRAFANA_PORT:-3000}"
    echo
    echo "🔧 Management Commands:"
    echo "  • View logs: docker-compose logs -f"
    echo "  • Stop services: docker-compose down"
    echo "  • Update services: docker-compose pull && docker-compose up -d"
    echo
    echo "📊 Service Status:"
    docker-compose ps
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    # Add cleanup logic if needed
}

# Main execution
main() {
    log "Starting AI Memory Service Docker Setup"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run setup steps
    check_docker
    create_env_file
    create_directories
    validate_compose
    pull_images
    start_services
    show_service_info
    
    success "Setup completed successfully!"
    
    echo
    echo "🎉 AI Memory Service is now running!"
    echo "Visit http://localhost:8080/health to verify the service is working."
    echo
    echo "📖 For more information, see the documentation in docker/README.md"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo "AI Memory Service Docker Setup"
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --help, -h    Show this help message"
            echo "  --no-start    Setup but don't start services"
            echo ""
            exit 0
            ;;
        --no-start)
            NO_START=1
            shift
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function if not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi