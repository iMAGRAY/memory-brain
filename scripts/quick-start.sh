#!/bin/bash

# AI Memory Service - Quick Start Script
# Automated setup and deployment with security best practices
# Requires only Docker and model files from user

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"

# Function: Print colored output
print_step() {
    echo -e "${BLUE}🔹 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${PURPLE}💡 $1${NC}"
}

# Function: Generate secure password
generate_password() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "import secrets; print(secrets.token_urlsafe(25))"
    else
        # Fallback: use date and random
        echo "$(date +%s)$(shuf -i 1000-9999 -n 1)" | sha256sum | cut -c1-25
    fi
}

# Function: Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker >/dev/null 2>&1; then
        print_error "Docker is not installed. Please install Docker Desktop."
        print_info "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version >/dev/null 2>&1; then
        print_error "Docker Compose is not available. Please update Docker Desktop."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    
    print_success "Docker is available and running"
    
    # Check system resources
    local available_memory
    if command -v free >/dev/null 2>&1; then
        available_memory=$(free -m | awk 'NR==2{printf "%d", $7}')
    elif command -v vm_stat >/dev/null 2>&1; then
        # macOS
        available_memory=$(($(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.') * 4096 / 1024 / 1024))
    else
        available_memory=8192  # Assume 8GB
    fi
    
    if [ "$available_memory" -lt 4096 ]; then
        print_warning "Less than 4GB RAM available ($available_memory MB). Performance may be affected."
        print_info "Recommended: 8GB+ RAM for optimal performance"
    fi
    
    print_success "Prerequisites check completed"
}

# Function: Setup environment file
setup_environment() {
    print_step "Setting up environment configuration..."
    
    if [ -f "$ENV_FILE" ]; then
        print_warning "Environment file already exists at $ENV_FILE"
        read -p "Do you want to overwrite it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Using existing .env file"
            return 0
        fi
    fi
    
    if [ ! -f "$ENV_EXAMPLE" ]; then
        print_error "Template file .env.example not found!"
        exit 1
    fi
    
    print_step "Generating secure passwords..."
    
    # Generate secure passwords
    NEO4J_PASSWORD=$(generate_password)
    GRAFANA_PASSWORD=$(generate_password)
    
    # Copy template and replace placeholders
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    
    # Replace placeholder passwords with generated ones
    if command -v sed >/dev/null 2>&1; then
        sed -i.bak "s/your_secure_neo4j_password_here/$NEO4J_PASSWORD/g" "$ENV_FILE"
        sed -i.bak "s/your_grafana_password_here/$GRAFANA_PASSWORD/g" "$ENV_FILE"
        rm -f "$ENV_FILE.bak"
    else
        # Fallback for systems without sed
        cp "$ENV_FILE" "$ENV_FILE.bak"
        awk -v neo4j_pass="$NEO4J_PASSWORD" -v grafana_pass="$GRAFANA_PASSWORD" '
        {
            gsub(/your_secure_neo4j_password_here/, neo4j_pass);
            gsub(/your_grafana_password_here/, grafana_pass);
            print;
        }' "$ENV_FILE.bak" > "$ENV_FILE"
        rm -f "$ENV_FILE.bak"
    fi
    
    print_success "Environment file created with secure passwords"
    print_warning "IMPORTANT: Please edit $ENV_FILE to add your OpenAI API key and review settings"
    print_info "Neo4j password: $NEO4J_PASSWORD"
    print_info "Grafana password: $GRAFANA_PASSWORD"
}

# Function: Check model files
check_model_files() {
    print_step "Checking for model files..."
    
    local models_dir="$PROJECT_ROOT/models"
    local model_path="$models_dir/embeddinggemma-300m"
    
    if [ ! -d "$model_path" ]; then
        print_warning "EmbeddingGemma-300m model not found at $model_path"
        print_info "The service will attempt to download models at runtime"
        print_info "For faster startup, you can:"
        print_info "1. Create $models_dir directory"
        print_info "2. Download EmbeddingGemma-300m model files there"
        
        # Create models directory
        mkdir -p "$models_dir"
        print_success "Created models directory: $models_dir"
        
        return 0
    fi
    
    print_success "Model files found at $model_path"
}

# Function: Build and start services
start_services() {
    print_step "Building and starting AI Memory Service..."
    
    cd "$PROJECT_ROOT"
    
    # Pull latest base images
    print_step "Pulling latest base images..."
    docker compose pull neo4j || print_warning "Failed to pull Neo4j image"
    
    # Build the application
    print_step "Building AI Memory Service (this may take several minutes)..."
    if ! docker compose build ai-memory-service; then
        print_error "Failed to build AI Memory Service"
        print_info "Check the logs above for build errors"
        exit 1
    fi
    
    print_success "Build completed successfully"
    
    # Start services
    print_step "Starting services..."
    if ! docker compose up -d; then
        print_error "Failed to start services"
        print_info "Run 'docker compose logs' to see error details"
        exit 1
    fi
    
    print_success "Services started successfully"
}

# Function: Wait for services to be ready
wait_for_services() {
    print_step "Waiting for services to be ready..."
    
    # Wait for Neo4j
    print_step "Waiting for Neo4j database..."
    local max_attempts=60
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if docker compose exec -T neo4j cypher-shell -u neo4j -p "$(grep NEO4J_PASSWORD "$ENV_FILE" | cut -d'=' -f2)" "RETURN 1" >/dev/null 2>&1; then
            print_success "Neo4j is ready"
            break
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_warning "Neo4j taking longer than expected to start"
    fi
    
    # Wait for AI Memory Service
    print_step "Waiting for AI Memory Service..."
    attempt=0
    max_attempts=120  # 4 minutes for model loading
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
            print_success "AI Memory Service is ready"
            break
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_warning "AI Memory Service taking longer than expected to start"
        print_info "This is normal for first startup (model loading). Check logs: docker compose logs ai-memory-service"
    fi
}

# Function: Display service information
show_service_info() {
    print_success "🎉 AI Memory Service is running!"
    echo
    echo -e "${PURPLE}📋 Service Information:${NC}"
    echo -e "${GREEN}🌐 Main API:${NC}          http://localhost:8080"
    echo -e "${GREEN}🌐 Health Check:${NC}      http://localhost:8080/health"  
    echo -e "${GREEN}🗄️  Neo4j Browser:${NC}     http://localhost:7474"
    echo -e "${GREEN}👤 Neo4j Username:${NC}     neo4j"
    echo -e "${GREEN}🔑 Neo4j Password:${NC}     $(grep NEO4J_PASSWORD "$ENV_FILE" | cut -d'=' -f2)"
    echo
    echo -e "${BLUE}📊 Optional Monitoring (run with --monitoring):${NC}"
    echo -e "${BLUE}📈 Grafana:${NC}            http://localhost:3000 (admin/$(grep GRAFANA_PASSWORD "$ENV_FILE" | cut -d'=' -f2))"
    echo -e "${BLUE}🔍 Prometheus:${NC}         http://localhost:9090"
    echo
    echo -e "${YELLOW}🛠️  Management Commands:${NC}"
    echo -e "${YELLOW}📋 View logs:${NC}          docker compose logs -f"
    echo -e "${YELLOW}🛑 Stop services:${NC}      docker compose down"
    echo -e "${YELLOW}🔄 Restart:${NC}            docker compose restart"
    echo -e "${YELLOW}🧹 Clean restart:${NC}      docker compose down -v && ./scripts/quick-start.sh"
    echo
    echo -e "${GREEN}🧪 Test the service:${NC}"
    echo "curl http://localhost:8080/health"
    echo
    print_info "Edit .env file to configure OpenAI API key and other settings"
}

# Function: Main execution
main() {
    echo -e "${GREEN}🚀 AI Memory Service - Quick Start${NC}"
    echo -e "${GREEN}====================================${NC}"
    echo
    
    # Parse command line arguments
    local enable_monitoring=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --monitoring)
                enable_monitoring=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [--monitoring] [--help]"
                echo "  --monitoring    Enable Grafana and Prometheus monitoring"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Execute setup steps
    check_prerequisites
    setup_environment
    check_model_files
    start_services
    
    # Start monitoring if requested
    if [ "$enable_monitoring" = true ]; then
        print_step "Starting monitoring services..."
        docker compose --profile monitoring up -d
        print_success "Monitoring services started"
    fi
    
    wait_for_services
    show_service_info
    
    print_success "Setup completed successfully!"
}

# Run main function with all arguments
main "$@"