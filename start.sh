#!/bin/bash

# AI Memory Service - Quick Start Script
# This script helps you quickly start the AI Memory Service with all dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ASCII Art Banner
echo "
╔══════════════════════════════════════════════════════════════╗
║                    AI MEMORY SERVICE                         ║
║              Intelligent Memory System for AI                ║
║                  Powered by GPT-5-nano                      ║
╚══════════════════════════════════════════════════════════════╝
"

# Check prerequisites
print_info "Checking prerequisites..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if Rust is installed (for local development)
if command -v cargo &> /dev/null; then
    print_info "Rust is installed ($(cargo --version))"
else
    print_warn "Rust is not installed. You'll need it for local development."
fi

# Check if .env file exists
if [ ! -f .env ]; then
    print_warn ".env file not found. Creating from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        print_info "Created .env file. Please edit it with your configuration:"
        print_info "  - Set OPENAI_API_KEY for GPT-5-nano"
        print_info "  - Set NEO4J_PASSWORD for database"
        print_info "  - Adjust other settings as needed"
        echo ""
        read -p "Press Enter after you've configured .env file..." 
    else
        print_error ".env.example not found. Cannot create .env file."
        exit 1
    fi
fi

# Load environment variables
print_info "Loading environment variables..."
export $(grep -v '^#' .env | xargs)

# Validate critical environment variables
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "sk-your-openai-api-key-here" ]; then
    print_error "OPENAI_API_KEY is not set or still has the default value."
    print_error "Please edit .env file and set your OpenAI API key."
    exit 1
fi

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p data logs models cache config monitoring/prometheus monitoring/grafana/dashboards monitoring/grafana/datasources

# Check what mode to run in
echo ""
echo "Select run mode:"
echo "1) Docker Compose (Recommended for production)"
echo "2) Local Development (Rust + Python)"
echo "3) Docker Compose with Monitoring (Prometheus + Grafana)"
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        print_info "Starting services with Docker Compose..."
        
        # Build images if needed
        print_info "Building Docker images..."
        docker-compose build
        
        # Start services
        print_info "Starting services..."
        docker-compose up -d
        
        # Wait for services to be healthy
        print_info "Waiting for services to be healthy..."
        sleep 10
        
        # Check service health
        if curl -f http://localhost:8080/health &> /dev/null; then
            print_info "Memory Service is running!"
        else
            print_warn "Memory Service is still starting up..."
        fi
        
        if curl -f http://localhost:5000/health &> /dev/null; then
            print_info "Embedding Service is running!"
        else
            print_warn "Embedding Service is still starting up..."
        fi
        
        print_info "Services started successfully!"
        print_info "  - REST API: http://localhost:8080"
        print_info "  - Embedding Service: http://localhost:5000"
        print_info "  - Neo4j Browser: http://localhost:7474"
        print_info ""
        print_info "To view logs: docker-compose logs -f"
        print_info "To stop services: docker-compose down"
        ;;
        
    2)
        print_info "Starting in local development mode..."
        
        # Check if Python embedding service is running
        if ! curl -f http://localhost:5000/health &> /dev/null 2>&1; then
            print_info "Starting Python embedding service..."
            python embedding_service.py &
            EMBEDDING_PID=$!
            sleep 5
        else
            print_info "Embedding service is already running"
        fi
        
        # Build Rust project
        print_info "Building Rust project..."
        cargo build --release
        
        # Start main service
        print_info "Starting AI Memory Service..."
        cargo run --release --bin memory-server
        
        # Cleanup on exit
        if [ ! -z "$EMBEDDING_PID" ]; then
            kill $EMBEDDING_PID
        fi
        ;;
        
    3)
        print_info "Starting services with monitoring..."
        
        # Build images
        print_info "Building Docker images..."
        docker-compose build
        
        # Start all services including monitoring
        print_info "Starting services with monitoring stack..."
        docker-compose --profile monitoring up -d
        
        # Wait for services
        print_info "Waiting for services to be healthy..."
        sleep 15
        
        print_info "Services started successfully!"
        print_info "  - REST API: http://localhost:8080"
        print_info "  - Embedding Service: http://localhost:5000"
        print_info "  - Neo4j Browser: http://localhost:7474"
        print_info "  - Prometheus: http://localhost:9091"
        print_info "  - Grafana: http://localhost:3000 (admin/admin)"
        print_info ""
        print_info "To view logs: docker-compose logs -f"
        print_info "To stop services: docker-compose --profile monitoring down"
        ;;
        
    *)
        print_error "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
print_info "Quick test command:"
echo 'curl -X POST http://localhost:8080/api/v1/memory \
  -H "Content-Type: application/json" \
  -d "{\"content\": \"Test memory\", \"context_hint\": \"testing\"}"'