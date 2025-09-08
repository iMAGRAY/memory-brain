# AI Memory Service - Production-Ready Self-Contained Build
# Secure, optimized Rust service with integrated PyO3 Python embeddings
# Zero hardcoded credentials, environment-driven configuration

ARG RUST_VERSION=1.83
ARG PYTHON_VERSION=3.11

# ============================================================================
# Stage 1: Rust Build Environment with PyO3 Integration
# ============================================================================
FROM rust:${RUST_VERSION}-slim-bookworm AS rust-builder

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libsqlite3-dev \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python runtime and development dependencies for PyO3
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for PyO3 integration
RUN pip3 install --no-cache-dir --break-system-packages \
    torch==2.8.0 \
    transformers==4.56.0 \
    sentence-transformers==5.1.0 \
    numpy==1.26.4 \
    && rm -rf ~/.cache/pip

# Set up PyO3 build environment
ENV PYO3_PYTHON=python3
ENV PYO3_NO_PYTHON=0
ENV PYO3_CROSS_LIB_DIR=/usr/lib/python3.11
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages

WORKDIR /app

# Copy dependency manifests for Docker layer caching optimization
COPY Cargo.toml Cargo.lock ./
COPY .cargo/ ./.cargo/

# Create dummy source files to pre-build dependencies (caching optimization)
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    echo "// Dummy lib for dependency caching" > src/lib.rs

# Pre-build dependencies with dummy sources for better caching
RUN cargo build --release
RUN rm -rf src

# Copy actual source code
COPY src/ ./src/
COPY build.rs ./build.rs

# Validate critical files exist before building
RUN test -f src/main.rs && test -s src/main.rs || (echo "❌ src/main.rs missing or empty" && exit 1)
RUN test -f build.rs && test -s build.rs || (echo "❌ build.rs missing or empty" && exit 1)

# Build final Rust binaries with PyO3 integration
RUN touch src/main.rs && \
    PYTHONPATH=/usr/local/lib/python3.11/site-packages cargo build --release --bins

# Verify critical binaries were built successfully
RUN ls -la target/release/ && \
    test -f target/release/memory-server || (echo "❌ memory-server binary not found" && exit 1) && \
    echo "✅ Rust build with PyO3 embedding integration completed successfully"

# ============================================================================
# Stage 2: Final Runtime Image
# ============================================================================
FROM debian:bookworm-slim AS final

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libsqlite3-0 \
    libgomp1 \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python runtime dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
    torch==2.8.0 \
    transformers==4.56.0 \
    sentence-transformers==5.1.0 \
    numpy==1.26.4 \
    && rm -rf ~/.cache/pip

# Create non-root user for security
RUN groupadd -r aiservice && useradd -r -g aiservice -u 1000 aiservice

# Copy Rust binary from build stage
COPY --from=rust-builder /app/target/release/memory-server /usr/local/bin/

# Create app structure and set permissions
WORKDIR /app
RUN mkdir -p config data logs models cache && \
    chown -R aiservice:aiservice /app

# Copy configuration template (no secrets)
COPY --chown=aiservice:aiservice config/ ./config/

# Set up environment variables (no secrets)
ENV RUST_LOG=info,ai_memory_service=debug \
    RUST_BACKTRACE=1 \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    CONFIG_DIR=/app/config \
    MODEL_DIR=/app/models \
    CACHE_DIR=/app/cache \
    PYTHONPATH=/usr/local/lib/python3.11/site-packages

# Switch to non-root user
USER aiservice

# Create secure startup script with environment-based configuration
RUN cat > /app/start-service.sh << 'EOF'
#!/bin/bash
set -euo pipefail

echo "🚀 Starting AI Memory Service with Security Best Practices..."
echo "=============================================================="

# Validate environment
echo "📋 Environment Check:"
echo "  - Runtime: Integrated PyO3 Python embeddings"
echo "  - Models: ${MODEL_DIR}"
echo "  - Data: ${DATA_DIR}"
echo "  - Config: ${CONFIG_DIR}"
echo "  - Cache: ${CACHE_DIR}"

# Security validation - ensure no hardcoded credentials
if [ -z "${NEO4J_PASSWORD:-}" ]; then
    echo "❌ NEO4J_PASSWORD environment variable is required"
    exit 1
fi

if [ -z "${NEO4J_URI:-}" ]; then
    export NEO4J_URI="bolt://neo4j:7687"
    echo "⚠️  Using default NEO4J_URI: ${NEO4J_URI}"
fi

# Validate critical binaries
echo "🔧 Validating system components..."
if command -v memory-server >/dev/null 2>&1; then
    echo "✅ Memory server binary found"
else
    echo "❌ Memory server binary not found"
    exit 1
fi

if python3 -c "import torch, transformers, sentence_transformers, numpy" 2>/dev/null; then
    echo "✅ Python dependencies validated"
else
    echo "❌ Python dependencies validation failed"
    exit 1
fi

# Create secure configuration from environment variables
if [ ! -f "${CONFIG_DIR}/config.toml" ]; then
    echo "📝 Creating secure configuration from environment..."
    cat > "${CONFIG_DIR}/config.toml" << CONFIG_EOF
[server]
host = "${SERVICE_HOST:-0.0.0.0}"
port = ${SERVICE_PORT:-8080}
workers = ${WORKERS:-4}
environment = "${ENVIRONMENT:-production}"
cors_origins = ["${CORS_ORIGINS:-*}"]

[storage]
neo4j_uri = "${NEO4J_URI}"
neo4j_user = "${NEO4J_USER:-neo4j}"
neo4j_password = "${NEO4J_PASSWORD}"
connection_pool_size = ${NEO4J_POOL_SIZE:-10}

[embedding]
model_path = "${EMBEDDING_MODEL_PATH:-${MODEL_DIR}/embeddinggemma-300m}"
tokenizer_path = "${TOKENIZER_PATH:-${MODEL_DIR}/embeddinggemma-300m/tokenizer.json}"
batch_size = ${EMBEDDING_BATCH_SIZE:-32}
max_sequence_length = ${MAX_SEQUENCE_LENGTH:-2048}
embedding_dimension = ${EMBEDDING_DIMENSION:-512}
normalize_embeddings = ${NORMALIZE_EMBEDDINGS:-true}
precision = "${EMBEDDING_PRECISION:-float32}"
use_specialized_prompts = ${USE_SPECIALIZED_PROMPTS:-true}

[cache]
l1_size = ${L1_CACHE_SIZE:-1000}
l2_size = ${L2_CACHE_SIZE:-10000}
ttl_seconds = ${CACHE_TTL:-3600}
compression_enabled = ${CACHE_COMPRESSION:-true}

[brain]
max_memories = ${MAX_MEMORIES:-100000}
importance_threshold = ${IMPORTANCE_THRESHOLD:-0.3}
consolidation_interval = ${CONSOLIDATION_INTERVAL:-300}
decay_rate = ${MEMORY_DECAY_RATE:-0.01}
CONFIG_EOF
    echo "✅ Secure configuration created"
fi

# Validate model directory exists if specified
if [ ! -z "${EMBEDDING_MODEL_PATH:-}" ] && [ ! -d "${EMBEDDING_MODEL_PATH}" ]; then
    echo "❌ Model directory not found: ${EMBEDDING_MODEL_PATH}"
    echo "💡 Please mount your model directory to ${MODEL_DIR} or set EMBEDDING_MODEL_PATH"
    exit 1
fi

# Start the main service
echo "🎯 Starting memory service..."
exec memory-server --config "${CONFIG_DIR}/config.toml"
EOF

RUN chmod +x /app/start-service.sh

# Expose ports
EXPOSE 8080

# Health check with proper timeout
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f -s --max-time 10 http://localhost:8080/health || exit 1

# Volume mounts for user data (no sensitive config in image)
VOLUME ["/app/data", "/app/logs", "/app/models", "/app/cache"]

# Start the service
CMD ["/app/start-service.sh"]

# Security-focused container metadata
LABEL org.opencontainers.image.title="AI Memory Service" \
      org.opencontainers.image.description="Secure AI Memory Service with integrated PyO3 embeddings and environment-driven configuration" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.vendor="AI Memory Project" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/ai-memory/service" \
      security.credentials="environment-only" \
      security.user="non-root" \
      security.scanning="required"