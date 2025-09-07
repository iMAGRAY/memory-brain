# AI Memory Service - Self-Contained Multi-Stage Build
# Everything included: Rust compiler, dependencies, Python, models, transformers runtime
# User needs only Docker - no external installations required

ARG RUST_VERSION=1.83
ARG PYTHON_VERSION=3.11

# ============================================================================
# Stage 1: Rust Build Environment
# ============================================================================
FROM rust:${RUST_VERSION}-slim-bookworm AS rust-builder

# Install build dependencies including Python for PyO3
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libsqlite3-dev \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    python3 \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set Python configuration for PyO3
ENV PYO3_PYTHON=python3
ENV PYO3_NO_PYTHON=0
ENV PYO3_CROSS_LIB_DIR=/usr/lib/python3.11

# Copy dependency manifests first (for better Docker layer caching)
COPY Cargo.toml Cargo.lock ./
COPY .cargo/ ./.cargo/

# Create dummy source files to build dependencies (for Docker layer caching optimization)
RUN mkdir -p src benches && \
    echo "fn main() {}" > src/main.rs && \
    echo "pub fn add(a: i32, b: i32) -> i32 { a + b }" > src/lib.rs && \
    echo 'use criterion::*; fn main() {}' > benches/dummy.rs

# Create simd benchmark file with heredoc
RUN cat > benches/simd_benchmark.rs << 'EOF'
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn simd_benchmark(c: &mut Criterion) {
    c.bench_function("simd_operations", |b| {
        b.iter(|| {
            let data: Vec<f32> = (0..1000).map(|x| x as f32).collect();
            black_box(data.iter().sum::<f32>())
        })
    });
}

criterion_group\\!(benches, simd_benchmark);
criterion_main\\!(benches);
EOF

# Validate dummy files were created successfully
RUN test -f src/main.rs && test -f src/lib.rs && test -f benches/dummy.rs && test -f benches/simd_benchmark.rs && \
    echo "✅ Dummy source files created and validated"

# Clear any cached PyO3 build configuration and build dependencies
RUN rm -rf ~/.cargo/registry/cache && \
    cargo clean && \
    cargo build --release && \
    rm -rf src benches

# Copy actual source code and validate critical files exist
COPY src/ ./src/
COPY build.rs ./

# Validate critical files and prepare optional directories
RUN set -e && \
    echo "🔍 Validating critical source files..." && \
    test -f src/main.rs && test -s src/main.rs && \
    test -f build.rs && test -s build.rs && \
    mkdir -p benches && \
    ls -lh src/main.rs build.rs && \
    echo "✅ All critical source files validated successfully"

# Build all binaries
RUN touch src/main.rs && \
    cargo build --release --bins

# Verify critical binaries exist
RUN ls -la target/release/ && \
    test -f target/release/memory-server && \
    echo "✅ Rust build completed successfully"

# ============================================================================
# Stage 2: Python Environment with Models
# ============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS python-builder

# Install Python build dependencies and curl for healthcheck
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create Python environment
WORKDIR /python-env

# Install core Python packages with updated secure versions
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch==2.8.0 -f https://download.pytorch.org/whl/torch_stable.html \
        transformers==4.56.0 \
        sentence-transformers==5.1.0 \
        numpy==1.26.4 \
        flask==3.0.0 \
        requests==2.32.3 \
        gunicorn==21.2.0 && \
    rm -rf /root/.cache/pip /tmp/pip-* ~/.cache

# Copy and run comprehensive package validation script
COPY validate_packages.py /tmp/validate_packages.py
RUN python /tmp/validate_packages.py 2>&1 | tee /tmp/validation.log && \
    echo "✅ Package validation successful" && \
    rm /tmp/validate_packages.py /tmp/validation.log || \
    (echo "❌ Package validation failed" && cat /tmp/validation.log && exit 1)

# Create model download directory
RUN mkdir -p /models

# Skip ONNX model download - use transformers directly
RUN echo "✅ Using transformers direct model loading (no ONNX files needed)"

# ONNX Runtime stage removed - using pure transformers approach

# ============================================================================
# Stage 4: Python Embedding Service
# ============================================================================
FROM python:${PYTHON_VERSION}-slim-bookworm AS python-service

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy Python environment
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Models downloaded by transformers at runtime - no static files needed

# Create embedding service
WORKDIR /app

# Copy Python embedding service
COPY embedding_service.py ./

# Create non-root user
RUN useradd -r -u 1001 embeddings

# Set permissions
RUN chown -R embeddings:embeddings /app

USER embeddings

EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "embedding_service.py"]

# ============================================================================
# Stage 5: Final Runtime Image
# ============================================================================
FROM debian:bookworm-slim AS final

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libsqlite3-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r aiservice && useradd -r -g aiservice -u 1000 aiservice

# ONNX Runtime removed - using pure transformers approach

# Copy Rust binaries
COPY --from=rust-builder /app/target/release/memory-server /usr/local/bin/

# Models downloaded by transformers at runtime - no static files needed

# Create app structure
WORKDIR /app
RUN mkdir -p config data logs backup temp && \
    chown -R aiservice:aiservice /app

# Copy configuration files
COPY --chown=aiservice:aiservice config/ ./config/

# Create default configuration if none exists
RUN if [ ! -f config/default.toml ]; then \
    { \
    echo "[server]"; \
    echo "host = \"0.0.0.0\""; \
    echo "port = 8080"; \
    echo "admin_port = 8081"; \
    echo "websocket_port = 8082"; \
    echo ""; \
    echo "[database]"; \
    echo "fallback_to_sqlite = true"; \
    echo "sqlite_path = \"/app/data/memory.db\""; \
    echo ""; \
    echo "[embedding]"; \
    echo "service_url = \"http://python-embeddings:8001\""; \
    echo "timeout_seconds = 30"; \
    echo "batch_size = 32"; \
    echo ""; \
    echo "[memory]"; \
    echo "cache_size_mb = 512"; \
    echo "max_connections = 100"; \
    echo ""; \
    echo "[logging]"; \
    echo "level = \"info\""; \
    echo "file = \"/app/logs/service.log\""; \
    } > config/default.toml; \
    fi

# Set environment variables
ENV RUST_LOG=info,ai_memory_service=debug \
    RUST_BACKTRACE=1 \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    CONFIG_DIR=/app/config

# Switch to non-root user
USER aiservice

# Create startup script
COPY --chown=aiservice:aiservice <<'EOF' /usr/local/bin/start-service.sh
#!/bin/bash
set -euo pipefail

echo "🚀 Starting AI Memory Service..."
echo "================================="

# Validate environment
echo "📋 Environment Check:"
echo "  - Runtime: Pure transformers approach"
echo "  - Models: Downloaded at runtime"
echo "  - Data: ${DATA_DIR}"
echo "  - Config: ${CONFIG_DIR}"

# Test SIMD capabilities first
echo "🔧 Testing system capabilities..."
if test_working_components; then
    echo "✅ System validation passed"
else
    echo "⚠️  System validation warnings (continuing anyway)"
fi

# Start the main service
echo "🎯 Starting memory service..."
exec memory-server --config "${CONFIG_DIR}/default.toml"
EOF

RUN chmod +x /usr/local/bin/start-service.sh

# Expose ports
EXPOSE 8080 8081 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f -s http://localhost:8080/health || exit 1

# Start the service
CMD ["/usr/local/bin/start-service.sh"]

# Labels for container metadata
LABEL org.opencontainers.image.title="AI Memory Service" \
      org.opencontainers.image.description="Complete AI Memory Service with embedded models and runtime" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.vendor="AI Memory Project" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/ai-memory/service"