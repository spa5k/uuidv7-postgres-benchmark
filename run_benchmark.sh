#!/bin/bash

# UUIDv7 PostgreSQL Benchmark Runner
# Reproducible benchmark with consistent system specifications

set -e

echo "=========================================="
echo "UUIDv7 PostgreSQL Benchmark Runner"
echo "=========================================="

# System Specifications (for reproducibility)
echo "System Specifications:"
echo "  Docker Containers: 2GB RAM, 2 CPU cores each"
echo "  PostgreSQL Shared Buffers: 256MB"
echo "  PostgreSQL Work Memory: 4MB"
echo "  Effective Cache Size: 1GB"
echo "  Test Iterations: 5,000 single-threaded, 500x5 workers concurrent"
echo "  Python Requirements: psycopg3, matplotlib, seaborn, pandas"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    exit 1
fi

# Check if Python dependencies are installed
echo "Checking Python dependencies..."
python3 -c "
import sys
missing = []
try:
    import psycopg
except ImportError:
    missing.append('psycopg')
try:
    import matplotlib
except ImportError:
    missing.append('matplotlib')
try:
    import seaborn
except ImportError:
    missing.append('seaborn')
try:
    import pandas
except ImportError:
    missing.append('pandas')
    
if missing:
    print(f'❌ Missing Python packages: {missing}')
    print('Install with: pip install psycopg[binary] matplotlib seaborn pandas')
    sys.exit(1)
else:
    print('✅ All Python dependencies found')
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Clean up any existing containers
echo ""
echo "🧹 Cleaning up existing containers..."
docker-compose down -v 2>/dev/null || true

# Start PostgreSQL containers
echo ""
echo "🚀 Starting PostgreSQL containers..."
docker-compose up -d

# Wait for databases to be ready
echo ""
echo "⏳ Waiting for databases to be ready..."
echo "  PostgreSQL 17 (port 5434)..."
timeout=60
while ! docker exec postgres17-uuidv7 pg_isready -U postgres > /dev/null 2>&1; do
    sleep 2
    timeout=$((timeout - 2))
    if [ $timeout -le 0 ]; then
        echo "❌ PostgreSQL 17 failed to start"
        docker-compose logs postgres17
        exit 1
    fi
done
echo "  ✅ PostgreSQL 17 ready"

echo "  PostgreSQL 18 beta (port 5435)..."
timeout=60
while ! docker exec postgres18-uuidv7 pg_isready -U postgres > /dev/null 2>&1; do
    sleep 2
    timeout=$((timeout - 2))
    if [ $timeout -le 0 ]; then
        echo "❌ PostgreSQL 18 failed to start"
        docker-compose logs postgres18
        exit 1
    fi
done
echo "  ✅ PostgreSQL 18 beta ready"

# Give databases extra time to fully initialize
echo "  Waiting additional 10 seconds for full initialization..."
sleep 10

# Verify PostgreSQL versions and native UUIDv7 support
echo ""
echo "📋 Verifying PostgreSQL versions..."
echo "  PostgreSQL 17:"
docker exec postgres17-uuidv7 psql -U postgres -d benchmark -c "SELECT version();" | grep PostgreSQL || echo "    ❌ Failed to get version"

echo "  PostgreSQL 18:"
docker exec postgres18-uuidv7 psql -U postgres -d benchmark -c "SELECT version();" | grep PostgreSQL || echo "    ❌ Failed to get version"

echo "  Checking native UUIDv7 support in PostgreSQL 18:"
if docker exec postgres18-uuidv7 psql -U postgres -d benchmark -c "SELECT uuidv7();" > /dev/null 2>&1; then
    echo "    ✅ Native uuidv7() function available"
else
    echo "    ❌ Native uuidv7() function not available"
fi

# Run the comprehensive benchmark
echo ""
echo "🏃 Running comprehensive benchmark..."
echo "  This will take approximately 5-10 minutes..."
echo ""

python3 benchmark_pg18_native.py

echo ""
echo "✅ Benchmark completed successfully!"
echo ""
echo "📊 Results saved to:"
echo "  - enhanced_benchmark_results.json (detailed results)"
echo "  - benchmark_data/performance_summary.json (summary)"
echo "  - benchmark_data/chart_data.json (chart data)"
echo "  - enhanced_postgresql_benchmark.png (visualizations)"
echo ""
echo "🧹 Cleaning up containers..."
docker-compose down

echo ""
echo "✅ Benchmark complete! Check the files above for results."