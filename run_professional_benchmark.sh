#!/bin/bash

# Professional PostgreSQL UUIDv7 Benchmark Suite Runner
# Automated setup and execution with comprehensive error handling

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/results/logs"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Create log directory
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo -e "${BLUE}$(date '+%Y-%m-%d %H:%M:%S')${NC} | $1" | tee -a "$LOG_DIR/setup_$TIMESTAMP.log"
}

error() {
    echo -e "${RED}$(date '+%Y-%m-%d %H:%M:%S')${NC} | ERROR: $1" | tee -a "$LOG_DIR/setup_$TIMESTAMP.log"
}

success() {
    echo -e "${GREEN}$(date '+%Y-%m-%d %H:%M:%S')${NC} | ✅ $1" | tee -a "$LOG_DIR/setup_$TIMESTAMP.log"
}

warning() {
    echo -e "${YELLOW}$(date '+%Y-%m-%d %H:%M:%S')${NC} | ⚠️  $1" | tee -a "$LOG_DIR/setup_$TIMESTAMP.log"
}

print_banner() {
    echo -e "${BOLD}${BLUE}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║            Professional PostgreSQL UUIDv7 Benchmark Suite Runner            ║
║                                                                              ║
║  Automated setup and execution with professional-grade methodology          ║
║                                                                              ║
║  Features:                                                                   ║
║  • Automated Docker container management                                    ║
║  • Environment validation and optimization                                  ║
║  • High-precision benchmark execution (50k+ iterations)                    ║
║  • Comprehensive statistical analysis                                       ║
║  • Professional reporting and data export                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

check_dependencies() {
    log "Checking system dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    # Check Python version (minimum 3.8)
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        error "Python 3.8+ required, found Python $python_version"
        exit 1
    fi
    
    success "All dependencies are available"
}

setup_python_environment() {
    log "Setting up Python environment..."
    
    cd "$SCRIPT_DIR"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    log "Upgrading pip..."
    pip install --upgrade pip > /dev/null 2>&1
    
    # Install requirements
    log "Installing Python requirements..."
    pip install -r requirements.txt > /dev/null 2>&1 || {
        error "Failed to install Python requirements"
        exit 1
    }
    
    success "Python environment ready"
}

manage_docker_containers() {
    log "Managing Docker containers..."
    
    cd "$SCRIPT_DIR"
    
    # Clean up any existing containers
    log "Stopping existing containers..."
    docker-compose down -v > /dev/null 2>&1 || true
    
    # Start containers
    log "Starting PostgreSQL containers..."
    docker-compose up -d || {
        error "Failed to start Docker containers"
        exit 1
    }
    
    # Wait for containers to be ready
    log "Waiting for PostgreSQL containers to be ready..."
    
    # PostgreSQL 17
    log "Checking PostgreSQL 17 (port 5434)..."
    max_attempts=60
    attempt=0
    while ! docker exec postgres17-uuidv7 pg_isready -U postgres > /dev/null 2>&1; do
        sleep 2
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            error "PostgreSQL 17 failed to start within $((max_attempts * 2)) seconds"
            docker-compose logs postgres17
            exit 1
        fi
    done
    success "PostgreSQL 17 is ready"
    
    # PostgreSQL 18
    log "Checking PostgreSQL 18 (port 5435)..."
    attempt=0
    while ! docker exec postgres18-uuidv7 pg_isready -U postgres > /dev/null 2>&1; do
        sleep 2
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            error "PostgreSQL 18 failed to start within $((max_attempts * 2)) seconds"
            docker-compose logs postgres18
            exit 1
        fi
    done
    success "PostgreSQL 18 is ready"
    
    # Additional wait for full initialization
    log "Waiting for full database initialization..."
    sleep 10
    
    success "All PostgreSQL containers are ready"
}

verify_database_functions() {
    log "Verifying database functions..."
    
    # Test PostgreSQL 17
    log "Testing PostgreSQL 17 functions..."
    if docker exec postgres17-uuidv7 psql -U postgres -d benchmark -c "SELECT uuid_generate_v7();" > /dev/null 2>&1; then
        success "PostgreSQL 17 UUIDv7 functions working"
    else
        error "PostgreSQL 17 UUIDv7 functions not working"
        exit 1
    fi
    
    # Test PostgreSQL 18 and native UUIDv7
    log "Testing PostgreSQL 18 functions..."
    if docker exec postgres18-uuidv7 psql -U postgres -d benchmark -c "SELECT uuid_generate_v7();" > /dev/null 2>&1; then
        success "PostgreSQL 18 custom UUIDv7 functions working"
    else
        error "PostgreSQL 18 custom UUIDv7 functions not working"
        exit 1
    fi
    
    log "Testing PostgreSQL 18 native UUIDv7..."
    if docker exec postgres18-uuidv7 psql -U postgres -d benchmark -c "SELECT uuidv7();" > /dev/null 2>&1; then
        success "PostgreSQL 18 native uuidv7() function working"
    else
        warning "PostgreSQL 18 native uuidv7() function not available"
    fi
}

run_benchmark() {
    local mode="$1"
    log "Running benchmark in $mode mode..."
    
    cd "$SCRIPT_DIR"
    source venv/bin/activate
    
    case "$mode" in
        "quick")
            python3 professional_benchmark.py --quick --verbose
            ;;
        "extensive")
            python3 professional_benchmark.py --extensive --verbose
            ;;
        "standard"|*)
            python3 professional_benchmark.py --verbose
            ;;
    esac
}

cleanup() {
    log "Cleaning up..."
    cd "$SCRIPT_DIR"
    
    # Stop containers
    docker-compose down > /dev/null 2>&1 || true
    
    success "Cleanup completed"
}

show_usage() {
    echo "Usage: $0 [mode]"
    echo ""
    echo "Modes:"
    echo "  quick      - Quick benchmark (5K iterations, 3 runs)"
    echo "  standard   - Standard benchmark (50K iterations, 5 runs) [default]"
    echo "  extensive  - Extensive benchmark (100K iterations, 10 runs)"
    echo ""
    echo "Examples:"
    echo "  $0              # Run standard benchmark"
    echo "  $0 quick        # Run quick benchmark"
    echo "  $0 extensive    # Run extensive benchmark"
}

main() {
    local mode="${1:-standard}"
    
    # Validate mode
    case "$mode" in
        "quick"|"standard"|"extensive")
            ;;
        "help"|"-h"|"--help")
            show_usage
            exit 0
            ;;
        *)
            error "Invalid mode: $mode"
            show_usage
            exit 1
            ;;
    esac
    
    print_banner
    
    log "Starting Professional PostgreSQL UUIDv7 Benchmark Suite"
    log "Mode: $mode"
    log "Timestamp: $TIMESTAMP"
    
    # Set up trap for cleanup on exit
    trap cleanup EXIT
    
    # Main execution flow
    check_dependencies
    setup_python_environment
    manage_docker_containers
    verify_database_functions
    
    # Run the benchmark
    local benchmark_start=$(date +%s)
    run_benchmark "$mode"
    local benchmark_end=$(date +%s)
    local duration=$((benchmark_end - benchmark_start))
    
    # Show results
    echo ""
    echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${GREEN}║                          BENCHMARK COMPLETED SUCCESSFULLY!                  ║${NC}"
    echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    success "Benchmark completed in $duration seconds"
    success "Results saved to: $SCRIPT_DIR/results/"
    
    echo ""
    echo -e "${BOLD}📁 Generated Files:${NC}"
    echo "  📊 Charts: results/charts/"
    echo "  📋 Reports: results/reports/"
    echo "  💾 Raw Data: results/raw_data/"
    echo "  📈 CSV Export: results/exports/"
    echo "  📝 Logs: results/logs/"
    
    echo ""
    echo -e "${BOLD}📈 Quick Access:${NC}"
    if [ -d "$SCRIPT_DIR/results/reports" ]; then
        latest_report=$(ls -t "$SCRIPT_DIR/results/reports"/*.md 2>/dev/null | head -1)
        if [ -n "$latest_report" ]; then
            echo "  Latest Report: $latest_report"
        fi
    fi
    
    if [ -d "$SCRIPT_DIR/results/charts" ]; then
        latest_chart=$(ls -t "$SCRIPT_DIR/results/charts"/*.png 2>/dev/null | head -1)
        if [ -n "$latest_chart" ]; then
            echo "  Latest Chart: $latest_chart"
        fi
    fi
    
    echo ""
    log "Professional benchmark suite completed successfully!"
}

# Run main function with all arguments
main "$@"