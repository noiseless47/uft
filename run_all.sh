#!/bin/bash

# Football Squad Selection - Complete Pipeline Runner
# Usage: ./run_all.sh [--dry-run] [--fast-demo] [--help]

set -e  # Exit on any error

# Configuration
PROJECT_NAME="football-squad-selection"
DOCKER_IMAGE="$PROJECT_NAME:latest"
DOCKER_DEV_IMAGE="$PROJECT_NAME:dev"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Football Squad Selection Pipeline Runner

Usage: $0 [OPTIONS]

OPTIONS:
    --dry-run       Show what would be executed without running
    --fast-demo     Run with synthetic data (15-30 minutes)
    --help          Show this help message
    --dev           Run in development mode with Jupyter
    --no-docker     Run locally without Docker
    --gpu           Enable GPU support (if available)

EXAMPLES:
    $0                    # Full pipeline (4-6 hours)
    $0 --fast-demo        # Quick demo with synthetic data
    $0 --dry-run          # Preview commands
    $0 --dev              # Development environment

ENVIRONMENT VARIABLES:
    MLFLOW_TRACKING_URI   MLflow tracking server
    DATA_PATH             Custom data directory
    N_JOBS                Number of parallel jobs
EOF
}

# Parse command line arguments
DRY_RUN=false
FAST_DEMO=false
DEV_MODE=false
NO_DOCKER=false
GPU_SUPPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --fast-demo)
            FAST_DEMO=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --no-docker)
            NO_DOCKER=true
            shift
            ;;
        --gpu)
            GPU_SUPPORT=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if [[ "$NO_DOCKER" == false ]]; then
        if ! command -v docker &> /dev/null; then
            error "Docker is required but not installed"
            exit 1
        fi
        
        if ! docker info &> /dev/null; then
            error "Docker daemon is not running"
            exit 1
        fi
    else
        if ! command -v python3 &> /dev/null; then
            error "Python 3 is required but not installed"
            exit 1
        fi
    fi
    
    # Check disk space (need ~20GB)
    available_space=$(df . | tail -1 | awk '{print $4}')
    required_space=20971520  # 20GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        warning "Low disk space. Required: 20GB, Available: $(($available_space/1024/1024))GB"
    fi
    
    success "Prerequisites check passed"
}

# Build Docker image
build_docker() {
    if [[ "$NO_DOCKER" == true ]]; then
        return 0
    fi
    
    log "Building Docker image..."
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "Would run: docker build -t $DOCKER_IMAGE -f docker/Dockerfile ."
        return 0
    fi
    
    if [[ "$DEV_MODE" == true ]]; then
        docker build -t $DOCKER_DEV_IMAGE -f docker/Dockerfile --target dev .
    else
        docker build -t $DOCKER_IMAGE -f docker/Dockerfile --target prod .
    fi
    
    success "Docker image built successfully"
}

# Setup environment
setup_environment() {
    log "Setting up environment..."
    
    # Create necessary directories
    mkdir -p data/{raw,processed,features,manifests,sample_small}
    mkdir -p experiments/{mlflow,optuna_studies,runs}
    mkdir -p artifacts
    mkdir -p logs
    mkdir -p paper/{figures,tables,supplement}
    
    # Copy sample environment file
    if [[ ! -f .env && -f .env.example ]]; then
        cp .env.example .env
        warning "Created .env from .env.example. Please configure API keys if needed."
    fi
    
    success "Environment setup complete"
}

# Data preparation
prepare_data() {
    log "Preparing data..."
    
    local cmd="python src/data/prepare_data.py"
    
    if [[ "$FAST_DEMO" == true ]]; then
        cmd="$cmd --synthetic --n_matches 100 --n_players 500"
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "Would run: $cmd"
        return 0
    fi
    
    if [[ "$NO_DOCKER" == true ]]; then
        $cmd
    else
        docker run --rm -v "$(pwd):/app" $DOCKER_IMAGE $cmd
    fi
    
    success "Data preparation complete"
}

# Stage 1 training
train_stage1() {
    log "Training Stage-1 models (RF, XGBoost, LightGBM)..."
    
    local cmd="python src/models/stage1/train.py"
    
    if [[ "$FAST_DEMO" == true ]]; then
        cmd="$cmd --fast --n_trials 10"
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "Would run: $cmd"
        return 0
    fi
    
    local docker_args=""
    if [[ "$GPU_SUPPORT" == true ]]; then
        docker_args="--gpus all"
    fi
    
    if [[ "$NO_DOCKER" == true ]]; then
        $cmd
    else
        docker run --rm $docker_args -v "$(pwd):/app" $DOCKER_IMAGE $cmd
    fi
    
    success "Stage-1 training complete"
}

# Stage 2 training
train_stage2() {
    log "Training Stage-2 models (GBM + Matrix Factorization)..."
    
    local cmd="python src/models/stage2/train.py"
    
    if [[ "$FAST_DEMO" == true ]]; then
        cmd="$cmd --fast"
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "Would run: $cmd"
        return 0
    fi
    
    if [[ "$NO_DOCKER" == true ]]; then
        $cmd
    else
        docker run --rm -v "$(pwd):/app" $DOCKER_IMAGE $cmd
    fi
    
    success "Stage-2 training complete"
}

# Run simulation
run_simulation() {
    log "Running Monte Carlo simulation..."
    
    local cmd="python src/simulation/monte_carlo.py"
    
    if [[ "$FAST_DEMO" == true ]]; then
        cmd="$cmd --n_sims 1000"
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "Would run: $cmd"
        return 0
    fi
    
    if [[ "$NO_DOCKER" == true ]]; then
        $cmd
    else
        docker run --rm -v "$(pwd):/app" $DOCKER_IMAGE $cmd
    fi
    
    success "Simulation complete"
}

# Generate figures and tables
generate_outputs() {
    log "Generating figures and tables..."
    
    local cmd="python src/visualization/generate_all.py"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "Would run: $cmd"
        return 0
    fi
    
    if [[ "$NO_DOCKER" == true ]]; then
        $cmd
    else
        docker run --rm -v "$(pwd):/app" $DOCKER_IMAGE $cmd
    fi
    
    success "Figures and tables generated"
}

# Run tests
run_tests() {
    log "Running validation tests..."
    
    local cmd="python -m pytest tests/ -v"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "Would run: $cmd"
        return 0
    fi
    
    if [[ "$NO_DOCKER" == true ]]; then
        $cmd
    else
        docker run --rm -v "$(pwd):/app" $DOCKER_IMAGE $cmd
    fi
    
    success "Tests completed"
}

# Development mode
run_dev_mode() {
    log "Starting development environment..."
    
    if [[ "$NO_DOCKER" == true ]]; then
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
    else
        docker run --rm -p 8888:8888 -v "$(pwd):/app" $DOCKER_DEV_IMAGE
    fi
}

# Main execution
main() {
    log "Starting Football Squad Selection Pipeline"
    log "Mode: $([ "$FAST_DEMO" == true ] && echo "Fast Demo" || echo "Full Pipeline")"
    log "Docker: $([ "$NO_DOCKER" == true ] && echo "Disabled" || echo "Enabled")"
    
    # Record start time
    start_time=$(date +%s)
    
    # Run pipeline steps
    check_prerequisites
    
    if [[ "$DEV_MODE" == true ]]; then
        build_docker
        setup_environment
        run_dev_mode
        return 0
    fi
    
    build_docker
    setup_environment
    prepare_data
    train_stage1
    train_stage2
    run_simulation
    generate_outputs
    run_tests
    
    # Calculate runtime
    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    hours=$((runtime / 3600))
    minutes=$(((runtime % 3600) / 60))
    seconds=$((runtime % 60))
    
    success "Pipeline completed successfully!"
    log "Total runtime: ${hours}h ${minutes}m ${seconds}s"
    
    # Show key outputs
    log "Key outputs:"
    log "  - Models: artifacts/"
    log "  - Figures: paper/figures/"
    log "  - Tables: paper/tables/"
    log "  - MLflow: experiments/mlflow/"
    
    if [[ "$FAST_DEMO" == true ]]; then
        log "This was a fast demo. Run without --fast-demo for full results."
    fi
}

# Trap for cleanup
cleanup() {
    log "Cleaning up..."
    # Add any cleanup tasks here
}
trap cleanup EXIT

# Run main function
main "$@"
