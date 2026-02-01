#!/bin/bash

set -e  # Exit on any error

# Configuration
VERSION="${SYSTEM_BRAIN_VERSION:-latest}"
COMPOSE_PROJECT="karma-police"  
SERVICE_NAME="system-brain"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(dirname "$0")"
PROJECT_ROOT="$SCRIPT_DIR/../.."
SYSTEM_BRAIN_DIR="$SCRIPT_DIR/.."
SRC_DIR="$SYSTEM_BRAIN_DIR/src"
MODELS_DIR="$SYSTEM_BRAIN_DIR/models"
LOG_DIR="$SYSTEM_BRAIN_DIR/logs"
DATA_DIR="$SYSTEM_BRAIN_DIR/data"

# Ensure directories exist
mkdir -p "$LOG_DIR" "$DATA_DIR" "$MODELS_DIR"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${PURPLE}========================================${NC}"
    echo -e "${WHITE}   $1   ${NC}"
    echo -e "${PURPLE}========================================${NC}\n"
}

# Utility functions
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose."
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Get docker compose command (handles both docker-compose and docker compose)
get_compose_cmd() {
    if command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    else
        echo "docker compose"
    fi
}

get_container_id() {
    local service=$1
    docker ps -q -f "name=${COMPOSE_PROJECT}-${service}" 2>/dev/null | head -1 || echo ""
}

get_container_name() {
    local service=$1
    docker ps --format "{{.Names}}" -f "name=${COMPOSE_PROJECT}-${service}" 2>/dev/null | head -1 || echo ""
}

wait_for_container() {
    local service=$1
    local timeout=${2:-300}
    local elapsed=0
    local interval=5

    log_info "Waiting for service $service to be ready (timeout: ${timeout}s)..."

    while [ $elapsed -lt $timeout ]; do
        if docker ps -q -f "name=${COMPOSE_PROJECT}-${service}" -f "status=running" | grep -q .; then
            log_success "Service $service is running"
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    log_error "Service $service failed to become ready within ${timeout}s"
    return 1
}

save_logs() {
    local service=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="$LOG_DIR/${service}_${timestamp}.log"

    local container=$(get_container_name $service)
    if [ -n "$container" ]; then
        log_info "Saving logs for $service to $log_file"
        docker logs $container > "$log_file" 2>&1
        log_success "Logs saved to $log_file"
    else
        log_warn "No container found for service $service"
    fi
}

build_image() {
    log_info "Building system-brain Docker image..."
    cd "$PROJECT_ROOT"
    docker build -f Dockerfile.system-brain -t system-brain:${VERSION} .
    log_success "Image built: system-brain:${VERSION}"
}

log_header "SYSTEM'S BRAIN DOCKER DEPLOYMENT v$VERSION"

case "$1" in
    "prereq"|"prerequisites")
        check_prerequisites
        ;;

    "build")
        log_header "BUILDING DOCKER IMAGE"
        check_prerequisites
        build_image
        ;;

    "up"|"start")
        log_header "STARTING SERVICES"
        check_prerequisites
        
        COMPOSE_CMD=$(get_compose_cmd)
        cd "$PROJECT_ROOT"
        
        # Build if requested
        if [ "$2" = "--build" ] || [ "$2" = "-b" ]; then
            log_info "Building images..."
            $COMPOSE_CMD build system-brain
        fi
        
        log_info "Starting system-brain service..."
        $COMPOSE_CMD up -d system-brain
        
        wait_for_container "system-brain" 60
        log_success "System-brain service started"
        ;;

    "down"|"stop")
        log_header "STOPPING SERVICES"
        check_prerequisites
        
        COMPOSE_CMD=$(get_compose_cmd)
        cd "$PROJECT_ROOT"
        
        if [ "$2" = "--volumes" ] || [ "$2" = "-v" ]; then
            log_info "Stopping services and removing volumes..."
            $COMPOSE_CMD down -v
        else
            log_info "Stopping services..."
            $COMPOSE_CMD down
        fi
        
        log_success "Services stopped"
        ;;

    "restart")
        log_header "RESTARTING SERVICE"
        check_prerequisites
        
        SERVICE="${2:-system-brain}"
        save_logs "$SERVICE"
        
        COMPOSE_CMD=$(get_compose_cmd)
        cd "$PROJECT_ROOT"
        
        log_info "Restarting $SERVICE..."
        $COMPOSE_CMD restart $SERVICE
        
        wait_for_container "$SERVICE" 60
        log_success "$SERVICE restarted"
        ;;

    "data"|"collect"|"collect-data")
        log_header "BULK DATA COLLECTION"
        check_prerequisites

        # Defaults tuned for large collection
        DURATION=120
        RUNS=8

        # Parse optional flags
        shift
        while [ $# -gt 0 ]; do
            case "$1" in
                --duration)
                    DURATION="$2"
                    shift 2
                    ;;
                --duration=*)
                    DURATION="${1#*=}"
                    shift 1
                    ;;
                --runs)
                    RUNS="$2"
                    shift 2
                    ;;
                --runs=*)
                    RUNS="${1#*=}"
                    shift 1
                    ;;
                *)
                    log_warn "Unknown option '$1' (ignoring)"
                    shift 1
                    ;;
            esac
        done

        log_info "Using sample duration=${DURATION}s, runs per config=${RUNS}"
        
        log_info "Running data collection in Docker container..."
        docker run --rm \
            --name system-brain-data-collection \
            --gpus all \
            -v "$SRC_DIR:/workspace:ro" \
            -v "$DATA_DIR:/data:rw" \
            -v "$MODELS_DIR:/models:rw" \
            nvcr.io/nvidia/pytorch:24.09-py3 \
            bash -c "
                set -e
                echo '[SETUP] Installing required packages...'
                pip install --no-cache-dir scikit-learn numpy==1.26.4 psutil xgboost
                
                echo '[SETUP] Copying Python modules...'
                mkdir -p /app
                cp /workspace/*.py /app/
                
                echo '[SETUP] Starting telemetry collection'
                cd /app
                python collect_telemetry.py &
                TELEMETRY_PID=\$!
                
                echo '[INFO] Running data generation step'
                python -u pipeline_runner.py data --force-new-data --duration ${DURATION} --runs ${RUNS}
                
                kill \$TELEMETRY_PID 2>/dev/null || true
                echo '[OK] Data collection complete'
            "
        
        log_success "Data collection completed"
        ;;

    "train"|"training")
        log_header "TRAINING MODEL"
        check_prerequisites
        
        log_info "Running model training in Docker container..."
        docker run --rm \
            --name system-brain-training \
            --gpus all \
            -v "$SRC_DIR:/workspace:ro" \
            -v "$DATA_DIR:/data:rw" \
            -v "$MODELS_DIR:/models:rw" \
            pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
            bash -c "
                set -e
                echo '[SETUP] Installing required packages...'
                pip install --no-cache-dir scikit-learn numpy==1.26.4 psutil xgboost
                
                echo '[SETUP] Copying Python modules...'
                mkdir -p /app
                cp /workspace/*.py /app/
                
                echo '[INFO] Starting model training'
                cd /app
                python -u pipeline_runner.py train
                
                echo '[OK] Training complete'
            "
        
        log_success "Model training completed"
        ;;

    "test"|"testing")
        log_header "TESTING MODEL"
        check_prerequisites
        
        log_info "Running model tests in Docker container..."
        docker run --rm \
            --name system-brain-testing \
            --gpus all \
            -v "$SRC_DIR:/workspace:ro" \
            -v "$DATA_DIR:/data:ro" \
            -v "$MODELS_DIR:/models:ro" \
            pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
            bash -c "
                set -e
                echo '[SETUP] Installing required packages...'
                pip install --no-cache-dir scikit-learn numpy==1.26.4 psutil
                
                echo '[SETUP] Copying Python modules...'
                mkdir -p /app
                cp /workspace/*.py /app/
                
                echo '[INFO] Running model tests'
                cd /app
                python model_tester.py --test all
                
                echo '[OK] Testing complete'
            "
        
        log_success "Model testing completed"
        ;;

    "pipeline"|"all")
        log_header "RUNNING COMPLETE ML PIPELINE"
        check_prerequisites

        log_info "Running complete pipeline in Docker container..."
        
        FOLLOW_LOGS=""
        if [ "$2" = "--follow" ] || [ "$2" = "-f" ]; then
            FOLLOW_LOGS="-t"
        fi
        
        docker run --rm $FOLLOW_LOGS \
            --name system-brain-pipeline \
            --gpus all \
            -v "$SRC_DIR:/workspace:ro" \
            -v "$DATA_DIR:/data:rw" \
            -v "$MODELS_DIR:/models:rw" \
            nvcr.io/nvidia/pytorch:24.09-py3 \
            bash -c "
                set -e
                echo '[SETUP] Installing required packages...'
                pip install --no-cache-dir scikit-learn numpy==1.26.4 psutil xgboost
                
                echo '[SETUP] Copying Python modules...'
                mkdir -p /app
                cp /workspace/*.py /app/
                
                echo '[PIPELINE] Starting telemetry collection'
                cd /app
                python collect_telemetry.py &
                TELEMETRY_PID=\$!
                
                echo '[PIPELINE] Running complete pipeline'
                python -u pipeline_runner.py all
                
                kill \$TELEMETRY_PID 2>/dev/null || true
                echo '[OK] Pipeline complete'
            "
        
        log_success "Pipeline completed"
        ;;

    "demo")
        log_header "RUNNING LIVE DEMO"
        check_prerequisites
        
        log_info "Starting live demo..."
        docker run -d \
            --name system-brain-demo \
            --gpus all \
            -v "$SRC_DIR:/workspace:ro" \
            -v "$MODELS_DIR:/models:ro" \
            -p 5000:5000 \
            pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime \
            bash -c "
                set -e
                pip install --no-cache-dir scikit-learn numpy==1.26.4 psutil flask
                mkdir -p /app
                cp /workspace/*.py /app/
                cd /app
                python live_classifier_service.py --models-path /models --port 5000
            "
        
        wait_for_container "demo" 60
        log_success "Demo started at http://localhost:5000"
        log_info "View logs with: $0 logs demo -f"
        ;;

    "status")
        log_header "DEPLOYMENT STATUS"
        
        echo -e "${CYAN}Docker Containers:${NC}"
        docker ps --filter "name=${COMPOSE_PROJECT}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        
        echo ""
        echo -e "${CYAN}Docker Volumes:${NC}"
        docker volume ls --filter "name=${COMPOSE_PROJECT}" --format "table {{.Name}}\t{{.Driver}}"
        
        echo ""
        echo -e "${CYAN}Docker Images:${NC}"
        docker images --filter "reference=system-brain" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
        ;;

    "logs")
        SERVICE="${2:-system-brain}"
        FOLLOW="${3:-}"
        
        log_info "Showing logs for service: $SERVICE"
        
        CONTAINER=$(get_container_name $SERVICE)
        if [ -z "$CONTAINER" ]; then
            # Try running container by name
            CONTAINER="system-brain-$SERVICE"
        fi
        
        if [ "$FOLLOW" = "-f" ] || [ "$FOLLOW" = "follow" ]; then
            docker logs -f --tail 100 $CONTAINER 2>/dev/null || log_error "Container not found: $CONTAINER"
        else
            docker logs --tail 100 $CONTAINER 2>/dev/null || log_error "Container not found: $CONTAINER"
        fi
        ;;

    "clean")
        log_header "CLEANING UP"
        check_prerequisites

        CLEAN_TYPE="${2:-basic}"
        COMPOSE_CMD=$(get_compose_cmd)
        cd "$PROJECT_ROOT"

        case $CLEAN_TYPE in
            "all")
                log_warn "Performing FULL cleanup (including volumes and data)..."
                read -p "Are you sure you want to delete ALL data? (yes/no): " confirmation
                if [ "$confirmation" = "yes" ]; then
                    log_info "Stopping containers..."
                    $COMPOSE_CMD down -v
                    
                    log_info "Removing system-brain containers..."
                    docker ps -aq --filter "name=system-brain" | xargs -r docker rm -f
                    
                    log_info "Removing system-brain images..."
                    docker images -q system-brain | xargs -r docker rmi -f
                    
                    log_info "Cleaning up data directories..."
                    rm -rf "$DATA_DIR"/* "$LOG_DIR"/*
                    
                    log_success "Full cleanup complete"
                else
                    log_info "Cleanup cancelled"
                fi
                ;;
            "containers")
                log_info "Removing stopped system-brain containers..."
                docker ps -aq --filter "name=system-brain" --filter "status=exited" | xargs -r docker rm
                log_success "Containers cleanup complete"
                ;;
            *)
                log_info "Performing basic cleanup (stopped containers)..."
                docker ps -aq --filter "name=system-brain" --filter "status=exited" | xargs -r docker rm
                log_success "Basic cleanup complete"
                ;;
        esac
        ;;

    "shell"|"exec")
        log_header "INTERACTIVE SHELL"
        SERVICE="${2:-system-brain}"
        
        CONTAINER=$(get_container_name $SERVICE)
        if [ -z "$CONTAINER" ]; then
            log_error "No running container found for service: $SERVICE"
            exit 1
        fi

        log_info "Opening shell in container: $CONTAINER"
        docker exec -it $CONTAINER /bin/bash
        ;;

    "monitor"|"watch")
        log_header "MONITORING DASHBOARD"
        INTERVAL="${2:-5}"

        log_info "Starting monitoring dashboard (refresh every ${INTERVAL}s, press Ctrl+C to exit)"

        while true; do
            clear
            echo -e "${PURPLE}========================================${NC}"
            echo -e "${WHITE}   SYSTEM'S BRAIN MONITORING DASHBOARD   ${NC}"
            echo -e "${PURPLE}========================================${NC}"
            echo -e "Refresh interval: ${INTERVAL}s | $(date)"
            echo

            # Containers status
            echo -e "${CYAN}CONTAINERS:${NC}"
            docker ps --filter "name=system-brain" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "  No containers found"

            echo ""
            echo -e "${CYAN}RESOURCE USAGE:${NC}"
            docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" \
                $(docker ps -q --filter "name=system-brain") 2>/dev/null || echo "  No running containers"

            sleep $INTERVAL
        done
        ;;

    "health"|"healthcheck")
        log_header "HEALTH CHECK"

        log_info "Performing comprehensive health check..."

        # Check Docker daemon
        if docker info &>/dev/null; then
            log_success "✓ Docker daemon running"
        else
            log_error "✗ Docker daemon not running"
        fi

        # Check system-brain service
        COMPOSE_CMD=$(get_compose_cmd)
        cd "$PROJECT_ROOT"
        
        if $COMPOSE_CMD ps system-brain 2>/dev/null | grep -q "Up"; then
            log_success "✓ system-brain service running"
        else
            log_warn "⚠ system-brain service not running"
        fi

        # Check volumes
        VOLUME_COUNT=$(docker volume ls --filter "name=${COMPOSE_PROJECT}" --format "{{.Name}}" 2>/dev/null | wc -l)
        if [ "$VOLUME_COUNT" -gt 0 ]; then
            log_success "✓ Volumes found: $VOLUME_COUNT"
        else
            log_warn "⚠ No volumes found"
        fi

        # Check running containers
        CONTAINER_COUNT=$(docker ps --filter "name=system-brain" --format "{{.Names}}" 2>/dev/null | wc -l)
        if [ "$CONTAINER_COUNT" -gt 0 ]; then
            log_success "✓ Running containers: $CONTAINER_COUNT"
        else
            log_warn "⚠ No running containers"
        fi

        # Check disk space
        DISK_USAGE=$(docker system df --format "{{.Type}}\t{{.Size}}" 2>/dev/null)
        echo ""
        echo -e "${CYAN}DISK USAGE:${NC}"
        echo "$DISK_USAGE"

        log_success "Health check completed"
        ;;

    "prune")
        log_header "PRUNING DOCKER RESOURCES"
        
        log_warn "This will remove unused Docker resources (containers, networks, images)"
        read -p "Continue? (yes/no): " confirmation
        
        if [ "$confirmation" = "yes" ]; then
            log_info "Pruning Docker system..."
            docker system prune -f
            log_success "Pruning complete"
        else
            log_info "Prune cancelled"
        fi
        ;;

    "backup")
        log_header "BACKUP DATA"
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        BACKUP_DIR="$LOG_DIR/backup_$TIMESTAMP"

        log_info "Creating backup in: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"

        # Backup models and data
        if [ -d "$MODELS_DIR" ] && [ "$(ls -A $MODELS_DIR)" ]; then
            log_info "Backing up models..."
            tar -czf "$BACKUP_DIR/models.tar.gz" -C "$SYSTEM_BRAIN_DIR" models
        fi

        if [ -d "$DATA_DIR" ] && [ "$(ls -A $DATA_DIR)" ]; then
            log_info "Backing up data..."
            tar -czf "$BACKUP_DIR/data.tar.gz" -C "$SYSTEM_BRAIN_DIR" data
        fi

        # Backup Docker Compose configuration
        if [ -f "$PROJECT_ROOT/docker-compose.yml" ]; then
            cp "$PROJECT_ROOT/docker-compose.yml" "$BACKUP_DIR/"
        fi

        log_success "Backup completed: $BACKUP_DIR"
        ;;

    "restore")
        log_header "RESTORE FROM BACKUP"
        BACKUP_PATH="${2:-}"

        if [ -z "$BACKUP_PATH" ] || [ ! -d "$BACKUP_PATH" ]; then
            log_error "Please provide a valid backup directory"
            echo "Usage: $0 restore <backup_dir>"
            exit 1
        fi

        log_warn "This will overwrite existing models and data"
        read -p "Continue? (yes/no): " confirmation

        if [ "$confirmation" = "yes" ]; then
            if [ -f "$BACKUP_PATH/models.tar.gz" ]; then
                log_info "Restoring models..."
                tar -xzf "$BACKUP_PATH/models.tar.gz" -C "$SYSTEM_BRAIN_DIR"
            fi

            if [ -f "$BACKUP_PATH/data.tar.gz" ]; then
                log_info "Restoring data..."
                tar -xzf "$BACKUP_PATH/data.tar.gz" -C "$SYSTEM_BRAIN_DIR"
            fi

            log_success "Restore completed"
        else
            log_info "Restore cancelled"
        fi
        ;;

    "debug")
        log_header "DEBUG INFORMATION"
        SERVICE="${2:-system-brain}"

        log_info "Collecting debug information for service: $SERVICE"

        CONTAINER=$(get_container_name $SERVICE)
        if [ -n "$CONTAINER" ]; then
            echo -e "\n${CYAN}CONTAINER INFORMATION:${NC}"
            docker inspect $CONTAINER

            echo -e "\n${CYAN}CONTAINER LOGS (last 50 lines):${NC}"
            docker logs --tail=50 $CONTAINER

            echo -e "\n${CYAN}CONTAINER PROCESSES:${NC}"
            docker exec $CONTAINER ps aux 2>/dev/null || log_warn "Cannot access container processes"

            echo -e "\n${CYAN}CONTAINER RESOURCES:${NC}"
            docker stats --no-stream $CONTAINER 2>/dev/null || log_warn "Cannot get resource usage"

        else
            log_error "No container found for service: $SERVICE"
        fi

        # Save debug info to file
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        DEBUG_FILE="$LOG_DIR/debug_${SERVICE}_${TIMESTAMP}.log"
        {
            echo "=== DEBUG INFORMATION FOR $SERVICE ==="
            echo "Timestamp: $(date)"
            echo "Service: $SERVICE"
            echo "Container: $CONTAINER"
            echo
            docker inspect $CONTAINER 2>&1
            echo
            echo "=== LOGS ==="
            docker logs $CONTAINER 2>&1
        } > "$DEBUG_FILE"

        log_success "Debug information saved to: $DEBUG_FILE"
        ;;

    *)
        echo -e "${WHITE}SYSTEM'S BRAIN DOCKER DEPLOYMENT TOOL v$VERSION${NC}"
        echo ""
        echo -e "${CYAN}USAGE:${NC} $0 <command> [options]"
        echo ""
        echo -e "${YELLOW}DEPLOYMENT COMMANDS:${NC}"
        echo "  prereq              - Check prerequisites (Docker, Docker Compose)"
        echo "  build               - Build system-brain Docker image"
        echo "  up/start [-b]       - Start services (--build to rebuild)"
        echo "  down/stop [-v]      - Stop services (--volumes to remove volumes)"
        echo "  restart [service]   - Restart a service"
        echo ""
        echo -e "${YELLOW}ML PIPELINE COMMANDS:${NC}"
        echo "  data/collect        - Run bulk data collection"
        echo "    --duration <sec>  - Sample duration (default: 120)"
        echo "    --runs <n>        - Runs per config (default: 8)"
        echo "  train/training      - Run model training"
        echo "  test/testing        - Run comprehensive model testing"
        echo "  pipeline/all [-f]   - Run complete ML pipeline (--follow for logs)"
        echo "  demo                - Start live demo service"
        echo ""
        echo -e "${YELLOW}MONITORING & DEBUGGING:${NC}"
        echo "  status              - Check deployment status"
        echo "  logs [svc] [-f]     - View logs (e.g., logs system-brain -f)"
        echo "  monitor/watch [int] - Real-time monitoring dashboard"
        echo "  health/healthcheck  - Comprehensive health check"
        echo "  debug [service]     - Collect debug information"
        echo ""
        echo -e "${YELLOW}MAINTENANCE COMMANDS:${NC}"
        echo "  clean [type]        - Cleanup (basic|containers|all)"
        echo "  prune               - Prune unused Docker resources"
        echo "  backup              - Backup data and models"
        echo "  restore <dir>       - Restore from backup"
        echo "  shell/exec [svc]    - Interactive shell in container"
        echo ""
        echo -e "${GREEN}QUICK START:${NC}"
        echo "  1. $0 prereq        ${BLUE}# Check prerequisites${NC}"
        echo "  2. $0 build         ${BLUE}# Build Docker image${NC}"
        echo "  3. $0 up            ${BLUE}# Start services${NC}"
        echo "  4. $0 pipeline -f   ${BLUE}# Run pipeline with logs${NC}"
        echo ""
        echo -e "${GREEN}MONITORING WORKFLOW:${NC}"
        echo "  $0 monitor          ${BLUE}# Watch dashboard${NC}"
        echo "  $0 logs system-brain -f ${BLUE}# Follow logs${NC}"
        echo "  $0 status           ${BLUE}# Check status${NC}"
        echo ""
        echo -e "${GREEN}DEBUG WORKFLOW:${NC}"
        echo "  $0 health           ${BLUE}# Check system health${NC}"
        echo "  $0 debug system-brain ${BLUE}# Collect debug info${NC}"
        echo "  $0 shell system-brain ${BLUE}# Access container shell${NC}"
        echo ""
        echo -e "${RED}CLEANUP:${NC}"
        echo "  $0 clean containers ${BLUE}# Clean stopped containers${NC}"
        echo "  $0 clean all        ${BLUE}# Full cleanup (DANGER!)${NC}"
        echo "  $0 prune            ${BLUE}# Prune Docker system${NC}"
        echo ""
        echo -e "${PURPLE}Examples:${NC}"
        echo "  $0 up --build                     # Start with fresh build"
        echo "  $0 data --duration 60 --runs 5    # Custom data collection"
        echo "  $0 pipeline --follow              # Run pipeline and follow logs"
        echo "  $0 monitor 3                      # Monitor with 3s refresh"
        echo "  $0 logs system-brain -f           # Follow logs"
        echo "  $0 shell system-brain             # Get shell in container"
        echo ""
        exit 1
        ;;
esac
