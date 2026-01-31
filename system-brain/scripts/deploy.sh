#!/bin/bash

set -e  # Exit on any error

# Configuration
NAMESPACE="eidf219ns"
TEAM="teamovercooked"
VERSION="${SYSTEM_BRAIN_VERSION:-latest}"

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
K8S_DIR="$SCRIPT_DIR/../k8s"
SRC_DIR="$SCRIPT_DIR/../src"
LOG_DIR="$SCRIPT_DIR/../logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

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

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi



    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log_error "Namespace $NAMESPACE not found."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

get_pod_name() {
    local component=$1
    kubectl -n $NAMESPACE get pods -l component=$component --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || echo ""
}

get_latest_job() {
    local component=$1
    kubectl -n $NAMESPACE get jobs -l component=$component --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || echo ""
}

wait_for_pod() {
    local pod_name=$1
    local timeout=${2:-300}

    log_info "Waiting for pod $pod_name to be ready (timeout: ${timeout}s)..."

    if kubectl -n $NAMESPACE wait --for=condition=Ready pod/$pod_name --timeout=${timeout}s; then
        log_success "Pod $pod_name is ready"
        return 0
    else
        log_error "Pod $pod_name failed to become ready within ${timeout}s"
        return 1
    fi
}

save_logs() {
    local component=$1
    local pod_name=$(get_pod_name $component)
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="$LOG_DIR/${component}_${timestamp}.log"

    if [ -n "$pod_name" ]; then
        log_info "Saving logs for $component to $log_file"
        kubectl -n $NAMESPACE logs $pod_name > "$log_file" 2>&1
        log_success "Logs saved to $log_file"
    else
        log_warn "No pod found for component $component"
    fi
}

log_header "SYSTEM'S BRAIN K8S DEPLOYMENT v$VERSION"

case "$1" in
    "prereq"|"prerequisites")
        check_prerequisites
        ;;

    "setup"|"configmap")
        log_header "CREATING CONFIGMAP FROM SOURCES"
        check_prerequisites
        log_info "Creating ConfigMap from Python sources..."
        bash $SCRIPT_DIR/create_configmap.sh
        log_success "ConfigMap created successfully"

        # Verify ConfigMap contents
        log_info "Verifying ConfigMap contents..."
        kubectl -n $NAMESPACE describe configmap teamovercooked-ml-src | grep -E "Data|=====" || true
        ;;



    "overnight")
        echo "Deploying Overnight Training CronJob..."
        # First ensure ConfigMap is up to date
        bash $SCRIPT_DIR/create_configmap.sh
        # Deploy cronjob
        kubectl apply -f $K8S_DIR/overnight-training.yaml
        echo "Overnight training CronJob deployed"
        echo "Check status: kubectl -n $NAMESPACE get cronjobs"
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
        log_info "Updating ConfigMap before launch..."
        bash $SCRIPT_DIR/create_configmap.sh

        JOB_YAML=$(mktemp)
        cat > $JOB_YAML <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  generateName: teamovercooked-data-
  namespace: ${NAMESPACE}
  labels:
    app: teamovercooked
    component: data-collection
    kueue.x-k8s.io/queue-name: ${NAMESPACE}-user-queue
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: teamovercooked
        component: data-collection
    spec:
      restartPolicy: Never
      containers:
      - name: data-collector
        image: nvcr.io/nvidia/pytorch:24.09-py3
        imagePullPolicy: IfNotPresent
        command: ["/bin/bash", "-c"]
        args:
        - |
          set -e

          echo "[SETUP] Installing required packages..."
          pip install --no-cache-dir scikit-learn numpy==1.26.4 psutil xgboost

          echo "[SETUP] Copying Python modules from ConfigMap..."
          cp /config/*.py /workspace/

          echo "[SETUP] Starting telemetry collection"
          python /workspace/collect_telemetry.py &
          TELEMETRY_PID=$!

          echo "[INFO] Running data generation step"
          cd /workspace
          python -u pipeline_runner.py data --force-new-data --duration ${DURATION} --runs ${RUNS}

          kill $TELEMETRY_PID 2>/dev/null || true
          echo "[OK] Data collection complete"
        resources:
          requests:
            cpu: "8"
            memory: "32Gi"
            nvidia.com/gpu: 1
          limits:
            cpu: "16"
            memory: "64Gi"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: models-volume
          mountPath: /models
        - name: data-volume
          mountPath: /data
        - name: config
          mountPath: /config
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: teamovercooked-models-pvc
      - name: data-volume
        persistentVolumeClaim:
          claimName: teamovercooked-data-pvc
      - name: config
        configMap:
          name: teamovercooked-ml-src
EOF

        log_info "Submitting bulk data collection job..."
        kubectl create -f $JOB_YAML
        rm $JOB_YAML
        log_success "Data collection job submitted"
        log_info "Follow logs with: ./deploy.sh logs data-collection -f"
        ;;

    "demo")
        echo "[ACTION] Deploying Live Demo..."
        # First ensure ConfigMap is up to date
        bash $SCRIPT_DIR/create_configmap.sh
        kubectl apply -f $K8S_DIR/live-demo.yaml
        echo "[OK] Live demo deployed"
        echo "Check logs: kubectl -n $NAMESPACE logs -l app=teamovercooked-demo -f"
        ;;

    "pipeline"|"all")
        log_header "RUNNING COMPLETE ML PIPELINE"
        check_prerequisites

        # First ensure ConfigMap is up to date
        log_info "Updating ConfigMap..."
        bash $SCRIPT_DIR/create_configmap.sh

        # Deploy complete pipeline using create for generateName
        log_info "Deploying pipeline..."
        kubectl create -f $K8S_DIR/pipeline.yaml

        # Wait for pod to be created and get its name
        sleep 5
        POD_NAME=$(get_pod_name pipeline)
        if [ -n "$POD_NAME" ]; then
            log_success "Pipeline deployed: $POD_NAME"

            # Option to wait and follow logs
            if [ "$2" == "--follow" ] || [ "$2" == "-f" ]; then
                log_info "Following pipeline logs..."
                kubectl -n $NAMESPACE logs -f $POD_NAME
            else
                log_info "Pipeline is running. Use './deploy.sh logs pipeline -f' to follow logs"
                log_info "Check status with: './deploy.sh status'"
            fi
        else
            log_error "Failed to get pipeline pod name"
        fi
        ;;

    "test"|"testing")
        echo "[ACTION] Running Model Tests..."
        # First ensure ConfigMap is up to date
        bash $SCRIPT_DIR/create_configmap.sh
        # Create testing job
        JOB_YAML=$(mktemp)
        cat > $JOB_YAML <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  generateName: teamovercooked-testing-
  namespace: eidf219ns
  labels:
    app: teamovercooked
    component: testing
    kueue.x-k8s.io/queue-name: eidf219ns-user-queue
spec:
  backoffLimit: 2
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        nvidia.com/gpu.present: "true"
      containers:
      - name: testing
        image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
        command: ["/bin/bash", "-c"]
        args:
          - |
            pip install numpy==1.26.4 scikit-learn
            cp /src/*.py /workspace/
            cd /workspace
            python model_tester.py --test all
        volumeMounts:
        - name: src
          mountPath: /src
        - name: models
          mountPath: /models
        - name: data
          mountPath: /data
        resources:
          requests:
            cpu: 2
            memory: 4Gi
            nvidia.com/gpu: 1
          limits:
            cpu: 4
            memory: 8Gi
            nvidia.com/gpu: 1
      volumes:
      - name: src
        configMap:
          name: teamovercooked-ml-src
      - name: models
        persistentVolumeClaim:
          claimName: teamovercooked-models-pvc
      - name: data
        persistentVolumeClaim:
          claimName: teamovercooked-data-pvc
EOF
        kubectl create -f $JOB_YAML
        rm $JOB_YAML
        echo "[OK] Testing job deployed"
        echo "Check logs: kubectl -n $NAMESPACE logs -l component=testing -f"
        ;;

    "pvc"|"storage")
        echo "Creating Persistent Storage..."
        kubectl apply -f $K8S_DIR/pvc.yaml
        echo "PVCs created"
        echo "Check status: kubectl -n $NAMESPACE get pvc"
        ;;

    "status")
        echo "Checking Deployment Status..."
        echo ""
        echo "Jobs:"
        kubectl -n $NAMESPACE get jobs | grep -E "(teamovercooked|$TEAM)"
        echo ""
        echo "CronJobs:"
        kubectl -n $NAMESPACE get cronjobs | grep teamovercooked
        echo ""
        echo "Pods:"
        kubectl -n $NAMESPACE get pods | grep -E "(teamovercooked|$TEAM)"
        echo ""
        echo "PVCs:"
        kubectl -n $NAMESPACE get pvc | grep $TEAM
        ;;

    "logs")
        COMPONENT="${2:-training}"
        FOLLOW="${3:-}"
        echo "Showing logs for component: $COMPONENT"

        # Build log command
        LOG_CMD="kubectl -n $NAMESPACE logs"
        if [ "$FOLLOW" = "-f" ] || [ "$FOLLOW" = "follow" ]; then
            LOG_FLAGS="--tail=50 -f"
        else
            LOG_FLAGS="--tail=100"
        fi

        # Try by label first, then fall back to getting pod name
        if kubectl -n $NAMESPACE get pods -l component=$COMPONENT 2>/dev/null | grep -q teamovercooked; then
            $LOG_CMD -l component=$COMPONENT $LOG_FLAGS
        else
            # Get the most recent teamovercooked pod
            POD=$(kubectl -n $NAMESPACE get pods | grep teamovercooked | tail -1 | awk '{print $1}')
            if [ -n "$POD" ]; then
                echo "Showing logs for pod: $POD"
                $LOG_CMD $POD $LOG_FLAGS
            else
                echo "No teamovercooked pods found"
            fi
        fi
        ;;

    "clean")
        log_header "CLEANING UP DEPLOYMENTS"
        check_prerequisites

        CLEAN_TYPE="${2:-basic}"

        case $CLEAN_TYPE in
            "all")
                log_warn "Performing FULL cleanup (including PVCs)..."
                read -p "Are you sure you want to delete ALL data? (yes/no): " confirmation
                if [ "$confirmation" = "yes" ]; then
                    kubectl -n $NAMESPACE delete jobs -l app=teamovercooked --ignore-not-found=true
                    kubectl -n $NAMESPACE delete cronjobs -l app=teamovercooked --ignore-not-found=true
                    kubectl -n $NAMESPACE delete deployments -l app=teamovercooked --ignore-not-found=true
                    kubectl -n $NAMESPACE delete configmap teamovercooked-ml-src --ignore-not-found=true
                    kubectl -n $NAMESPACE delete pvc -l app=teamovercooked --ignore-not-found=true
                    log_success "Full cleanup complete"
                else
                    log_info "Cleanup cancelled"
                fi
                ;;
            "jobs")
                log_info "Cleaning up jobs only..."
                kubectl -n $NAMESPACE delete jobs -l app=teamovercooked --ignore-not-found=true
                log_success "Jobs cleanup complete"
                ;;
            *)
                log_info "Performing basic cleanup (jobs, cronjobs, deployments, configmap)..."
                kubectl -n $NAMESPACE delete jobs -l app=teamovercooked --ignore-not-found=true
                kubectl -n $NAMESPACE delete cronjobs -l app=teamovercooked --ignore-not-found=true
                kubectl -n $NAMESPACE delete deployments -l app=teamovercooked --ignore-not-found=true
                kubectl -n $NAMESPACE delete configmap teamovercooked-ml-src --ignore-not-found=true
                log_success "Basic cleanup complete"
                ;;
        esac
        ;;

    "restart")
        log_header "RESTARTING COMPONENT"
        COMPONENT="${2:-pipeline}"
        check_prerequisites

        log_info "Restarting $COMPONENT..."

        # Save logs before restart
        save_logs $COMPONENT

        # Delete current job/deployment
        kubectl -n $NAMESPACE delete jobs -l component=$COMPONENT --ignore-not-found=true

        # Redeploy based on component
        case $COMPONENT in
            "pipeline")
                log_info "Redeploying pipeline..."
                bash $SCRIPT_DIR/create_configmap.sh
                kubectl create -f $K8S_DIR/pipeline.yaml
                ;;
            *)
                log_error "Unknown component: $COMPONENT"
                exit 1
                ;;
        esac

        log_success "$COMPONENT restarted"
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

            # Jobs status
            echo -e "${CYAN}JOBS:${NC}"
            kubectl -n $NAMESPACE get jobs -l app=teamovercooked --no-headers 2>/dev/null | while read line; do
                if [[ $line == *"1/1"* ]]; then
                    echo -e "  ${GREEN}✓${NC} $line"
                elif [[ $line == *"0/1"* ]]; then
                    echo -e "  ${YELLOW}⏳${NC} $line"
                else
                    echo -e "  ${RED}✗${NC} $line"
                fi
            done || echo "  No jobs found"

            echo

            # Pods status
            echo -e "${CYAN}PODS:${NC}"
            kubectl -n $NAMESPACE get pods -l app=teamovercooked --no-headers 2>/dev/null | while read line; do
                if [[ $line == *"Running"* ]]; then
                    echo -e "  ${GREEN}✓${NC} $line"
                elif [[ $line == *"Pending"* ]] || [[ $line == *"ContainerCreating"* ]]; then
                    echo -e "  ${YELLOW}⏳${NC} $line"
                elif [[ $line == *"Completed"* ]]; then
                    echo -e "  ${BLUE}✓${NC} $line"
                else
                    echo -e "  ${RED}✗${NC} $line"
                fi
            done || echo "  No pods found"

            echo

            # PVCs status
            echo -e "${CYAN}STORAGE:${NC}"
            kubectl -n $NAMESPACE get pvc -l app=teamovercooked --no-headers 2>/dev/null | while read line; do
                if [[ $line == *"Bound"* ]]; then
                    echo -e "  ${GREEN}✓${NC} $line"
                else
                    echo -e "  ${YELLOW}⏳${NC} $line"
                fi
            done || echo "  No PVCs found"

            sleep $INTERVAL
        done
        ;;

    "telemetry")
        log_header "TELEMETRY INSPECTOR"
        COMPONENT="${2:-pipeline}"
        POD_NAME=$(get_pod_name $COMPONENT)

        if [ -z "$POD_NAME" ]; then
            log_error "No pod found for component: $COMPONENT"
            exit 1
        fi

        log_info "Inspecting telemetry from pod: $POD_NAME"

        # Check if telemetry file exists
        if kubectl -n $NAMESPACE exec $POD_NAME -- test -f /tmp/system_eye_metrics.json 2>/dev/null; then
            log_success "Telemetry file found"

            # Show metadata
            echo -e "\n${CYAN}TELEMETRY METADATA:${NC}"
            kubectl -n $NAMESPACE exec $POD_NAME -- cat /tmp/system_eye_metrics.json | head -10

            # Show sample count
            SAMPLE_COUNT=$(kubectl -n $NAMESPACE exec $POD_NAME -- cat /tmp/system_eye_metrics.json | grep -o '"sample_count": [0-9]*' | grep -o '[0-9]*' || echo "0")
            log_info "Current sample count: $SAMPLE_COUNT"

            # Show file size
            FILE_SIZE=$(kubectl -n $NAMESPACE exec $POD_NAME -- stat -c%s /tmp/system_eye_metrics.json 2>/dev/null || echo "0")
            log_info "Telemetry file size: $FILE_SIZE bytes"

        else
            log_warn "Telemetry file not found at /tmp/system_eye_metrics.json"
        fi
        ;;

    "debug")
        log_header "DEBUG INFORMATION"
        COMPONENT="${2:-pipeline}"

        log_info "Collecting debug information for component: $COMPONENT"

        # Get pod info
        POD_NAME=$(get_pod_name $COMPONENT)
        if [ -n "$POD_NAME" ]; then
            echo -e "\n${CYAN}POD INFORMATION:${NC}"
            kubectl -n $NAMESPACE describe pod $POD_NAME

            echo -e "\n${CYAN}POD LOGS (last 50 lines):${NC}"
            kubectl -n $NAMESPACE logs $POD_NAME --tail=50

            echo -e "\n${CYAN}POD PROCESSES:${NC}"
            kubectl -n $NAMESPACE exec $POD_NAME -- ps aux 2>/dev/null || log_warn "Cannot access pod processes"

            echo -e "\n${CYAN}POD RESOURCES:${NC}"
            kubectl -n $NAMESPACE top pod $POD_NAME 2>/dev/null || log_warn "Cannot get resource usage"

        else
            log_error "No pod found for component: $COMPONENT"
        fi

        # Save debug info to file
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        DEBUG_FILE="$LOG_DIR/debug_${COMPONENT}_${TIMESTAMP}.log"
        {
            echo "=== DEBUG INFORMATION FOR $COMPONENT ==="
            echo "Timestamp: $(date)"
            echo "Namespace: $NAMESPACE"
            echo "Component: $COMPONENT"
            echo "Pod: $POD_NAME"
            echo
            kubectl -n $NAMESPACE describe pod $POD_NAME 2>&1
            echo
            echo "=== LOGS ==="
            kubectl -n $NAMESPACE logs $POD_NAME 2>&1
        } > "$DEBUG_FILE"

        log_success "Debug information saved to: $DEBUG_FILE"
        ;;

    "shell"|"exec")
        log_header "INTERACTIVE SHELL"
        COMPONENT="${2:-pipeline}"
        POD_NAME=$(get_pod_name $COMPONENT)

        if [ -z "$POD_NAME" ]; then
            log_error "No pod found for component: $COMPONENT"
            exit 1
        fi

        log_info "Opening shell in pod: $POD_NAME"
        kubectl -n $NAMESPACE exec -it $POD_NAME -- /bin/bash
        ;;

    "port-forward"|"forward")
        log_header "PORT FORWARDING"
        COMPONENT="${2:-demo}"
        LOCAL_PORT="${3:-8080}"
        REMOTE_PORT="${4:-8080}"

        POD_NAME=$(get_pod_name $COMPONENT)
        if [ -z "$POD_NAME" ]; then
            log_error "No pod found for component: $COMPONENT"
            exit 1
        fi

        log_info "Port forwarding $LOCAL_PORT:$REMOTE_PORT to pod: $POD_NAME"
        log_info "Access at: http://localhost:$LOCAL_PORT"
        kubectl -n $NAMESPACE port-forward $POD_NAME $LOCAL_PORT:$REMOTE_PORT
        ;;

    "backup")
        log_header "BACKUP DATA"
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        BACKUP_DIR="$LOG_DIR/backup_$TIMESTAMP"

        log_info "Creating backup in: $BACKUP_DIR"
        mkdir -p "$BACKUP_DIR"

        # Backup models and data from PVCs
        for component in models data; do
            PVC_NAME="teamovercooked-${component}-pvc"
            if kubectl -n $NAMESPACE get pvc $PVC_NAME &>/dev/null; then
                log_info "Backing up $component PVC..."

                # Create a temporary pod to access PVC
                BACKUP_POD="backup-$component-$(date +%s)"
                kubectl run $BACKUP_POD -n $NAMESPACE --image=busybox --restart=Never --rm -i --tty=false \
                    --overrides='{
                        "spec": {
                            "containers": [{
                                "name": "backup",
                                "image": "busybox",
                                "command": ["tar", "czf", "/backup/'$component'.tar.gz", "-C", "/'$component'", "."],
                                "volumeMounts": [{
                                    "name": "data",
                                    "mountPath": "/'$component'"
                                }, {
                                    "name": "backup",
                                    "mountPath": "/backup"
                                }]
                            }],
                            "volumes": [{
                                "name": "data",
                                "persistentVolumeClaim": {"claimName": "'$PVC_NAME'"}
                            }, {
                                "name": "backup",
                                "hostPath": {"path": "'$BACKUP_DIR'"}
                            }]
                        }
                    }' --wait &

                sleep 5
                kubectl -n $NAMESPACE wait --for=condition=Ready pod/$BACKUP_POD --timeout=60s
                kubectl -n $NAMESPACE delete pod $BACKUP_POD
            fi
        done

        # Backup configurations
        kubectl -n $NAMESPACE get configmap teamovercooked-ml-src -o yaml > "$BACKUP_DIR/configmap.yaml"
        kubectl -n $NAMESPACE get pvc -l app=teamovercooked -o yaml > "$BACKUP_DIR/pvcs.yaml"

        log_success "Backup completed: $BACKUP_DIR"
        ;;

    "health"|"healthcheck")
        log_header "HEALTH CHECK"

        log_info "Performing comprehensive health check..."

        # Check cluster connectivity
        if kubectl cluster-info &>/dev/null; then
            log_success "✓ Cluster connectivity"
        else
            log_error "✗ Cluster connectivity"
        fi

        # Check namespace
        if kubectl get namespace $NAMESPACE &>/dev/null; then
            log_success "✓ Namespace exists: $NAMESPACE"
        else
            log_error "✗ Namespace missing: $NAMESPACE"
        fi

        # Check ConfigMap
        if kubectl -n $NAMESPACE get configmap teamovercooked-ml-src &>/dev/null; then
            log_success "✓ ConfigMap exists"
        else
            log_warn "⚠ ConfigMap missing"
        fi

        # Check PVCs
        PVC_COUNT=$(kubectl -n $NAMESPACE get pvc -l app=teamovercooked --no-headers 2>/dev/null | wc -l)
        if [ "$PVC_COUNT" -gt 0 ]; then
            log_success "✓ PVCs found: $PVC_COUNT"
        else
            log_warn "⚠ No PVCs found"
        fi

        # Check running pods
        RUNNING_PODS=$(kubectl -n $NAMESPACE get pods -l app=teamovercooked --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
        if [ "$RUNNING_PODS" -gt 0 ]; then
            log_success "✓ Running pods: $RUNNING_PODS"
        else
            log_warn "⚠ No running pods"
        fi

        # Check recent jobs
        RECENT_JOBS=$(kubectl -n $NAMESPACE get jobs -l app=teamovercooked --no-headers 2>/dev/null | wc -l)
        log_info "Recent jobs: $RECENT_JOBS"

        log_success "Health check completed"
        ;;

    *)
        echo -e "${WHITE}SYSTEM'S BRAIN DEPLOYMENT TOOL v$VERSION${NC}"
        echo ""
        echo -e "${CYAN}USAGE:${NC} $0 <command> [options]"
        echo ""
        echo -e "${YELLOW}DEPLOYMENT COMMANDS:${NC}"
        echo "  prereq              - Check prerequisites"
        echo "  setup/configmap     - Create ConfigMap from Python sources"
        echo "  pvc/storage         - Create persistent storage volumes"
        echo "  pipeline/all [-f]   - Run complete ML pipeline (--follow for logs)"
        echo "  train/training      - Run training job with progressive training"
        echo "  test/testing        - Run comprehensive model testing"
        echo "  demo                - Deploy live demo (long-running)"
        echo "  overnight           - Deploy overnight training CronJob"
        echo ""
        echo -e "${YELLOW}MONITORING & DEBUGGING:${NC}"
        echo "  status              - Check deployment status"
        echo "  logs [comp] [-f]    - View logs (e.g., logs pipeline -f)"
        echo "  monitor/watch [int] - Real-time monitoring dashboard"
        echo "  telemetry [comp]    - Inspect telemetry data"
        echo "  debug [comp]        - Collect debug information"
        echo "  health/healthcheck  - Comprehensive health check"
        echo ""
        echo -e "${YELLOW}MAINTENANCE COMMANDS:${NC}"
        echo "  clean [type]        - Cleanup (basic|jobs|all)"
        echo "  restart [comp]      - Restart component"
        echo "  backup              - Backup data and configurations"
        echo "  shell/exec [comp]   - Interactive shell in pod"
        echo "  forward [comp] [lp] - Port forward (component, local-port)"
        echo ""
        echo -e "${GREEN}QUICK START:${NC}"
        echo "  1. $0 prereq        ${BLUE}# Check prerequisites${NC}"
        echo "  2. $0 setup         ${BLUE}# Create ConfigMap${NC}"
        echo "  3. $0 pvc           ${BLUE}# Create storage${NC}"
        echo "  4. $0 pipeline -f   ${BLUE}# Run pipeline with logs${NC}"
        echo ""
        echo -e "${GREEN}MONITORING WORKFLOW:${NC}"
        echo "  $0 monitor          ${BLUE}# Watch dashboard${NC}"
        echo "  $0 telemetry        ${BLUE}# Check telemetry${NC}"
        echo "  $0 logs pipeline -f ${BLUE}# Follow logs${NC}"
        echo ""
        echo -e "${GREEN}DEBUG WORKFLOW:${NC}"
        echo "  $0 health           ${BLUE}# Check system health${NC}"
        echo "  $0 debug pipeline   ${BLUE}# Collect debug info${NC}"
        echo "  $0 shell pipeline   ${BLUE}# Access pod shell${NC}"
        echo ""
        echo -e "${RED}CLEANUP:${NC}"
        echo "  $0 clean jobs       ${BLUE}# Clean only jobs${NC}"
        echo "  $0 clean all        ${BLUE}# Full cleanup (DANGER!)${NC}"
        echo ""
        echo -e "${PURPLE}Examples:${NC}"
        echo "  $0 pipeline --follow              # Run pipeline and follow logs"
        echo "  $0 monitor 3                      # Monitor with 3s refresh"
        echo "  $0 clean jobs                     # Clean up completed jobs"
        echo "  $0 logs pipeline -f               # Follow pipeline logs"
        echo "  $0 shell pipeline                 # Get shell in pipeline pod"
        echo "  $0 forward demo 8080 80           # Forward demo port"
        echo ""
        exit 1
        ;;
esac
