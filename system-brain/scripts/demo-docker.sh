#!/bin/bash

CONTAINER_NAME="system-brain-demo"
IMAGE="system-brain:latest"
HOST_PORT=5000
CONTAINER_PORT=5000

MODELS_DIR="$(pwd)/models"
OUTPUT_DIR="$(pwd)"

echo "[STARTING] SYSTEM'S BRAIN LIVE DEMO (Docker)"
echo "============================================"

case "$1" in
    "start"|"run")
        echo "[CHECK] Ensuring trained model exists..."
        if [ ! -d "$MODELS_DIR" ] || [ -z "$(ls -A "$MODELS_DIR")" ]; then
            echo "[MODEL] No trained model found. Training first..."
            ./deploy-docker.sh train
        fi

        echo "[CLEAN] Removing old demo container (if any)..."
        docker rm -f $CONTAINER_NAME 2>/dev/null || true

        echo "[STARTING] Launching live demo container..."
        docker run -d \
            --name $CONTAINER_NAME \
            --gpus all \
            -p ${HOST_PORT}:${CONTAINER_PORT} \
            -v "$MODELS_DIR:/models:ro" \
            $IMAGE

        echo "[WAIT] Waiting for container to start..."
        sleep 3

        if docker ps | grep -q $CONTAINER_NAME; then
            echo "[OK] Live demo is running!"
            echo ""
            echo "[WEB] Open in browser:"
            echo "   http://localhost:${HOST_PORT}"
            echo ""
            echo "[WATCH] Logs:"
            echo "   $0 logs"
            echo ""
            echo "[SHELL] Exec:"
            echo "   $0 exec"
        else
            echo "[ERROR] Demo failed to start"
            exit 1
        fi
        ;;

    "logs"|"watch")
        echo "[WATCH] Watching live demo logs..."
        docker logs -f $CONTAINER_NAME
        ;;

    "exec"|"shell")
        echo "[SHELL] Opening shell in demo container..."
        docker exec -it $CONTAINER_NAME bash
        ;;

    "status")
        echo "[STATUS] Demo container:"
        docker ps -f "name=$CONTAINER_NAME"

        echo ""
        echo "[PORTS]"
        docker port $CONTAINER_NAME 2>/dev/null || echo "Container not running"
        ;;

    "stop"|"clean")
        echo "[STOP] Stopping live demo..."
        docker rm -f $CONTAINER_NAME 2>/dev/null || echo "Demo not running"
        echo "[OK] Demo stopped"
        ;;

    "scenarios")
        echo "[ACTION] Running real workload test scenarios..."
        docker exec $CONTAINER_NAME python3 - <<'EOF'
from live_classifier import run_quick_real_tests
run_quick_real_tests()
EOF
        ;;

    "dashboard")
        echo "[DATA] Generating live dashboard..."
        docker exec $CONTAINER_NAME python3 - <<'EOF'
from web_dashboard import generate_web_dashboard
html = generate_web_dashboard()
print(html)
EOF
        > system_brain_dashboard.html

        echo "[OK] Dashboard saved to: system_brain_dashboard.html"
        echo "[WEB] Open it in a browser to view"
        ;;

    *)
        echo "Usage: $0 {start|logs|status|stop|scenarios|dashboard}"
        echo ""
        echo "Commands:"
        echo "  start      - Run live demo container"
        echo "  logs       - Watch demo logs"
        echo "  status     - Check container status"
        echo "  stop       - Stop and remove demo container"
        echo "  scenarios  - Run quick test scenarios"
        echo "  dashboard  - Generate HTML dashboard"
        echo ""
        echo "Live Demo Features:"
        echo "  [LOOP] Real-time telemetry"
        echo "  [MODEL] Live ML classification"
        echo "  [DATA] Confidence scoring"
        echo "  [STATS] Performance monitoring"
        exit 1
        ;;
esac
