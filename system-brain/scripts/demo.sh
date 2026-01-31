#!/bin/bash

NAMESPACE="eidf219ns"

echo "[STARTING] SYSTEM'S BRAIN LIVE DEMO LAUNCHER"
echo "====================================="

case "$1" in
    "start"|"run")
        echo "[BUILD] Setting up live demo environment..."

        # Ensure we have a trained model
        echo "[LIST] Checking for trained model..."
        ./deploy.sh status | grep -q "Complete" || {
            echo "[MODEL] No trained model found. Training model first..."
            ./deploy.sh 2
            sleep 90
        }

        echo "[STARTING] Deploying live demo..."
        kubectl apply -f live-demo.yaml

        echo "[WAIT] Waiting for demo pod to be ready..."
        kubectl -n $NAMESPACE wait --for=condition=ready pod -l app=system-brain-demo --timeout=120s

        echo "[OK] Live demo is starting!"
        echo ""
        echo "[WATCH] To watch the live demo:"
        echo "   kubectl -n $NAMESPACE logs -f deployment/system-brain-live-demo"
        echo ""
        echo "[BUILD] To interact with the demo:"
        echo "   kubectl -n $NAMESPACE exec -it deployment/system-brain-live-demo -- bash"
        echo ""
        echo "[STOP] To stop the demo:"
        echo "   $0 stop"
        ;;

    "logs"|"watch")
        echo "[WATCH] Watching live demo output..."
        kubectl -n $NAMESPACE logs -f deployment/system-brain-live-demo
        ;;

    "exec"|"shell")
        echo "[BUILD] Opening shell in demo container..."
        kubectl -n $NAMESPACE exec -it deployment/system-brain-live-demo -- bash
        ;;

    "status")
        echo "[DATA] Demo Status:"
        kubectl -n $NAMESPACE get deployment system-brain-live-demo
        echo ""
        echo "[RUN] Running Pods:"
        kubectl -n $NAMESPACE get pods -l app=system-brain-demo
        ;;

    "stop"|"clean")
        echo "[STOP] Stopping live demo..."
        kubectl -n $NAMESPACE delete deployment system-brain-live-demo 2>/dev/null || echo "Demo not running"
        kubectl -n $NAMESPACE delete service system-brain-demo-service 2>/dev/null || echo "Service not found"
        kubectl delete configmap live-demo-code -n $NAMESPACE 2>/dev/null || echo "ConfigMap not found"
        echo "[OK] Demo stopped"
        ;;

    "scenarios")
        echo "[ACTION] Running real workload test scenarios..."
        kubectl -n $NAMESPACE exec deployment/system-brain-live-demo -- python3 -c "
exec(open('/code/live_classifier.py').read())
run_quick_real_tests()
"
        ;;

    "dashboard")
        echo "[DATA] Generating live dashboard view..."
        kubectl -n $NAMESPACE exec deployment/system-brain-live-demo -- python3 -c "
exec(open('/code/web_dashboard.py').read())
generate_web_dashboard()
        " > system_brain_dashboard.html
        echo "[OK] Dashboard saved to: system_brain_dashboard.html"
        echo "[WEB] Open in browser to view the dashboard template"
        ;;

    *)
        echo "Usage: $0 {start|logs|status|stop|scenarios|dashboard}"
        echo ""
        echo "Commands:"
        echo "  start      - Deploy and start the live demo"
        echo "  logs       - Watch live demo output"
        echo "  status     - Check demo deployment status"
        echo "  stop       - Stop and clean up demo"
        echo "  scenarios  - Run quick test scenarios"
        echo "  dashboard  - Generate web dashboard"
        echo ""
        echo "Live Demo Features:"
        echo "  [LOOP] Real-time telemetry simulation"
        echo "  [MODEL] Live ML workload classification"
        echo "  [DATA] Confidence scoring"
        echo "  [TARGET] Multiple workload scenarios"
        echo "  [STATS] Performance monitoring"
        exit 1
        ;;
esac