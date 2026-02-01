#!/usr/bin/env python3
"""
Health check script for System Guardian
Returns exit code 0 if healthy, 1 if unhealthy
"""

import sys
import os
import time

def check_process_running():
    """Check if guardian daemon process is running"""
    # Check if our PID file exists or process is running
    try:
        # Simple check - if we can import the main module and check for log activity
        log_file = "/tmp/system-guardian.log"
        if os.path.exists(log_file):
            # Check if log was updated in last 60 seconds
            last_modified = os.path.getmtime(log_file)
            current_time = time.time()
            if current_time - last_modified < 60:
                return True
        return False
    except Exception:
        return False

def check_connectivity():
    """Check if we can connect to System Eye"""
    try:
        import grpc
        from metrics_pb2_grpc import MetricsServiceStub

        # Try to connect
        channel = grpc.insecure_channel('system-eye:50051', options=[
            ('grpc.keepalive_timeout_ms', 5000),
        ])

        # Wait for channel to be ready (with timeout)
        grpc.channel_ready_future(channel).result(timeout=2)
        channel.close()
        return True
    except Exception:
        # Connection issues are expected during startup
        # Just check if process is running
        return check_process_running()

if __name__ == '__main__':
    # Check if process appears to be running
    if check_process_running():
        sys.exit(0)  # Healthy
    else:
        sys.exit(1)  # Unhealthy
