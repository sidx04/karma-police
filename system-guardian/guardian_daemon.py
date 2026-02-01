"""
System Guardian Daemon - Main orchestrator for self-healing system.

Continuously monitors system metrics via gRPC, detects deadlocks and anomalies,
and performs automatic healing actions.
"""

import logging
import signal
import sys
import time
import argparse
from typing import Optional, Dict
from datetime import datetime
import yaml

from data.collectors.euler_eye_grpc_adapter import SystemEyeGRPCAdapter
from deadlock_detector import DeadlockDetector
from anomaly_detector import AnomalyDetector
from self_healer import SelfHealer
from alert_manager import AlertManager


class GuardianDaemon:
    """Main daemon orchestrating detection and healing"""

    def __init__(self, config_path: Optional[str] = None):
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.running = False

        # Components
        self.grpc_adapter: Optional[SystemEyeGRPCAdapter] = None
        self.deadlock_detector: Optional[DeadlockDetector] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.self_healer: Optional[SelfHealer] = None
        self.alert_manager: Optional[AlertManager] = None
        self._grpc_connected = False

        # Statistics
        self.stats = {
            'start_time': datetime.now(),
            'metrics_received': 0,
            'deadlocks_detected': 0,
            'anomalies_detected': 0,
            'healing_actions': 0,
            'errors': 0
        }

        self.logger.info("System Guardian initialized")

    def _setup_logging(self) -> logging.Logger:
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/tmp/system-guardian.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'grpc': {
                'server': 'localhost:50051',
                'reconnect_interval': 5
            },
            'deadlock_detector': {
                'min_wait_time_ms': 5000,
                'critical_wait_time_ms': 30000
            },
            'anomaly_detector': {
                'cpu_spike_threshold': 80.0,
                'cpu_spike_duration_samples': 3,
                'memory_growth_mb_per_min': 100,
                'memory_leak_samples': 5,
                'io_wait_threshold_ms': 1000
            },
            'self_healer': {
                'enabled': True,
                'dry_run': False,
                'max_restarts_per_hour': 3,
                'cooldown_minutes': 5,
                'protected_processes': [
                    'systemd', 'init', 'sshd', 'system-eye', 'system-guardian'
                ]
            },
            'detection': {
                'check_interval_seconds': 10,
                'enable_deadlock_detection': True,
                'enable_anomaly_detection': True
            }
        }

        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Merge user config with defaults
                    self._merge_dicts(default_config, user_config)
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_path}: {e}")
                self.logger.info("Using default configuration")

        return default_config

    def _merge_dicts(self, base: Dict, override: Dict):
        """Recursively merge override dict into base"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_dicts(base[key], value)
            else:
                base[key] = value

    def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Initialize gRPC adapter
            self.grpc_adapter = SystemEyeGRPCAdapter(
                server_address=self.config['grpc']['server']
            )

            # Initialize detectors
            self.deadlock_detector = DeadlockDetector(
                config=self.config['deadlock_detector']
            )

            self.anomaly_detector = AnomalyDetector(
                config=self.config['anomaly_detector']
            )

            # Initialize healer
            self.self_healer = SelfHealer(
                config=self.config['self_healer']
            )

            # Initialize alert manager
            self.alert_manager = AlertManager(
                config=self.config.get('alerts', {})
            )

            self.logger.info("All components initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    def run(self):
        """Main daemon loop"""
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("System Guardian daemon started")
        self.logger.info(f"Configuration: {self.config}")

        while self.running:
            try:
                self._run_monitoring_cycle()
            except Exception as e:
                self.logger.error(f"Error in monitoring cycle: {e}")
                self.stats['errors'] += 1
                time.sleep(5)  # Brief pause before retry

    def _run_monitoring_cycle(self):
        """Run one complete monitoring and healing cycle"""
        try:
            # Connect to gRPC server if not connected
            if not self._grpc_connected:
                self.logger.info(f"Connecting to System Eye at {self.config['grpc']['server']}...")
                self._grpc_connected = self.grpc_adapter.connect()
                if not self._grpc_connected:
                    self.logger.error("Failed to connect to System Eye")
                    time.sleep(self.config['grpc']['reconnect_interval'])
                    return
                time.sleep(1)

            # Get current metrics snapshot
            metrics = self.grpc_adapter.get_metrics(include_processes=True)

            if metrics is None:
                self.logger.warning("No metrics available")
                self._grpc_connected = False  # Force reconnect on next cycle
                time.sleep(self.config['detection']['check_interval_seconds'])
                return

            self.stats['metrics_received'] += 1

            # Run detections
            deadlocks = []
            anomalies = []

            if self.config['detection']['enable_deadlock_detection']:
                deadlocks = self._detect_deadlocks(metrics)

            if self.config['detection']['enable_anomaly_detection']:
                anomalies = self._detect_anomalies(metrics)

            # Perform healing actions
            if deadlocks or anomalies:
                self._perform_healing(deadlocks, anomalies)

            # Log status periodically
            if self.stats['metrics_received'] % 10 == 0:
                self._log_status()

            # Sleep until next check
            time.sleep(self.config['detection']['check_interval_seconds'])

        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.logger.error(f"Monitoring cycle error: {e}", exc_info=True)
            self.stats['errors'] += 1

    def _detect_deadlocks(self, metrics) -> list:
        """Run deadlock detection"""
        try:
            # Extract process list from metrics
            processes = []
            if hasattr(metrics.raw_proto, 'processes') and metrics.raw_proto.processes:
                processes = list(metrics.raw_proto.processes.processes)

            if not processes:
                return []

            deadlocks = self.deadlock_detector.detect_deadlocks(processes)

            if deadlocks:
                self.stats['deadlocks_detected'] += len(deadlocks)
                for dl in deadlocks:
                    self.logger.warning(
                        f"DEADLOCK DETECTED: {len(dl.cycle_pids)} processes - "
                        f"PIDs {dl.cycle_pids} - Severity: {dl.severity}"
                    )
                    # Send alert
                    if self.alert_manager:
                        self.alert_manager.alert_deadlock_detected(dl)

            return deadlocks

        except Exception as e:
            self.logger.error(f"Deadlock detection error: {e}")
            return []

    def _detect_anomalies(self, metrics) -> list:
        """Run anomaly detection"""
        try:
            anomalies = self.anomaly_detector.detect_anomalies(metrics)

            if anomalies:
                self.stats['anomalies_detected'] += len(anomalies)
                for anomaly in anomalies:
                    self.logger.warning(
                        f"ANOMALY DETECTED: {anomaly.anomaly_type} - "
                        f"Severity: {anomaly.severity} - "
                        f"Affected PIDs: {anomaly.affected_pids} - "
                        f"{anomaly.description}"
                    )
                    # Send alert
                    if self.alert_manager:
                        self.alert_manager.alert_anomaly_detected(anomaly)

            return anomalies

        except Exception as e:
            self.logger.error(f"Anomaly detection error: {e}")
            return []

    def _perform_healing(self, deadlocks: list, anomalies: list):
        """Perform healing actions for detected issues"""
        try:
            healing_results = []

            # Heal deadlocks
            for deadlock in deadlocks:
                results = self.self_healer.heal_deadlock(deadlock)
                healing_results.extend(results)

                # Send healing alert
                if self.alert_manager:
                    self.alert_manager.alert_deadlock_healed(deadlock, results)

                for result in results:
                    if result.success:
                        self.logger.info(
                            f"HEALING SUCCESS: {result.action.value} - {result.message}"
                        )
                    else:
                        self.logger.warning(
                            f"HEALING FAILED: {result.action.value} - {result.message} - "
                            f"Error: {result.error}"
                        )

            # Heal anomalies
            for anomaly in anomalies:
                results = self.self_healer.heal_anomaly(anomaly)
                healing_results.extend(results)

                # Send healing alert
                if self.alert_manager:
                    self.alert_manager.alert_anomaly_healed(anomaly, results)

                for result in results:
                    if result.success:
                        self.logger.info(
                            f"HEALING SUCCESS: {result.action.value} - {result.message}"
                        )
                    else:
                        self.logger.warning(
                            f"HEALING FAILED: {result.action.value} - {result.message} - "
                            f"Error: {result.error}"
                        )

            self.stats['healing_actions'] += len(healing_results)

        except Exception as e:
            self.logger.error(f"Healing error: {e}", exc_info=True)

    def _log_status(self):
        """Log daemon status"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()

        self.logger.info(
            f"STATUS: Uptime={uptime:.0f}s | "
            f"Metrics={self.stats['metrics_received']} | "
            f"Deadlocks={self.stats['deadlocks_detected']} | "
            f"Anomalies={self.stats['anomalies_detected']} | "
            f"Healings={self.stats['healing_actions']} | "
            f"Errors={self.stats['errors']}"
        )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
        self.shutdown()

    def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down System Guardian...")

        # Log final statistics
        self._log_status()

        # Get summaries from components
        if self.deadlock_detector:
            dl_summary = self.deadlock_detector.get_deadlock_summary()
            self.logger.info(f"Deadlock summary: {dl_summary}")

        if self.anomaly_detector:
            an_summary = self.anomaly_detector.get_anomaly_summary()
            self.logger.info(f"Anomaly summary: {an_summary}")

        if self.self_healer:
            heal_summary = self.self_healer.get_healing_summary()
            self.logger.info(f"Healing summary: {heal_summary}")

        if self.alert_manager:
            alert_summary = self.alert_manager.get_alert_summary()
            self.logger.info(f"Alert summary: {alert_summary}")

        # Disconnect gRPC
        if self.grpc_adapter and self._grpc_connected:
            self.grpc_adapter.close()

        self.logger.info("System Guardian shutdown complete")
        sys.exit(0)


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(description='System Guardian Self-Healing Daemon')
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (YAML)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode (no actual healing actions)'
    )

    args = parser.parse_args()

    # Create daemon
    daemon = GuardianDaemon(config_path=args.config)

    # Override dry-run if specified
    if args.dry_run:
        daemon.config['self_healer']['dry_run'] = True
        print("Running in DRY-RUN mode - no actual healing actions will be performed")

    # Initialize
    if not daemon.initialize():
        print("Failed to initialize daemon", file=sys.stderr)
        sys.exit(1)

    # Run
    try:
        daemon.run()
    except KeyboardInterrupt:
        daemon.shutdown()
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
