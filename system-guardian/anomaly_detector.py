"""
Anomaly Detector - Detects system anomalies using rule-based and ML-based methods.

Identifies CPU spikes, memory leaks, I/O bottlenecks, and other performance anomalies.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import statistics


@dataclass
class Anomaly:
    """Information about a detected anomaly"""
    detected_at: datetime
    anomaly_type: str  # 'cpu_spike', 'memory_leak', 'io_bottleneck', 'process_spike'
    severity: str  # 'critical', 'high', 'medium', 'low'
    affected_pids: List[int]
    process_names: Dict[int, str]
    metrics: Dict[str, float]
    description: str
    recommendation: str


class AnomalyDetector:
    """Detects system and process anomalies"""

    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Thresholds
        self.cpu_spike_threshold = self.config.get('cpu_spike_threshold', 80.0)  # %
        self.cpu_spike_duration = self.config.get('cpu_spike_duration_samples', 3)
        self.memory_growth_threshold = self.config.get('memory_growth_mb_per_min', 100)
        self.memory_leak_samples = self.config.get('memory_leak_samples', 5)
        self.io_wait_threshold = self.config.get('io_wait_threshold_ms', 1000)

        # History windows
        self.history_size = self.config.get('history_size', 100)
        self.cpu_history: deque = deque(maxlen=self.history_size)
        self.memory_history: deque = deque(maxlen=self.history_size)
        self.process_history: Dict[int, deque] = {}

        # Detected anomalies
        self.anomaly_history: List[Anomaly] = []

        self.logger.info(f"AnomalyDetector initialized (CPU threshold: {self.cpu_spike_threshold}%)")

    def detect_anomalies(self, metrics) -> List[Anomaly]:
        """
        Detect all types of anomalies in the current metrics

        Args:
            metrics: MetricsSnapshot with system metrics

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Update history
        self._update_history(metrics)

        # Run detection algorithms
        anomalies.extend(self._detect_cpu_spikes(metrics))
        anomalies.extend(self._detect_memory_leaks(metrics))
        anomalies.extend(self._detect_io_bottlenecks(metrics))
        anomalies.extend(self._detect_process_anomalies(metrics))

        # Update history
        self.anomaly_history.extend(anomalies)
        if len(self.anomaly_history) > 1000:
            self.anomaly_history = self.anomaly_history[-1000:]

        if anomalies:
            self.logger.warning(f"Detected {len(anomalies)} anomalies")

        return anomalies

    def _update_history(self, metrics):
        """Update metric history for trend analysis"""
        timestamp = datetime.now()

        # System-level history
        self.cpu_history.append({
            'timestamp': timestamp,
            'overall': metrics.cpu_overall,
            'per_core': list(metrics.raw_proto.cpu.per_core) if hasattr(metrics.raw_proto, 'cpu') else []
        })

        self.memory_history.append({
            'timestamp': timestamp,
            'used': metrics.memory_used,
            'percent': metrics.memory_percent
        })

        # Per-process history
        if hasattr(metrics.raw_proto, 'processes') and metrics.raw_proto.processes:
            for proc in metrics.raw_proto.processes.processes:
                pid = proc.pid
                if pid not in self.process_history:
                    self.process_history[pid] = deque(maxlen=20)

                self.process_history[pid].append({
                    'timestamp': timestamp,
                    'cpu': proc.cpu_percent,
                    'memory': proc.memory_rss,
                    'state': proc.state
                })

    def _detect_cpu_spikes(self, metrics) -> List[Anomaly]:
        """Detect sustained CPU spikes"""
        anomalies = []

        if len(self.cpu_history) < self.cpu_spike_duration:
            return anomalies

        # Check if CPU has been high for sustained period
        recent = list(self.cpu_history)[-self.cpu_spike_duration:]
        high_cpu_count = sum(1 for h in recent if h['overall'] > self.cpu_spike_threshold)

        if high_cpu_count == self.cpu_spike_duration:
            # Sustained spike detected
            avg_cpu = statistics.mean([h['overall'] for h in recent])

            # Find top CPU consumers
            top_processes = []
            if hasattr(metrics.raw_proto, 'processes') and metrics.raw_proto.processes:
                procs = list(metrics.raw_proto.processes.top_cpu)[:5]
                top_processes = [(p.pid, p.name, p.cpu_percent) for p in procs]

            anomaly = Anomaly(
                detected_at=datetime.now(),
                anomaly_type='cpu_spike',
                severity='critical' if avg_cpu > 95 else 'high',
                affected_pids=[p[0] for p in top_processes],
                process_names={p[0]: p[1] for p in top_processes},
                metrics={'avg_cpu': avg_cpu, 'duration_samples': self.cpu_spike_duration},
                description=f"Sustained CPU spike: {avg_cpu:.1f}% for {self.cpu_spike_duration} samples",
                recommendation="Consider restarting top CPU consumers or scaling resources"
            )
            anomalies.append(anomaly)

        return anomalies

    def _detect_memory_leaks(self, metrics) -> List[Anomaly]:
        """Detect memory leaks (continuous growth)"""
        anomalies = []

        # Need sufficient history
        if len(self.memory_history) < self.memory_leak_samples:
            return anomalies

        # Check system memory growth
        recent = list(self.memory_history)[-self.memory_leak_samples:]
        memory_values = [h['used'] for h in recent]

        # Check if memory is consistently growing
        is_growing = all(memory_values[i] < memory_values[i+1]
                        for i in range(len(memory_values)-1))

        if is_growing:
            growth_mb = (memory_values[-1] - memory_values[0]) / (1024 * 1024)
            time_span = (recent[-1]['timestamp'] - recent[0]['timestamp']).total_seconds() / 60

            if time_span > 0:
                growth_rate = growth_mb / time_span

                if growth_rate > self.memory_growth_threshold:
                    # Find processes with growing memory
                    leaking_procs = self._find_memory_growing_processes()

                    anomaly = Anomaly(
                        detected_at=datetime.now(),
                        anomaly_type='memory_leak',
                        severity='critical' if growth_rate > self.memory_growth_threshold * 2 else 'high',
                        affected_pids=[p[0] for p in leaking_procs],
                        process_names={p[0]: p[1] for p in leaking_procs},
                        metrics={'growth_rate_mb_per_min': growth_rate, 'total_growth_mb': growth_mb},
                        description=f"Memory leak detected: {growth_rate:.1f} MB/min growth",
                        recommendation="Investigate and restart leaking processes"
                    )
                    anomalies.append(anomaly)

        return anomalies

    def _find_memory_growing_processes(self) -> List[Tuple[int, str, float]]:
        """Find processes with continuously growing memory"""
        growing_procs = []

        for pid, history in self.process_history.items():
            if len(history) < 5:
                continue

            recent = list(history)[-5:]
            memory_values = [h['memory'] for h in recent]

            # Check consistent growth
            is_growing = all(memory_values[i] < memory_values[i+1]
                            for i in range(len(memory_values)-1))

            if is_growing:
                growth_mb = (memory_values[-1] - memory_values[0]) / (1024 * 1024)
                if growth_mb > 10:  # At least 10MB growth
                    # Get process name from current metrics
                    proc_name = f"PID-{pid}"
                    growing_procs.append((pid, proc_name, growth_mb))

        return sorted(growing_procs, key=lambda x: x[2], reverse=True)[:5]

    def _detect_io_bottlenecks(self, metrics) -> List[Anomaly]:
        """Detect I/O bottlenecks"""
        anomalies = []

        if not hasattr(metrics.raw_proto, 'processes') or not metrics.raw_proto.processes:
            return anomalies

        # Find processes in uninterruptible sleep (D state) for long time
        blocked_procs = []

        for proc in metrics.raw_proto.processes.processes:
            if proc.state == 'D':  # Uninterruptible sleep (usually I/O)
                if hasattr(proc, 'io_wait_time_ms') and proc.HasField('io_wait_time_ms'):
                    if proc.io_wait_time_ms > self.io_wait_threshold:
                        blocked_procs.append((proc.pid, proc.name, proc.io_wait_time_ms))

        if len(blocked_procs) >= 3:  # Multiple processes blocked
            anomaly = Anomaly(
                detected_at=datetime.now(),
                anomaly_type='io_bottleneck',
                severity='high',
                affected_pids=[p[0] for p in blocked_procs],
                process_names={p[0]: p[1] for p in blocked_procs},
                metrics={'blocked_count': len(blocked_procs)},
                description=f"I/O bottleneck: {len(blocked_procs)} processes in uninterruptible sleep",
                recommendation="Check disk I/O, consider I/O prioritization or caching"
            )
            anomalies.append(anomaly)

        return anomalies

    def _detect_process_anomalies(self, metrics) -> List[Anomaly]:
        """Detect process-specific anomalies"""
        anomalies = []

        if not hasattr(metrics.raw_proto, 'processes') or not metrics.raw_proto.processes:
            return anomalies

        # Detect processes with excessive threads
        for proc in metrics.raw_proto.processes.processes:
            if proc.num_threads > 1000:  # Excessive threads
                anomaly = Anomaly(
                    detected_at=datetime.now(),
                    anomaly_type='excessive_threads',
                    severity='medium',
                    affected_pids=[proc.pid],
                    process_names={proc.pid: proc.name},
                    metrics={'thread_count': proc.num_threads},
                    description=f"Process {proc.name} has {proc.num_threads} threads",
                    recommendation="Investigate thread leak or reduce thread pool size"
                )
                anomalies.append(anomaly)

        return anomalies

    def get_anomaly_summary(self) -> Dict:
        """Get summary of anomaly detection history"""
        recent = [a for a in self.anomaly_history
                 if (datetime.now() - a.detected_at).total_seconds() < 3600]

        by_type = {}
        for anomaly in self.anomaly_history:
            by_type[anomaly.anomaly_type] = by_type.get(anomaly.anomaly_type, 0) + 1

        return {
            'total_detected': len(self.anomaly_history),
            'last_hour': len(recent),
            'by_type': by_type,
            'by_severity': {
                'critical': sum(1 for a in self.anomaly_history if a.severity == 'critical'),
                'high': sum(1 for a in self.anomaly_history if a.severity == 'high'),
                'medium': sum(1 for a in self.anomaly_history if a.severity == 'medium'),
                'low': sum(1 for a in self.anomaly_history if a.severity == 'low')
            }
        }
