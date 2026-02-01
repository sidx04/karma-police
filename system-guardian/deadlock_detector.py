"""
Deadlock Detector - Detects deadlocks using cycle detection in process dependency graphs.

Analyzes process wait channels, file locks, and blocking relationships to identify
circular dependencies that indicate deadlocks.
"""

import logging
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import networkx as nx


@dataclass
class DeadlockInfo:
    """Information about a detected deadlock"""
    detected_at: datetime
    cycle_pids: List[int]
    process_names: Dict[int, str]
    wait_channels: Dict[int, str]
    file_locks: Dict[int, List[str]]
    severity: str  # 'critical', 'high', 'medium'
    duration_seconds: float = 0.0


class DeadlockDetector:
    """Detects deadlocks by analyzing process dependencies"""

    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Detection thresholds
        self.min_wait_time_ms = self.config.get('min_wait_time_ms', 5000)  # 5 seconds
        self.critical_wait_time_ms = self.config.get('critical_wait_time_ms', 30000)  # 30 seconds

        # State tracking
        self.previous_deadlocks: Dict[str, DeadlockInfo] = {}
        self.deadlock_history: List[DeadlockInfo] = []

        self.logger.info(f"DeadlockDetector initialized (min_wait={self.min_wait_time_ms}ms)")

    def detect_deadlocks(self, processes: List) -> List[DeadlockInfo]:
        """
        Detect deadlocks by analyzing process dependencies

        Args:
            processes: List of ProcessInfo protobuf messages

        Returns:
            List of detected deadlocks
        """
        deadlocks = []

        # Build process dependency graph
        graph = self._build_dependency_graph(processes)

        if graph is None or len(graph.nodes()) == 0:
            return deadlocks

        # Find all cycles in the graph
        try:
            cycles = list(nx.simple_cycles(graph))
        except Exception as e:
            self.logger.error(f"Error detecting cycles: {e}")
            return deadlocks

        # Analyze each cycle to determine if it's a deadlock
        for cycle in cycles:
            deadlock = self._analyze_cycle(cycle, processes, graph)
            if deadlock:
                deadlocks.append(deadlock)
                self.logger.warning(
                    f"Deadlock detected: {len(cycle)} processes involved - "
                    f"PIDs: {cycle}"
                )

        # Update history
        self.deadlock_history.extend(deadlocks)
        if len(self.deadlock_history) > 100:
            self.deadlock_history = self.deadlock_history[-100:]

        return deadlocks

    def _build_dependency_graph(self, processes: List) -> Optional[nx.DiGraph]:
        """
        Build directed graph of process dependencies

        Edges represent: Process A depends on (is blocked by) Process B
        """
        graph = nx.DiGraph()
        process_map = {p.pid: p for p in processes}

        for proc in processes:
            # Only consider processes that are waiting
            if not self._is_waiting(proc):
                continue

            # Skip if wait time is too short (transient wait)
            if hasattr(proc, 'wait_time_ms') and proc.HasField('wait_time_ms'):
                if proc.wait_time_ms < self.min_wait_time_ms:
                    continue

            graph.add_node(proc.pid)

            # Add edges based on blocking relationships

            # 1. Explicit blocking PID
            if hasattr(proc, 'waiting_on_pid') and proc.HasField('waiting_on_pid'):
                blocking_pid = proc.waiting_on_pid
                if blocking_pid in process_map:
                    graph.add_edge(proc.pid, blocking_pid, reason='explicit_block')

            # 2. File lock contention
            if hasattr(proc, 'waiting_locks') and len(proc.waiting_locks) > 0:
                for waiting_lock in proc.waiting_locks:
                    # Find who holds this lock
                    holder_pid = waiting_lock.pid if hasattr(waiting_lock, 'pid') else None
                    if holder_pid and holder_pid != proc.pid and holder_pid in process_map:
                        graph.add_edge(proc.pid, holder_pid, reason='file_lock',
                                     path=waiting_lock.path)

            # 3. Infer from wait channels (heuristic)
            if hasattr(proc, 'wait_channel') and proc.HasField('wait_channel'):
                wait_chan = proc.wait_channel
                # If multiple processes wait on same thing, they might be interdependent
                if wait_chan in ['futex', 'pipe_read', 'pipe_write', 'unix_stream_recvmsg']:
                    for other in processes:
                        if other.pid == proc.pid:
                            continue
                        if hasattr(other, 'wait_channel') and other.HasField('wait_channel'):
                            if other.wait_channel == wait_chan:
                                # Bidirectional dependency possibility
                                graph.add_edge(proc.pid, other.pid, reason='shared_wait')

        return graph if len(graph.nodes()) > 0 else None

    def _is_waiting(self, proc) -> bool:
        """Check if process is in a waiting state"""
        if not hasattr(proc, 'state'):
            return False

        # D = uninterruptible sleep (I/O), S = interruptible sleep
        waiting_states = ['D', 'S']
        return proc.state in waiting_states

    def _analyze_cycle(self, cycle: List[int], processes: List,
                      graph: nx.DiGraph) -> Optional[DeadlockInfo]:
        """
        Analyze a cycle to determine if it's a real deadlock

        Returns DeadlockInfo if it's a deadlock, None otherwise
        """
        if len(cycle) < 2:
            return None

        process_map = {p.pid: p for p in processes}

        # Gather information about processes in cycle
        cycle_info = {
            'pids': cycle,
            'names': {},
            'wait_channels': {},
            'wait_times': {},
            'file_locks': {},
            'states': {}
        }

        total_wait_time = 0
        uninterruptible_count = 0

        for pid in cycle:
            if pid not in process_map:
                continue

            proc = process_map[pid]
            cycle_info['names'][pid] = proc.name if hasattr(proc, 'name') else f"PID-{pid}"

            if hasattr(proc, 'state'):
                cycle_info['states'][pid] = proc.state
                if proc.state == 'D':  # Uninterruptible sleep
                    uninterruptible_count += 1

            if hasattr(proc, 'wait_channel') and proc.HasField('wait_channel'):
                cycle_info['wait_channels'][pid] = proc.wait_channel

            if hasattr(proc, 'wait_time_ms') and proc.HasField('wait_time_ms'):
                wait_time = proc.wait_time_ms
                cycle_info['wait_times'][pid] = wait_time
                total_wait_time += wait_time

            # Collect file locks
            locks = []
            if hasattr(proc, 'held_locks'):
                locks.extend([l.path for l in proc.held_locks])
            if hasattr(proc, 'waiting_locks'):
                locks.extend([l.path for l in proc.waiting_locks])
            if locks:
                cycle_info['file_locks'][pid] = locks

        # Determine severity
        avg_wait_time = total_wait_time / len(cycle) if len(cycle) > 0 else 0

        if avg_wait_time > self.critical_wait_time_ms or uninterruptible_count >= 2:
            severity = 'critical'
        elif avg_wait_time > self.min_wait_time_ms * 2:
            severity = 'high'
        else:
            severity = 'medium'

        # Create deadlock info
        deadlock = DeadlockInfo(
            detected_at=datetime.now(),
            cycle_pids=cycle,
            process_names=cycle_info['names'],
            wait_channels=cycle_info['wait_channels'],
            file_locks=cycle_info['file_locks'],
            severity=severity,
            duration_seconds=avg_wait_time / 1000.0
        )

        return deadlock

    def get_deadlock_summary(self) -> Dict:
        """Get summary of deadlock detection history"""
        return {
            'total_detected': len(self.deadlock_history),
            'current_active': len(self.previous_deadlocks),
            'by_severity': {
                'critical': sum(1 for d in self.deadlock_history if d.severity == 'critical'),
                'high': sum(1 for d in self.deadlock_history if d.severity == 'high'),
                'medium': sum(1 for d in self.deadlock_history if d.severity == 'medium')
            }
        }
