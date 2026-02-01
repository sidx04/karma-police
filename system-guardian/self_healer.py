"""
Self-Healer - Performs automatic mitigation actions for detected issues.

Handles process restarts, resource isolation, and other healing actions with safety checks.
"""

import logging
import os
import signal
import subprocess
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class HealingAction(Enum):
    """Types of healing actions"""
    RESTART_PROCESS = "restart_process"
    KILL_PROCESS = "kill_process"
    NICE_PROCESS = "nice_process"  # Reduce priority
    LIMIT_CPU = "limit_cpu"
    LIMIT_MEMORY = "limit_memory"
    LOG_ONLY = "log_only"


@dataclass
class HealingResult:
    """Result of a healing action"""
    action: HealingAction
    success: bool
    pids: List[int]
    timestamp: datetime
    message: str
    error: Optional[str] = None


class SelfHealer:
    """Performs automatic healing actions"""

    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Safety settings
        self.enabled = self.config.get('enabled', True)
        self.dry_run = self.config.get('dry_run', False)
        self.max_restarts_per_hour = self.config.get('max_restarts_per_hour', 3)
        self.cooldown_minutes = self.config.get('cooldown_minutes', 5)

        # Protected processes (never kill/restart)
        self.protected_names = set(self.config.get('protected_processes', [
            'systemd', 'init', 'sshd', 'system-eye', 'system-guardian'
        ]))

        # Action history for rate limiting
        self.action_history: Dict[int, List[datetime]] = {}
        self.healing_log: List[HealingResult] = []

        self.logger.info(
            f"SelfHealer initialized (enabled={self.enabled}, dry_run={self.dry_run})"
        )

    def heal_deadlock(self, deadlock_info) -> List[HealingResult]:
        """
        Heal a detected deadlock

        Strategy: Kill/restart the least important process in the cycle
        """
        if not self.enabled:
            return self._log_only_action("Deadlock healing disabled", deadlock_info.cycle_pids)

        results = []

        # Select victim process (lowest priority, newest, or highest PID)
        victim_pid = self._select_deadlock_victim(deadlock_info)

        if victim_pid is None:
            return [HealingResult(
                action=HealingAction.LOG_ONLY,
                success=False,
                pids=deadlock_info.cycle_pids,
                timestamp=datetime.now(),
                message="Could not select victim for deadlock",
                error="All processes protected"
            )]

        # Check if we can act on this process
        if not self._can_act_on_process(victim_pid):
            return self._log_only_action(
                f"Rate limit exceeded for PID {victim_pid}",
                [victim_pid]
            )

        # Attempt to break deadlock by killing victim
        result = self._kill_process(victim_pid, f"Deadlock victim from cycle {deadlock_info.cycle_pids}")
        results.append(result)

        self.logger.warning(
            f"Deadlock healing: killed PID {victim_pid} to break cycle {deadlock_info.cycle_pids}"
        )

        return results

    def heal_anomaly(self, anomaly) -> List[HealingResult]:
        """
        Heal a detected anomaly

        Strategy depends on anomaly type
        """
        if not self.enabled:
            return self._log_only_action(f"Healing disabled for {anomaly.anomaly_type}", anomaly.affected_pids)

        results = []

        if anomaly.anomaly_type == 'cpu_spike':
            results.extend(self._heal_cpu_spike(anomaly))
        elif anomaly.anomaly_type == 'memory_leak':
            results.extend(self._heal_memory_leak(anomaly))
        elif anomaly.anomaly_type == 'io_bottleneck':
            results.extend(self._heal_io_bottleneck(anomaly))
        elif anomaly.anomaly_type == 'excessive_threads':
            results.extend(self._heal_excessive_threads(anomaly))
        else:
            results.append(self._log_only_action(f"No healing strategy for {anomaly.anomaly_type}", anomaly.affected_pids)[0])

        return results

    def _heal_cpu_spike(self, anomaly) -> List[HealingResult]:
        """Heal CPU spike by reducing priority of top consumers"""
        results = []

        for pid in anomaly.affected_pids[:3]:  # Top 3 consumers
            if not self._can_act_on_process(pid):
                continue

            # Try to renice first (less disruptive)
            result = self._nice_process(pid, 10)  # Increase nice by 10
            results.append(result)

            if result.success:
                self.logger.info(f"Reduced priority of PID {pid} (CPU spike mitigation)")

        return results

    def _heal_memory_leak(self, anomaly) -> List[HealingResult]:
        """Heal memory leak by restarting leaking processes"""
        results = []

        for pid in anomaly.affected_pids[:2]:  # Top 2 leakers
            if not self._can_act_on_process(pid):
                continue

            # Restart process
            result = self._restart_process(pid, "Memory leak detected")
            results.append(result)

        return results

    def _heal_io_bottleneck(self, anomaly) -> List[HealingResult]:
        """Heal I/O bottleneck (mainly logging for now)"""
        # I/O issues often system-wide, be conservative
        return self._log_only_action("I/O bottleneck detected", anomaly.affected_pids)

    def _heal_excessive_threads(self, anomaly) -> List[HealingResult]:
        """Heal excessive threads by restarting process"""
        results = []

        for pid in anomaly.affected_pids:
            if not self._can_act_on_process(pid):
                continue

            result = self._restart_process(pid, "Excessive threads")
            results.append(result)

        return results

    def _select_deadlock_victim(self, deadlock_info) -> Optional[int]:
        """Select which process to kill to break deadlock"""
        candidates = []

        for pid in deadlock_info.cycle_pids:
            # Skip protected processes
            if pid in deadlock_info.process_names:
                name = deadlock_info.process_names[pid]
                if any(protected in name.lower() for protected in self.protected_names):
                    continue

            try:
                proc = psutil.Process(pid)
                # Prefer processes with lower priority, newer, or higher PID
                priority = proc.nice()
                create_time = proc.create_time()
                candidates.append((pid, priority, create_time))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not candidates:
            return None

        # Select lowest priority, then newest
        candidates.sort(key=lambda x: (x[1], -x[2], -x[0]))
        return candidates[0][0]

    def _can_act_on_process(self, pid: int) -> bool:
        """Check if we can perform action on this process (rate limiting)"""
        now = datetime.now()

        # Check action history
        if pid in self.action_history:
            recent_actions = [t for t in self.action_history[pid]
                            if (now - t).total_seconds() < 3600]

            if len(recent_actions) >= self.max_restarts_per_hour:
                self.logger.warning(f"Rate limit exceeded for PID {pid}")
                return False

            # Check cooldown
            if recent_actions:
                last_action = max(recent_actions)
                if (now - last_action).total_seconds() < self.cooldown_minutes * 60:
                    self.logger.debug(f"Cooldown active for PID {pid}")
                    return False

        # Check if process is protected
        try:
            proc = psutil.Process(pid)
            if any(protected in proc.name().lower() for protected in self.protected_names):
                self.logger.warning(f"PID {pid} ({proc.name()}) is protected")
                return False
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

        return True

    def _kill_process(self, pid: int, reason: str) -> HealingResult:
        """Kill a process"""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would kill PID {pid}: {reason}")
            return HealingResult(
                action=HealingAction.KILL_PROCESS,
                success=True,
                pids=[pid],
                timestamp=datetime.now(),
                message=f"[DRY RUN] Kill PID {pid}: {reason}"
            )

        try:
            os.kill(pid, signal.SIGTERM)
            self._record_action(pid)

            self.logger.warning(f"Killed PID {pid}: {reason}")
            return HealingResult(
                action=HealingAction.KILL_PROCESS,
                success=True,
                pids=[pid],
                timestamp=datetime.now(),
                message=f"Killed PID {pid}: {reason}"
            )
        except ProcessLookupError:
            return HealingResult(
                action=HealingAction.KILL_PROCESS,
                success=False,
                pids=[pid],
                timestamp=datetime.now(),
                message=f"PID {pid} not found",
                error="Process already terminated"
            )
        except PermissionError as e:
            return HealingResult(
                action=HealingAction.KILL_PROCESS,
                success=False,
                pids=[pid],
                timestamp=datetime.now(),
                message=f"Permission denied to kill PID {pid}",
                error=str(e)
            )

    def _restart_process(self, pid: int, reason: str) -> HealingResult:
        """Restart a process (kill and let systemd/supervisor restart it)"""
        # For now, just kill it and rely on process supervisor
        return self._kill_process(pid, f"Restart: {reason}")

    def _nice_process(self, pid: int, nice_increment: int) -> HealingResult:
        """Change process priority (renice)"""
        if self.dry_run:
            return HealingResult(
                action=HealingAction.NICE_PROCESS,
                success=True,
                pids=[pid],
                timestamp=datetime.now(),
                message=f"[DRY RUN] Would renice PID {pid} by +{nice_increment}"
            )

        try:
            proc = psutil.Process(pid)
            current_nice = proc.nice()
            new_nice = min(current_nice + nice_increment, 19)  # Max nice is 19
            proc.nice(new_nice)

            self._record_action(pid)

            return HealingResult(
                action=HealingAction.NICE_PROCESS,
                success=True,
                pids=[pid],
                timestamp=datetime.now(),
                message=f"Reniced PID {pid} from {current_nice} to {new_nice}"
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            return HealingResult(
                action=HealingAction.NICE_PROCESS,
                success=False,
                pids=[pid],
                timestamp=datetime.now(),
                message=f"Failed to renice PID {pid}",
                error=str(e)
            )

    def _log_only_action(self, message: str, pids: List[int]) -> List[HealingResult]:
        """Log an issue without taking action"""
        result = HealingResult(
            action=HealingAction.LOG_ONLY,
            success=True,
            pids=pids,
            timestamp=datetime.now(),
            message=message
        )
        self.healing_log.append(result)
        return [result]

    def _record_action(self, pid: int):
        """Record an action for rate limiting"""
        if pid not in self.action_history:
            self.action_history[pid] = []
        self.action_history[pid].append(datetime.now())

        # Clean old history
        cutoff = datetime.now() - timedelta(hours=1)
        self.action_history[pid] = [t for t in self.action_history[pid] if t > cutoff]

    def get_healing_summary(self) -> Dict:
        """Get summary of healing actions"""
        recent = [h for h in self.healing_log
                 if (datetime.now() - h.timestamp).total_seconds() < 3600]

        return {
            'total_actions': len(self.healing_log),
            'last_hour': len(recent),
            'successful': sum(1 for h in self.healing_log if h.success),
            'failed': sum(1 for h in self.healing_log if not h.success),
            'by_action': {
                action.value: sum(1 for h in self.healing_log if h.action == action)
                for action in HealingAction
            }
        }
