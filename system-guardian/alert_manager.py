"""
Alert Manager - Handles notifications and alerts for detected issues and healing actions.

Supports multiple alert channels: syslog, webhook, email, and custom integrations.
"""

import logging
import json
import syslog
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import requests


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertType(Enum):
    """Types of alerts"""
    DEADLOCK_DETECTED = "deadlock_detected"
    DEADLOCK_HEALED = "deadlock_healed"
    ANOMALY_DETECTED = "anomaly_detected"
    ANOMALY_HEALED = "anomaly_healed"
    HEALING_FAILED = "healing_failed"
    SYSTEM_ERROR = "system_error"


@dataclass
class Alert:
    """Alert information"""
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    metadata: Dict
    affected_pids: List[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['alert_type'] = self.alert_type.value
        data['severity'] = self.severity.value
        return data


class AlertManager:
    """Manages alerts and notifications"""

    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        self.enabled = self.config.get('enabled', True)

        # Channel configurations
        self.syslog_config = self.config.get('channels', {}).get('syslog', {})
        self.webhook_config = self.config.get('channels', {}).get('webhook', {})
        self.email_config = self.config.get('channels', {}).get('email', {})

        # Alert history
        self.alert_history: List[Alert] = []
        self.max_history = 1000

        # Initialize syslog if enabled
        if self.syslog_config.get('enabled', False):
            self._init_syslog()

        self.logger.info(f"AlertManager initialized (enabled={self.enabled})")

    def _init_syslog(self):
        """Initialize syslog connection"""
        try:
            facility = self.syslog_config.get('facility', 'daemon')
            facility_map = {
                'daemon': syslog.LOG_DAEMON,
                'user': syslog.LOG_USER,
                'local0': syslog.LOG_LOCAL0,
                'local1': syslog.LOG_LOCAL1,
            }
            syslog.openlog('system-guardian', syslog.LOG_PID, facility_map.get(facility, syslog.LOG_DAEMON))
            self.logger.info("Syslog integration enabled")
        except Exception as e:
            self.logger.warning(f"Failed to initialize syslog: {e}")

    def send_alert(self, alert: Alert):
        """Send an alert through configured channels"""
        if not self.enabled:
            return

        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

        # Log the alert
        self._log_alert(alert)

        # Send to configured channels
        if self.syslog_config.get('enabled', False):
            self._send_syslog(alert)

        if self.webhook_config.get('enabled', False):
            self._send_webhook(alert)

        if self.email_config.get('enabled', False):
            self._send_email(alert)

    def alert_deadlock_detected(self, deadlock_info):
        """Create alert for detected deadlock"""
        alert = Alert(
            timestamp=datetime.now(),
            alert_type=AlertType.DEADLOCK_DETECTED,
            severity=self._map_severity(deadlock_info.severity),
            title=f"Deadlock Detected: {len(deadlock_info.cycle_pids)} processes",
            message=f"Deadlock cycle detected involving PIDs: {deadlock_info.cycle_pids}. "
                   f"Process names: {list(deadlock_info.process_names.values())}. "
                   f"Average wait time: {deadlock_info.duration_seconds:.1f}s",
            metadata={
                'cycle_pids': deadlock_info.cycle_pids,
                'process_names': deadlock_info.process_names,
                'wait_channels': deadlock_info.wait_channels,
                'severity': deadlock_info.severity,
                'duration_seconds': deadlock_info.duration_seconds
            },
            affected_pids=deadlock_info.cycle_pids
        )
        self.send_alert(alert)

    def alert_deadlock_healed(self, deadlock_info, healing_results):
        """Create alert for healed deadlock"""
        successful = sum(1 for r in healing_results if r.success)

        alert = Alert(
            timestamp=datetime.now(),
            alert_type=AlertType.DEADLOCK_HEALED,
            severity=AlertSeverity.INFO,
            title=f"Deadlock Healed: {successful}/{len(healing_results)} actions successful",
            message=f"Deadlock involving PIDs {deadlock_info.cycle_pids} has been resolved. "
                   f"Actions taken: {[r.action.value for r in healing_results]}",
            metadata={
                'original_pids': deadlock_info.cycle_pids,
                'healing_results': [
                    {
                        'action': r.action.value,
                        'success': r.success,
                        'pids': r.pids,
                        'message': r.message
                    } for r in healing_results
                ]
            },
            affected_pids=deadlock_info.cycle_pids
        )
        self.send_alert(alert)

    def alert_anomaly_detected(self, anomaly):
        """Create alert for detected anomaly"""
        alert = Alert(
            timestamp=datetime.now(),
            alert_type=AlertType.ANOMALY_DETECTED,
            severity=self._map_severity(anomaly.severity),
            title=f"{anomaly.anomaly_type.upper()}: {anomaly.description}",
            message=f"Anomaly detected: {anomaly.description}. "
                   f"Affected processes: {anomaly.affected_pids}. "
                   f"Recommendation: {anomaly.recommendation}",
            metadata={
                'anomaly_type': anomaly.anomaly_type,
                'severity': anomaly.severity,
                'affected_pids': anomaly.affected_pids,
                'process_names': anomaly.process_names,
                'metrics': anomaly.metrics,
                'recommendation': anomaly.recommendation
            },
            affected_pids=anomaly.affected_pids
        )
        self.send_alert(alert)

    def alert_anomaly_healed(self, anomaly, healing_results):
        """Create alert for healed anomaly"""
        successful = sum(1 for r in healing_results if r.success)

        alert = Alert(
            timestamp=datetime.now(),
            alert_type=AlertType.ANOMALY_HEALED,
            severity=AlertSeverity.INFO,
            title=f"{anomaly.anomaly_type.upper()} Mitigated: {successful}/{len(healing_results)} actions successful",
            message=f"Anomaly '{anomaly.anomaly_type}' affecting PIDs {anomaly.affected_pids} has been mitigated. "
                   f"Actions: {[r.action.value for r in healing_results]}",
            metadata={
                'anomaly_type': anomaly.anomaly_type,
                'affected_pids': anomaly.affected_pids,
                'healing_results': [
                    {
                        'action': r.action.value,
                        'success': r.success,
                        'pids': r.pids,
                        'message': r.message
                    } for r in healing_results
                ]
            },
            affected_pids=anomaly.affected_pids
        )
        self.send_alert(alert)

    def alert_healing_failed(self, issue_type: str, issue_info, error: str):
        """Create alert for failed healing action"""
        alert = Alert(
            timestamp=datetime.now(),
            alert_type=AlertType.HEALING_FAILED,
            severity=AlertSeverity.HIGH,
            title=f"Healing Failed for {issue_type}",
            message=f"Failed to heal {issue_type}: {error}",
            metadata={
                'issue_type': issue_type,
                'error': error,
                'issue_info': str(issue_info)
            },
            affected_pids=[]
        )
        self.send_alert(alert)

    def alert_system_error(self, error_message: str, context: Dict = None):
        """Create alert for system errors"""
        alert = Alert(
            timestamp=datetime.now(),
            alert_type=AlertType.SYSTEM_ERROR,
            severity=AlertSeverity.MEDIUM,
            title="Guardian System Error",
            message=error_message,
            metadata=context or {},
            affected_pids=[]
        )
        self.send_alert(alert)

    def _map_severity(self, severity_str: str) -> AlertSeverity:
        """Map severity string to AlertSeverity enum"""
        mapping = {
            'critical': AlertSeverity.CRITICAL,
            'high': AlertSeverity.HIGH,
            'medium': AlertSeverity.MEDIUM,
            'low': AlertSeverity.LOW,
        }
        return mapping.get(severity_str.lower(), AlertSeverity.MEDIUM)

    def _log_alert(self, alert: Alert):
        """Log alert to standard logging"""
        level_map = {
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.HIGH: logging.ERROR,
            AlertSeverity.MEDIUM: logging.WARNING,
            AlertSeverity.LOW: logging.INFO,
            AlertSeverity.INFO: logging.INFO,
        }

        log_level = level_map.get(alert.severity, logging.INFO)
        self.logger.log(
            log_level,
            f"[{alert.alert_type.value.upper()}] {alert.title} - {alert.message}"
        )

    def _send_syslog(self, alert: Alert):
        """Send alert to syslog"""
        try:
            priority_map = {
                AlertSeverity.CRITICAL: syslog.LOG_CRIT,
                AlertSeverity.HIGH: syslog.LOG_ERR,
                AlertSeverity.MEDIUM: syslog.LOG_WARNING,
                AlertSeverity.LOW: syslog.LOG_NOTICE,
                AlertSeverity.INFO: syslog.LOG_INFO,
            }

            priority = priority_map.get(alert.severity, syslog.LOG_INFO)
            message = f"[{alert.alert_type.value}] {alert.title}: {alert.message}"
            syslog.syslog(priority, message)

        except Exception as e:
            self.logger.error(f"Failed to send syslog alert: {e}")

    def _send_webhook(self, alert: Alert):
        """Send alert to webhook"""
        try:
            url = self.webhook_config.get('url')
            if not url:
                return

            method = self.webhook_config.get('method', 'POST').upper()
            headers = self.webhook_config.get('headers', {'Content-Type': 'application/json'})

            payload = alert.to_dict()

            if method == 'POST':
                response = requests.post(url, json=payload, headers=headers, timeout=5)
            elif method == 'PUT':
                response = requests.put(url, json=payload, headers=headers, timeout=5)
            else:
                self.logger.warning(f"Unsupported webhook method: {method}")
                return

            if response.status_code not in [200, 201, 204]:
                self.logger.warning(f"Webhook returned status {response.status_code}: {response.text}")

        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")

    def _send_email(self, alert: Alert):
        """Send alert via email"""
        # Placeholder for email implementation
        # Would use smtplib to send email notifications
        try:
            # TODO: Implement email sending
            self.logger.debug(f"Email alert (not implemented): {alert.title}")
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    def get_alert_summary(self) -> Dict:
        """Get summary of alerts"""
        recent = [a for a in self.alert_history
                 if (datetime.now() - a.timestamp).total_seconds() < 3600]

        return {
            'total_alerts': len(self.alert_history),
            'last_hour': len(recent),
            'by_type': {
                alert_type.value: sum(1 for a in self.alert_history if a.alert_type == alert_type)
                for alert_type in AlertType
            },
            'by_severity': {
                severity.value: sum(1 for a in self.alert_history if a.severity == severity)
                for severity in AlertSeverity
            }
        }

    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts"""
        recent = sorted(self.alert_history, key=lambda a: a.timestamp, reverse=True)[:limit]
        return [a.to_dict() for a in recent]
