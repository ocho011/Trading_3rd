"""
Webhook health monitoring and metrics tracking for Discord notifications.

Provides comprehensive health monitoring, performance metrics, and alerting
for Discord webhook operations. Tracks success rates, response times,
rate limiting events, and overall service health.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from statistics import mean, median


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetrics:
    """
    Comprehensive health metrics for webhook monitoring.

    Attributes:
        total_requests: Total number of webhook requests
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        rate_limited_requests: Number of rate-limited requests
        timeout_requests: Number of timeout requests
        success_rate: Success rate percentage (0-100)
        failure_rate: Failure rate percentage (0-100)
        avg_response_time: Average response time in seconds
        median_response_time: Median response time in seconds
        min_response_time: Minimum response time in seconds
        max_response_time: Maximum response time in seconds
        requests_per_minute: Current requests per minute rate
        health_status: Current health status
        last_success_time: Timestamp of last successful request
        last_failure_time: Timestamp of last failed request
        consecutive_failures: Current consecutive failures
        uptime_percentage: Uptime percentage over monitoring period
        circuit_breaker_trips: Number of circuit breaker activations
        queue_size: Current message queue size
        processing_lag: Average processing lag in seconds
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    timeout_requests: int = 0
    success_rate: float = 0.0
    failure_rate: float = 0.0
    avg_response_time: float = 0.0
    median_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = float('inf')
    requests_per_minute: float = 0.0
    health_status: HealthStatus = HealthStatus.HEALTHY
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0
    circuit_breaker_trips: int = 0
    queue_size: int = 0
    processing_lag: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        data = asdict(self)
        data['health_status'] = self.health_status.value
        return data


@dataclass
class HealthAlert:
    """
    Health monitoring alert.

    Attributes:
        id: Unique alert identifier
        severity: Alert severity level
        title: Alert title/summary
        message: Detailed alert message
        timestamp: Alert creation timestamp
        metric_name: Name of metric that triggered alert
        metric_value: Value that triggered alert
        threshold: Threshold that was exceeded
        resolved: Whether alert has been resolved
        resolved_at: Timestamp when alert was resolved
    """
    id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: float
    metric_name: str
    metric_value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        data = asdict(self)
        data['severity'] = self.severity.value
        return data


@dataclass
class HealthThresholds:
    """
    Health monitoring thresholds configuration.

    Attributes:
        success_rate_warning: Success rate below this triggers warning (0-100)
        success_rate_critical: Success rate below this triggers critical (0-100)
        response_time_warning: Response time above this triggers warning (seconds)
        response_time_critical: Response time above this triggers critical (seconds)
        consecutive_failures_warning: Consecutive failures for warning
        consecutive_failures_critical: Consecutive failures for critical
        rate_limit_percentage_warning: Rate limit percentage for warning
        queue_size_warning: Queue size for warning
        queue_size_critical: Queue size for critical
        processing_lag_warning: Processing lag for warning (seconds)
        processing_lag_critical: Processing lag for critical (seconds)
    """
    success_rate_warning: float = 85.0
    success_rate_critical: float = 70.0
    response_time_warning: float = 5.0
    response_time_critical: float = 10.0
    consecutive_failures_warning: int = 3
    consecutive_failures_critical: int = 5
    rate_limit_percentage_warning: float = 10.0
    queue_size_warning: int = 100
    queue_size_critical: int = 500
    processing_lag_warning: float = 60.0
    processing_lag_critical: float = 300.0


class WebhookHealthMonitor:
    """
    Comprehensive health monitoring for Discord webhooks.

    Tracks performance metrics, detects issues, generates alerts,
    and provides health status reporting with configurable thresholds.
    """

    def __init__(
        self,
        thresholds: Optional[HealthThresholds] = None,
        monitoring_window_minutes: int = 60,
        alert_callback: Optional[Callable[[HealthAlert], None]] = None
    ) -> None:
        """
        Initialize health monitor.

        Args:
            thresholds: Health threshold configuration
            monitoring_window_minutes: Time window for metrics calculation
            alert_callback: Optional callback for health alerts
        """
        self._thresholds = thresholds or HealthThresholds()
        self._monitoring_window = monitoring_window_minutes * 60  # Convert to seconds
        self._alert_callback = alert_callback
        self._logger = logging.getLogger(__name__)

        # Thread safety
        self._lock = threading.RLock()

        # Metrics tracking
        self._start_time = time.time()
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._rate_limited_requests = 0
        self._timeout_requests = 0
        self._consecutive_failures = 0
        self._circuit_breaker_trips = 0

        # Time-series data (timestamp, success, response_time)
        self._request_history: deque[Tuple[float, bool, float]] = deque()
        self._last_success_time: Optional[float] = None
        self._last_failure_time: Optional[float] = None

        # Active alerts
        self._active_alerts: Dict[str, HealthAlert] = {}
        self._alert_counter = 0

        # Queue monitoring
        self._current_queue_size = 0
        self._processing_times: deque[float] = deque(maxlen=100)

    def record_request_start(self) -> float:
        """
        Record start of webhook request.

        Returns:
            float: Timestamp for measuring duration
        """
        return time.time()

    def record_request_success(self, start_time: float, response_time: Optional[float] = None) -> None:
        """
        Record successful webhook request.

        Args:
            start_time: Request start timestamp
            response_time: Optional response time override
        """
        current_time = time.time()
        duration = response_time if response_time is not None else current_time - start_time

        with self._lock:
            self._total_requests += 1
            self._successful_requests += 1
            self._consecutive_failures = 0
            self._last_success_time = current_time

            # Add to history
            self._request_history.append((current_time, True, duration))
            self._cleanup_old_history()

            # Check for resolved failure alerts
            self._check_resolved_alerts()

        self._logger.debug(f"Recorded successful request (duration: {duration:.2f}s)")

    def record_request_failure(
        self,
        start_time: float,
        error_type: str = "unknown",
        is_rate_limited: bool = False,
        is_timeout: bool = False
    ) -> None:
        """
        Record failed webhook request.

        Args:
            start_time: Request start timestamp
            error_type: Type of error that occurred
            is_rate_limited: Whether failure was due to rate limiting
            is_timeout: Whether failure was due to timeout
        """
        current_time = time.time()
        duration = current_time - start_time

        with self._lock:
            self._total_requests += 1
            self._failed_requests += 1
            self._consecutive_failures += 1
            self._last_failure_time = current_time

            if is_rate_limited:
                self._rate_limited_requests += 1
            if is_timeout:
                self._timeout_requests += 1

            # Add to history
            self._request_history.append((current_time, False, duration))
            self._cleanup_old_history()

            # Check for new alerts
            self._check_health_alerts()

        self._logger.debug(
            f"Recorded failed request (duration: {duration:.2f}s, type: {error_type}, "
            f"consecutive_failures: {self._consecutive_failures})"
        )

    def record_circuit_breaker_trip(self) -> None:
        """Record circuit breaker activation."""
        with self._lock:
            self._circuit_breaker_trips += 1

        self._logger.warning("Recorded circuit breaker trip")

    def update_queue_size(self, size: int) -> None:
        """
        Update current queue size.

        Args:
            size: Current queue size
        """
        with self._lock:
            self._current_queue_size = size

            # Check queue size alerts
            self._check_queue_alerts()

    def record_processing_time(self, processing_time: float) -> None:
        """
        Record message processing time.

        Args:
            processing_time: Time taken to process message
        """
        with self._lock:
            self._processing_times.append(processing_time)

            # Check processing lag alerts
            self._check_processing_lag_alerts()

    def get_health_metrics(self) -> HealthMetrics:
        """
        Get current health metrics.

        Returns:
            HealthMetrics: Current health status and metrics
        """
        with self._lock:
            # Calculate rates
            success_rate = 0.0
            failure_rate = 0.0
            if self._total_requests > 0:
                success_rate = (self._successful_requests / self._total_requests) * 100
                failure_rate = (self._failed_requests / self._total_requests) * 100

            # Calculate response time statistics
            recent_responses = [rt for _, _, rt in self._request_history]
            avg_response_time = mean(recent_responses) if recent_responses else 0.0
            median_response_time = median(recent_responses) if recent_responses else 0.0
            min_response_time = min(recent_responses) if recent_responses else 0.0
            max_response_time = max(recent_responses) if recent_responses else 0.0

            # Calculate requests per minute
            current_time = time.time()
            minute_ago = current_time - 60
            recent_requests = [ts for ts, _, _ in self._request_history if ts >= minute_ago]
            requests_per_minute = len(recent_requests)

            # Calculate uptime
            total_time = current_time - self._start_time
            uptime_percentage = self._calculate_uptime_percentage()

            # Calculate processing lag
            processing_lag = mean(self._processing_times) if self._processing_times else 0.0

            # Determine health status
            health_status = self._determine_health_status(success_rate, avg_response_time)

            return HealthMetrics(
                total_requests=self._total_requests,
                successful_requests=self._successful_requests,
                failed_requests=self._failed_requests,
                rate_limited_requests=self._rate_limited_requests,
                timeout_requests=self._timeout_requests,
                success_rate=success_rate,
                failure_rate=failure_rate,
                avg_response_time=avg_response_time,
                median_response_time=median_response_time,
                min_response_time=min_response_time,
                max_response_time=max_response_time,
                requests_per_minute=requests_per_minute,
                health_status=health_status,
                last_success_time=self._last_success_time,
                last_failure_time=self._last_failure_time,
                consecutive_failures=self._consecutive_failures,
                uptime_percentage=uptime_percentage,
                circuit_breaker_trips=self._circuit_breaker_trips,
                queue_size=self._current_queue_size,
                processing_lag=processing_lag
            )

    def get_active_alerts(self) -> List[HealthAlert]:
        """
        Get all active alerts.

        Returns:
            List[HealthAlert]: List of unresolved alerts
        """
        with self._lock:
            return [alert for alert in self._active_alerts.values() if not alert.resolved]

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Manually resolve an alert.

        Args:
            alert_id: Alert identifier

        Returns:
            bool: True if alert was resolved
        """
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = time.time()
                self._logger.info(f"Manually resolved alert: {alert.title}")
                return True

        return False

    def reset_metrics(self) -> None:
        """Reset all metrics and alerts."""
        with self._lock:
            self._start_time = time.time()
            self._total_requests = 0
            self._successful_requests = 0
            self._failed_requests = 0
            self._rate_limited_requests = 0
            self._timeout_requests = 0
            self._consecutive_failures = 0
            self._circuit_breaker_trips = 0
            self._current_queue_size = 0
            self._request_history.clear()
            self._processing_times.clear()
            self._last_success_time = None
            self._last_failure_time = None
            self._active_alerts.clear()
            self._alert_counter = 0

        self._logger.info("Health metrics reset")

    def _cleanup_old_history(self) -> None:
        """Remove old entries from request history."""
        cutoff_time = time.time() - self._monitoring_window
        while self._request_history and self._request_history[0][0] < cutoff_time:
            self._request_history.popleft()

    def _calculate_uptime_percentage(self) -> float:
        """Calculate uptime percentage based on recent history."""
        if not self._request_history:
            return 100.0

        successful_requests = sum(1 for _, success, _ in self._request_history if success)
        total_requests = len(self._request_history)

        if total_requests == 0:
            return 100.0

        return (successful_requests / total_requests) * 100

    def _determine_health_status(self, success_rate: float, avg_response_time: float) -> HealthStatus:
        """
        Determine overall health status based on metrics.

        Args:
            success_rate: Current success rate
            avg_response_time: Average response time

        Returns:
            HealthStatus: Determined health status
        """
        # Critical conditions
        if (success_rate < self._thresholds.success_rate_critical or
            avg_response_time > self._thresholds.response_time_critical or
            self._consecutive_failures >= self._thresholds.consecutive_failures_critical):
            return HealthStatus.CRITICAL

        # Unhealthy conditions
        if (success_rate < self._thresholds.success_rate_warning or
            avg_response_time > self._thresholds.response_time_warning or
            self._consecutive_failures >= self._thresholds.consecutive_failures_warning):
            return HealthStatus.UNHEALTHY

        # Check for degraded performance
        if (self._current_queue_size >= self._thresholds.queue_size_warning or
            (self._processing_times and mean(self._processing_times) > self._thresholds.processing_lag_warning)):
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def _check_health_alerts(self) -> None:
        """Check for health-related alert conditions."""
        current_metrics = self.get_health_metrics()

        # Success rate alerts
        if current_metrics.success_rate < self._thresholds.success_rate_critical:
            self._create_alert(
                AlertSeverity.CRITICAL,
                "Critical Success Rate",
                f"Success rate dropped to {current_metrics.success_rate:.1f}%",
                "success_rate",
                current_metrics.success_rate,
                self._thresholds.success_rate_critical
            )
        elif current_metrics.success_rate < self._thresholds.success_rate_warning:
            self._create_alert(
                AlertSeverity.WARNING,
                "Low Success Rate",
                f"Success rate dropped to {current_metrics.success_rate:.1f}%",
                "success_rate",
                current_metrics.success_rate,
                self._thresholds.success_rate_warning
            )

        # Response time alerts
        if current_metrics.avg_response_time > self._thresholds.response_time_critical:
            self._create_alert(
                AlertSeverity.CRITICAL,
                "Critical Response Time",
                f"Average response time is {current_metrics.avg_response_time:.2f}s",
                "response_time",
                current_metrics.avg_response_time,
                self._thresholds.response_time_critical
            )
        elif current_metrics.avg_response_time > self._thresholds.response_time_warning:
            self._create_alert(
                AlertSeverity.WARNING,
                "High Response Time",
                f"Average response time is {current_metrics.avg_response_time:.2f}s",
                "response_time",
                current_metrics.avg_response_time,
                self._thresholds.response_time_warning
            )

        # Consecutive failures alerts
        if self._consecutive_failures >= self._thresholds.consecutive_failures_critical:
            self._create_alert(
                AlertSeverity.CRITICAL,
                "Critical Consecutive Failures",
                f"{self._consecutive_failures} consecutive failures detected",
                "consecutive_failures",
                self._consecutive_failures,
                self._thresholds.consecutive_failures_critical
            )
        elif self._consecutive_failures >= self._thresholds.consecutive_failures_warning:
            self._create_alert(
                AlertSeverity.WARNING,
                "Multiple Consecutive Failures",
                f"{self._consecutive_failures} consecutive failures detected",
                "consecutive_failures",
                self._consecutive_failures,
                self._thresholds.consecutive_failures_warning
            )

    def _check_queue_alerts(self) -> None:
        """Check for queue-related alert conditions."""
        if self._current_queue_size >= self._thresholds.queue_size_critical:
            self._create_alert(
                AlertSeverity.CRITICAL,
                "Critical Queue Size",
                f"Queue size has reached {self._current_queue_size} messages",
                "queue_size",
                self._current_queue_size,
                self._thresholds.queue_size_critical
            )
        elif self._current_queue_size >= self._thresholds.queue_size_warning:
            self._create_alert(
                AlertSeverity.WARNING,
                "High Queue Size",
                f"Queue size has reached {self._current_queue_size} messages",
                "queue_size",
                self._current_queue_size,
                self._thresholds.queue_size_warning
            )

    def _check_processing_lag_alerts(self) -> None:
        """Check for processing lag alert conditions."""
        if not self._processing_times:
            return

        avg_processing_time = mean(self._processing_times)

        if avg_processing_time > self._thresholds.processing_lag_critical:
            self._create_alert(
                AlertSeverity.CRITICAL,
                "Critical Processing Lag",
                f"Average processing time is {avg_processing_time:.2f}s",
                "processing_lag",
                avg_processing_time,
                self._thresholds.processing_lag_critical
            )
        elif avg_processing_time > self._thresholds.processing_lag_warning:
            self._create_alert(
                AlertSeverity.WARNING,
                "High Processing Lag",
                f"Average processing time is {avg_processing_time:.2f}s",
                "processing_lag",
                avg_processing_time,
                self._thresholds.processing_lag_warning
            )

    def _check_resolved_alerts(self) -> None:
        """Check if any alerts can be automatically resolved."""
        current_metrics = self.get_health_metrics()

        for alert in list(self._active_alerts.values()):
            if alert.resolved:
                continue

            should_resolve = False

            # Check resolution conditions based on alert type
            if alert.metric_name == "success_rate":
                should_resolve = current_metrics.success_rate >= alert.threshold
            elif alert.metric_name == "response_time":
                should_resolve = current_metrics.avg_response_time <= alert.threshold
            elif alert.metric_name == "consecutive_failures":
                should_resolve = self._consecutive_failures < alert.threshold
            elif alert.metric_name == "queue_size":
                should_resolve = self._current_queue_size < alert.threshold
            elif alert.metric_name == "processing_lag":
                if self._processing_times:
                    should_resolve = mean(self._processing_times) <= alert.threshold

            if should_resolve:
                alert.resolved = True
                alert.resolved_at = time.time()
                self._logger.info(f"Auto-resolved alert: {alert.title}")

    def _create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float
    ) -> None:
        """Create new health alert."""
        # Check if similar alert already exists
        alert_key = f"{metric_name}_{severity.value}"
        if alert_key in self._active_alerts and not self._active_alerts[alert_key].resolved:
            return  # Don't create duplicate alerts

        self._alert_counter += 1
        alert = HealthAlert(
            id=f"alert_{self._alert_counter}",
            severity=severity,
            title=title,
            message=message,
            timestamp=time.time(),
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold
        )

        self._active_alerts[alert_key] = alert

        # Call alert callback if provided
        if self._alert_callback:
            try:
                self._alert_callback(alert)
            except Exception as e:
                self._logger.error(f"Error calling alert callback: {e}")

        self._logger.warning(f"Health alert created: {title} - {message}")


def create_health_monitor(
    success_rate_warning: float = 85.0,
    response_time_warning: float = 5.0,
    alert_callback: Optional[Callable[[HealthAlert], None]] = None
) -> WebhookHealthMonitor:
    """
    Factory function to create health monitor.

    Args:
        success_rate_warning: Success rate threshold for warnings
        response_time_warning: Response time threshold for warnings
        alert_callback: Optional callback for alerts

    Returns:
        WebhookHealthMonitor: Configured health monitor
    """
    thresholds = HealthThresholds(
        success_rate_warning=success_rate_warning,
        response_time_warning=response_time_warning
    )

    return WebhookHealthMonitor(
        thresholds=thresholds,
        alert_callback=alert_callback
    )