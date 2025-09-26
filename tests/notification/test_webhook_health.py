"""
Unit tests for webhook health monitoring system.

Tests health metrics tracking, alert generation, performance monitoring,
and failure pattern detection for Discord webhook health monitoring.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from trading_bot.notification.webhook_health import (
    AlertType,
    HealthAlert,
    HealthMetrics,
    HealthReporter,
    HealthThresholds,
    WebhookHealthMonitor,
    create_webhook_health_monitor,
)


class TestHealthMetrics:
    """Test cases for HealthMetrics data tracking."""

    def test_metrics_initialization(self):
        """Test initial metrics state."""
        metrics = HealthMetrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.success_rate == 0.0
        assert metrics.average_response_time == 0.0
        assert metrics.min_response_time is None
        assert metrics.max_response_time is None
        assert len(metrics.recent_response_times) == 0
        assert len(metrics.failure_reasons) == 0
        assert metrics.last_success_time is None
        assert metrics.last_failure_time is None

    def test_success_recording(self):
        """Test recording successful requests."""
        metrics = HealthMetrics()

        # Record successful requests
        metrics.record_success(response_time_ms=150)
        metrics.record_success(response_time_ms=200)

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 0
        assert metrics.success_rate == 100.0
        assert metrics.average_response_time == 175.0
        assert metrics.min_response_time == 150
        assert metrics.max_response_time == 200
        assert metrics.last_success_time is not None

    def test_failure_recording(self):
        """Test recording failed requests."""
        metrics = HealthMetrics()

        # Record failures
        metrics.record_failure("Connection timeout")
        metrics.record_failure("Rate limited")
        metrics.record_failure("Connection timeout")  # Duplicate reason

        assert metrics.total_requests == 3
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 3
        assert metrics.success_rate == 0.0
        assert metrics.failure_reasons["Connection timeout"] == 2
        assert metrics.failure_reasons["Rate limited"] == 1
        assert metrics.last_failure_time is not None

    def test_mixed_success_failure_recording(self):
        """Test recording mixed success and failure requests."""
        metrics = HealthMetrics()

        # Record mixed results
        metrics.record_success(response_time_ms=100)
        metrics.record_failure("Timeout")
        metrics.record_success(response_time_ms=200)
        metrics.record_failure("Rate limit")

        assert metrics.total_requests == 4
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 2
        assert metrics.success_rate == 50.0
        assert metrics.average_response_time == 150.0

    def test_response_time_window_management(self):
        """Test response time window management."""
        metrics = HealthMetrics(response_time_window=3)

        # Add more response times than window size
        for i in range(5):
            metrics.record_success(response_time_ms=(i + 1) * 100)

        # Should only keep last 3 response times
        assert len(metrics.recent_response_times) == 3
        assert metrics.recent_response_times == [300, 400, 500]
        assert metrics.average_response_time == 400.0

    def test_metrics_reset(self):
        """Test resetting metrics."""
        metrics = HealthMetrics()

        # Add some data
        metrics.record_success(response_time_ms=100)
        metrics.record_failure("Test error")

        # Reset metrics
        metrics.reset()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.success_rate == 0.0
        assert len(metrics.recent_response_times) == 0
        assert len(metrics.failure_reasons) == 0

    def test_time_range_filtering(self):
        """Test filtering metrics by time range."""
        metrics = HealthMetrics()

        # Record requests with different timestamps
        now = datetime.utcnow()
        old_time = now - timedelta(hours=2)

        # Mock timestamps for testing
        with patch("trading_bot.notification.webhook_health.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = old_time
            metrics.record_success(response_time_ms=100)

            mock_datetime.utcnow.return_value = now
            metrics.record_success(response_time_ms=200)
            metrics.record_failure("Recent failure")

        # Get metrics for last hour (should only include recent ones)
        recent_metrics = metrics.get_metrics_for_period(hours=1)

        assert recent_metrics.total_requests == 2  # 1 success + 1 failure
        assert recent_metrics.successful_requests == 1
        assert recent_metrics.failed_requests == 1


class TestHealthAlert:
    """Test cases for HealthAlert generation."""

    def test_alert_creation(self):
        """Test creating health alerts."""
        alert = HealthAlert(
            alert_type=AlertType.SUCCESS_RATE_LOW,
            message="Success rate dropped to 75%",
            severity="warning",
            metrics_snapshot=HealthMetrics(),
        )

        assert alert.alert_type == AlertType.SUCCESS_RATE_LOW
        assert alert.message == "Success rate dropped to 75%"
        assert alert.severity == "warning"
        assert alert.timestamp is not None
        assert alert.metrics_snapshot is not None

    def test_alert_serialization(self):
        """Test alert to/from dict conversion."""
        metrics = HealthMetrics()
        metrics.record_success(response_time_ms=100)

        alert = HealthAlert(
            alert_type=AlertType.RESPONSE_TIME_HIGH,
            message="Response time too high",
            severity="critical",
            metrics_snapshot=metrics,
        )

        # Convert to dict
        alert_dict = alert.to_dict()
        assert alert_dict["alert_type"] == "RESPONSE_TIME_HIGH"
        assert alert_dict["message"] == "Response time too high"
        assert alert_dict["severity"] == "critical"
        assert "metrics_snapshot" in alert_dict

        # Convert back from dict
        restored = HealthAlert.from_dict(alert_dict)
        assert restored.alert_type == AlertType.RESPONSE_TIME_HIGH
        assert restored.message == "Response time too high"
        assert restored.severity == "critical"


class TestHealthThresholds:
    """Test cases for HealthThresholds validation."""

    def test_valid_thresholds(self):
        """Test valid threshold configuration."""
        thresholds = HealthThresholds(
            success_rate_warning=90.0,
            success_rate_critical=80.0,
            response_time_warning=1000,
            response_time_critical=2000,
            failure_rate_window=300,
        )

        assert thresholds.success_rate_warning == 90.0
        assert thresholds.success_rate_critical == 80.0
        assert thresholds.response_time_warning == 1000
        assert thresholds.response_time_critical == 2000

    def test_invalid_success_rate_thresholds(self):
        """Test invalid success rate threshold validation."""
        with pytest.raises(ValueError, match="success_rate_warning must be 0-100"):
            HealthThresholds(success_rate_warning=150.0)

        with pytest.raises(ValueError, match="success_rate_critical must be 0-100"):
            HealthThresholds(success_rate_critical=-10.0)

    def test_invalid_response_time_thresholds(self):
        """Test invalid response time threshold validation."""
        with pytest.raises(ValueError, match="response_time_warning must be positive"):
            HealthThresholds(response_time_warning=-100)

        with pytest.raises(ValueError, match="response_time_critical must be positive"):
            HealthThresholds(response_time_critical=0)

    def test_threshold_ordering_validation(self):
        """Test threshold ordering validation."""
        with pytest.raises(
            ValueError, match="success_rate_critical must be <= success_rate_warning"
        ):
            HealthThresholds(
                success_rate_warning=80.0,
                success_rate_critical=90.0,  # Critical should be lower
            )

        with pytest.raises(
            ValueError, match="response_time_critical must be >= response_time_warning"
        ):
            HealthThresholds(
                response_time_warning=2000,
                response_time_critical=1000,  # Critical should be higher
            )


class TestWebhookHealthMonitor:
    """Test cases for WebhookHealthMonitor functionality."""

    def test_monitor_initialization(self):
        """Test health monitor initialization."""
        thresholds = HealthThresholds()
        monitor = WebhookHealthMonitor(thresholds)

        metrics = monitor.get_current_metrics()
        assert metrics.total_requests == 0
        assert monitor.get_health_status() == "healthy"

    @pytest.mark.asyncio
    async def test_record_success(self):
        """Test recording successful webhook requests."""
        thresholds = HealthThresholds()
        monitor = WebhookHealthMonitor(thresholds)

        # Record successful request
        await monitor.record_success("https://discord.com/test", response_time_ms=150)

        metrics = monitor.get_current_metrics()
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.success_rate == 100.0
        assert metrics.average_response_time == 150.0

    @pytest.mark.asyncio
    async def test_record_failure(self):
        """Test recording failed webhook requests."""
        thresholds = HealthThresholds()
        monitor = WebhookHealthMonitor(thresholds)

        # Record failed request
        await monitor.record_failure("https://discord.com/test", "Connection timeout")

        metrics = monitor.get_current_metrics()
        assert metrics.total_requests == 1
        assert metrics.failed_requests == 1
        assert metrics.success_rate == 0.0
        assert "Connection timeout" in metrics.failure_reasons

    @pytest.mark.asyncio
    async def test_success_rate_alert_generation(self):
        """Test alert generation based on success rate thresholds."""
        thresholds = HealthThresholds(
            success_rate_warning=90.0, success_rate_critical=75.0
        )
        monitor = WebhookHealthMonitor(thresholds)

        alert_callback = Mock()
        monitor.register_alert_callback(alert_callback)

        # Record requests that push success rate below warning threshold
        # 7 successes, 3 failures = 70% success rate (below critical)
        for _ in range(7):
            await monitor.record_success(
                "https://discord.com/test", response_time_ms=100
            )
        for _ in range(3):
            await monitor.record_failure("https://discord.com/test", "Timeout")

        # Should trigger critical alert
        assert alert_callback.called
        alert = alert_callback.call_args[0][0]
        assert alert.alert_type == AlertType.SUCCESS_RATE_LOW
        assert alert.severity == "critical"

    @pytest.mark.asyncio
    async def test_response_time_alert_generation(self):
        """Test alert generation based on response time thresholds."""
        thresholds = HealthThresholds(
            response_time_warning=500, response_time_critical=1000
        )
        monitor = WebhookHealthMonitor(thresholds)

        alert_callback = Mock()
        monitor.register_alert_callback(alert_callback)

        # Record requests with high response times
        for i in range(5):
            await monitor.record_success(
                "https://discord.com/test", response_time_ms=1200
            )

        # Should trigger critical alert
        assert alert_callback.called
        alert = alert_callback.call_args[0][0]
        assert alert.alert_type == AlertType.RESPONSE_TIME_HIGH
        assert alert.severity == "critical"

    @pytest.mark.asyncio
    async def test_consecutive_failures_alert(self):
        """Test alert generation for consecutive failures."""
        thresholds = HealthThresholds(consecutive_failure_threshold=3)
        monitor = WebhookHealthMonitor(thresholds)

        alert_callback = Mock()
        monitor.register_alert_callback(alert_callback)

        # Record consecutive failures
        for _ in range(4):  # Exceed threshold of 3
            await monitor.record_failure("https://discord.com/test", "Timeout")

        # Should trigger alert
        assert alert_callback.called
        alert = alert_callback.call_args[0][0]
        assert alert.alert_type == AlertType.CONSECUTIVE_FAILURES

    def test_health_status_calculation(self):
        """Test health status calculation logic."""
        thresholds = HealthThresholds(
            success_rate_warning=90.0,
            success_rate_critical=75.0,
            response_time_warning=500,
            response_time_critical=1000,
        )
        monitor = WebhookHealthMonitor(thresholds)

        # Initially healthy
        assert monitor.get_health_status() == "healthy"

        # Add requests to simulate warning conditions
        monitor._current_metrics.record_success(response_time_ms=600)  # Above warning
        monitor._current_metrics.record_success(response_time_ms=400)
        monitor._current_metrics.record_failure("Timeout")  # 66.7% success rate

        # Should be critical (below critical thresholds)
        assert monitor.get_health_status() == "critical"

    @pytest.mark.asyncio
    async def test_metrics_reset(self):
        """Test resetting health metrics."""
        thresholds = HealthThresholds()
        monitor = WebhookHealthMonitor(thresholds)

        # Add some data
        await monitor.record_success("https://discord.com/test", response_time_ms=100)
        await monitor.record_failure("https://discord.com/test", "Error")

        # Reset metrics
        monitor.reset_metrics()

        metrics = monitor.get_current_metrics()
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0

    def test_alert_callback_management(self):
        """Test alert callback registration and removal."""
        thresholds = HealthThresholds()
        monitor = WebhookHealthMonitor(thresholds)

        callback1 = Mock()
        callback2 = Mock()

        # Register callbacks
        monitor.register_alert_callback(callback1)
        monitor.register_alert_callback(callback2)

        # Remove one callback
        monitor.remove_alert_callback(callback1)

        # Trigger alert
        monitor._trigger_alert(AlertType.SUCCESS_RATE_LOW, "Test alert", "warning")

        # Only callback2 should be called
        callback1.assert_not_called()
        callback2.assert_called_once()

    @pytest.mark.asyncio
    async def test_periodic_health_check(self):
        """Test periodic health status checking."""
        thresholds = HealthThresholds()
        monitor = WebhookHealthMonitor(thresholds)

        status_callback = Mock()
        monitor.register_status_callback(status_callback)

        # Start periodic checking with short interval
        await monitor.start_periodic_checks(interval_seconds=0.1)

        try:
            # Wait for at least one check
            await asyncio.sleep(0.2)

            # Status callback should have been called
            assert status_callback.called

        finally:
            await monitor.stop_periodic_checks()


class TestHealthReporter:
    """Test cases for HealthReporter functionality."""

    def test_reporter_initialization(self):
        """Test health reporter initialization."""
        monitor = WebhookHealthMonitor(HealthThresholds())
        reporter = HealthReporter(monitor)

        assert reporter._monitor == monitor

    def test_generate_health_report(self):
        """Test health report generation."""
        monitor = WebhookHealthMonitor(HealthThresholds())
        reporter = HealthReporter(monitor)

        # Add some metrics
        monitor._current_metrics.record_success(response_time_ms=150)
        monitor._current_metrics.record_success(response_time_ms=200)
        monitor._current_metrics.record_failure("Timeout")

        # Generate report
        report = reporter.generate_health_report()

        assert "Health Status" in report
        assert "Total Requests: 3" in report
        assert "Success Rate: 66.67%" in report
        assert "Average Response Time: 175.0ms" in report
        assert "Timeout: 1" in report

    def test_generate_summary_report(self):
        """Test summary report generation."""
        monitor = WebhookHealthMonitor(HealthThresholds())
        reporter = HealthReporter(monitor)

        # Add metrics
        monitor._current_metrics.record_success(response_time_ms=100)
        monitor._current_metrics.record_success(response_time_ms=200)

        # Generate summary
        summary = reporter.generate_summary_report()

        assert "healthy" in summary.lower()
        assert "100.00%" in summary  # Success rate
        assert "150.0ms" in summary  # Average response time

    @pytest.mark.asyncio
    async def test_export_metrics_to_dict(self):
        """Test exporting metrics to dictionary format."""
        monitor = WebhookHealthMonitor(HealthThresholds())
        reporter = HealthReporter(monitor)

        # Add some data
        await monitor.record_success("https://discord.com/test", response_time_ms=150)
        await monitor.record_failure("https://discord.com/test", "Rate limited")

        # Export metrics
        metrics_dict = reporter.export_metrics_to_dict()

        assert metrics_dict["total_requests"] == 2
        assert metrics_dict["successful_requests"] == 1
        assert metrics_dict["failed_requests"] == 1
        assert metrics_dict["success_rate"] == 50.0
        assert "failure_reasons" in metrics_dict
        assert "health_status" in metrics_dict

    def test_export_alerts_to_dict(self):
        """Test exporting alerts to dictionary format."""
        monitor = WebhookHealthMonitor(HealthThresholds())
        reporter = HealthReporter(monitor)

        # Generate some alerts
        monitor._trigger_alert(
            AlertType.SUCCESS_RATE_LOW, "Low success rate", "warning"
        )
        monitor._trigger_alert(
            AlertType.RESPONSE_TIME_HIGH, "High response time", "critical"
        )

        # Export alerts
        alerts_dict = reporter.export_alerts_to_dict()

        assert len(alerts_dict["alerts"]) == 2
        assert alerts_dict["total_alerts"] == 2
        assert alerts_dict["warning_alerts"] == 1
        assert alerts_dict["critical_alerts"] == 1


class TestHealthMonitoringIntegration:
    """Integration test cases for health monitoring system."""

    @pytest.mark.asyncio
    async def test_end_to_end_monitoring(self):
        """Test complete health monitoring workflow."""
        thresholds = HealthThresholds(
            success_rate_warning=90.0,
            response_time_warning=500,
            consecutive_failure_threshold=2,
        )
        monitor = WebhookHealthMonitor(thresholds)
        reporter = HealthReporter(monitor)

        alert_callback = Mock()
        status_callback = Mock()

        monitor.register_alert_callback(alert_callback)
        monitor.register_status_callback(status_callback)

        # Simulate webhook activity
        webhook_url = "https://discord.com/api/webhooks/test"

        # Normal operation
        for _ in range(10):
            await monitor.record_success(webhook_url, response_time_ms=100)

        # Degraded performance
        for _ in range(5):
            await monitor.record_success(webhook_url, response_time_ms=600)  # Slow

        # Some failures
        for _ in range(3):
            await monitor.record_failure(webhook_url, "Rate limited")

        # Generate comprehensive report
        report = reporter.generate_health_report()
        reporter.generate_summary_report()
        metrics_export = reporter.export_metrics_to_dict()

        # Verify monitoring captured the activity
        assert "Total Requests: 18" in report
        assert metrics_export["total_requests"] == 18
        assert metrics_export["success_rate"] < 90.0  # Should be degraded

        # Alerts should have been triggered
        assert alert_callback.called


class TestFactoryFunction:
    """Test cases for create_webhook_health_monitor factory."""

    def test_create_with_defaults(self):
        """Test creating monitor with default configuration."""
        monitor = create_webhook_health_monitor()
        assert isinstance(monitor, WebhookHealthMonitor)
        assert monitor.get_health_status() == "healthy"

    def test_create_with_custom_thresholds(self):
        """Test creating monitor with custom thresholds."""
        thresholds = HealthThresholds(
            success_rate_warning=95.0, response_time_warning=300
        )
        monitor = create_webhook_health_monitor(thresholds)
        assert isinstance(monitor, WebhookHealthMonitor)

    def test_create_with_callbacks(self):
        """Test creating monitor with alert and status callbacks."""
        alert_callback = Mock()
        status_callback = Mock()

        monitor = create_webhook_health_monitor(
            alert_callback=alert_callback, status_callback=status_callback
        )

        # Trigger alert to test callback registration
        monitor._trigger_alert(AlertType.SUCCESS_RATE_LOW, "Test alert", "warning")

        alert_callback.assert_called_once()
