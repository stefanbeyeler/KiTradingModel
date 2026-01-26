"""
Resource Monitor Service for host protection.

Uses psutil for system resource monitoring.
Provides CPU, memory, and GPU utilization metrics.
Prevents training from starting when resources are constrained.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import psutil
from loguru import logger


@dataclass
class ResourceMetrics:
    """Current system resource metrics."""

    cpu_percent: float  # Overall CPU usage (0-100)
    cpu_per_core: list[float]  # Per-core usage
    memory_percent: float  # RAM usage (0-100)
    memory_used_gb: float  # RAM used in GB
    memory_available_gb: float  # RAM available in GB
    swap_percent: float  # Swap usage
    load_average: tuple[float, float, float]  # 1, 5, 15 minute load averages
    timestamp: datetime

    @property
    def is_critical(self) -> bool:
        """Check if resources are in critical state (>90%)."""
        return self.cpu_percent > 90 or self.memory_percent > 90

    @property
    def is_warning(self) -> bool:
        """Check if resources are in warning state (>75% CPU or >80% memory)."""
        return self.cpu_percent > 75 or self.memory_percent > 80


class ResourceMonitor:
    """
    Monitors system resources and provides protection thresholds.

    Prevents training jobs from starting when system resources are constrained.
    Supports callbacks for automatic actions when thresholds are exceeded.

    Thresholds (configurable):
    - CPU Warning: 75%
    - CPU Critical: 90%
    - Memory Warning: 80%
    - Memory Critical: 90%
    """

    def __init__(
        self,
        cpu_warning: float = 75.0,
        cpu_critical: float = 90.0,
        memory_warning: float = 80.0,
        memory_critical: float = 90.0,
        poll_interval: float = 5.0,
    ):
        """
        Initialize resource monitor.

        Args:
            cpu_warning: CPU percentage threshold for warning state
            cpu_critical: CPU percentage threshold for critical state
            memory_warning: Memory percentage threshold for warning state
            memory_critical: Memory percentage threshold for critical state
            poll_interval: Seconds between background monitoring checks
        """
        self.cpu_warning = cpu_warning
        self.cpu_critical = cpu_critical
        self.memory_warning = memory_warning
        self.memory_critical = memory_critical
        self.poll_interval = poll_interval

        self._latest_metrics: Optional[ResourceMetrics] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._callbacks: list[Callable] = []
        self._last_alert_level: Optional[str] = None

    def get_metrics(self) -> ResourceMetrics:
        """
        Get current resource metrics (synchronous).

        Returns:
            ResourceMetrics with current system state
        """
        # CPU percentage with short interval for accuracy
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(percpu=True)

        # Memory statistics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # System load average (1, 5, 15 minutes)
        load = psutil.getloadavg()

        return ResourceMetrics(
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            swap_percent=swap.percent,
            load_average=load,
            timestamp=datetime.now(timezone.utc),
        )

    def can_start_training(self) -> tuple[bool, str]:
        """
        Check if system resources allow starting new training.

        Uses warning thresholds (not critical) to prevent training
        from pushing the system into critical state.

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        metrics = self.get_metrics()

        # Check critical first (hard block)
        if metrics.cpu_percent > self.cpu_critical:
            return False, f"CPU critical: {metrics.cpu_percent:.1f}% (>{self.cpu_critical}%)"

        if metrics.memory_percent > self.memory_critical:
            return False, f"Memory critical: {metrics.memory_percent:.1f}% (>{self.memory_critical}%)"

        # Check warning (soft block for training)
        if metrics.cpu_percent > self.cpu_warning:
            return False, f"CPU high: {metrics.cpu_percent:.1f}% (>{self.cpu_warning}%)"

        if metrics.memory_percent > self.memory_warning:
            return False, f"Memory high: {metrics.memory_percent:.1f}% (>{self.memory_warning}%)"

        return True, "Resources available"

    def get_status(self) -> str:
        """
        Get current resource status level.

        Returns:
            'critical', 'warning', or 'healthy'
        """
        metrics = self._latest_metrics or self.get_metrics()
        if metrics.is_critical:
            return "critical"
        elif metrics.is_warning:
            return "warning"
        return "healthy"

    async def start(self) -> None:
        """Start background monitoring loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"Resource monitoring started (CPU warning={self.cpu_warning}%, "
            f"memory warning={self.memory_warning}%)"
        )

    async def stop(self) -> None:
        """Stop background monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Background loop for continuous monitoring."""
        while self._running:
            try:
                self._latest_metrics = self.get_metrics()

                # Determine current alert level
                current_level = None
                if self._latest_metrics.is_critical:
                    current_level = "critical"
                elif self._latest_metrics.is_warning:
                    current_level = "warning"

                # Trigger callbacks only on state change (not every poll)
                if current_level != self._last_alert_level:
                    if current_level:
                        for callback in self._callbacks:
                            try:
                                await callback(self._latest_metrics, current_level)
                            except Exception as e:
                                logger.error(f"Error in resource callback: {e}")
                    self._last_alert_level = current_level

                await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitor error: {e}")
                await asyncio.sleep(10)

    def register_callback(self, callback: Callable) -> None:
        """
        Register callback for threshold alerts.

        Callback signature: async def callback(metrics: ResourceMetrics, level: str)
        Level is either 'warning' or 'critical'.
        """
        self._callbacks.append(callback)

    def to_dict(self) -> dict[str, Any]:
        """
        Export current state for API response.

        Returns:
            Dictionary with all resource metrics and thresholds
        """
        metrics = self._latest_metrics or self.get_metrics()
        can_train, reason = self.can_start_training()

        return {
            "cpu_percent": round(metrics.cpu_percent, 1),
            "cpu_per_core": [round(c, 1) for c in metrics.cpu_per_core],
            "cpu_cores": len(metrics.cpu_per_core),
            "memory_percent": round(metrics.memory_percent, 1),
            "memory_used_gb": round(metrics.memory_used_gb, 2),
            "memory_available_gb": round(metrics.memory_available_gb, 2),
            "memory_total_gb": round(metrics.memory_used_gb + metrics.memory_available_gb, 2),
            "swap_percent": round(metrics.swap_percent, 1),
            "load_average": {
                "1min": round(metrics.load_average[0], 2),
                "5min": round(metrics.load_average[1], 2),
                "15min": round(metrics.load_average[2], 2),
            },
            "thresholds": {
                "cpu_warning": self.cpu_warning,
                "cpu_critical": self.cpu_critical,
                "memory_warning": self.memory_warning,
                "memory_critical": self.memory_critical,
            },
            "status": self.get_status(),
            "can_start_training": can_train,
            "training_block_reason": reason if not can_train else None,
            "monitoring_active": self._running,
            "timestamp": metrics.timestamp.isoformat(),
        }


# Singleton instance with default thresholds
# Will be reconfigured from settings in main.py
resource_monitor = ResourceMonitor()
