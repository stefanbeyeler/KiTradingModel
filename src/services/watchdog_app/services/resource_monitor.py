"""
Resource Monitor Service for host protection.

Uses psutil for system resource monitoring.
Uses pynvml (nvidia-ml-py) for GPU monitoring.
Provides CPU, memory, and GPU utilization metrics.
Prevents training from starting when resources are constrained.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import psutil
from loguru import logger

# Optional GPU monitoring via nvidia-ml-py
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.info("pynvml not available - GPU monitoring disabled")


@dataclass
class GPUMetrics:
    """GPU metrics from nvidia-ml-py."""

    gpu_index: int = 0
    name: str = "Unknown"
    memory_total_mb: float = 0.0
    memory_used_mb: float = 0.0
    memory_free_mb: float = 0.0
    memory_percent: float = 0.0
    gpu_utilization: float = 0.0
    temperature_celsius: float = 0.0
    power_usage_watts: float = 0.0
    is_healthy: bool = True
    error_message: Optional[str] = None


@dataclass
class ResourceMetrics:
    """Current system resource metrics including GPU."""

    cpu_percent: float  # Overall CPU usage (0-100)
    cpu_per_core: list[float]  # Per-core usage
    memory_percent: float  # RAM usage (0-100)
    memory_used_gb: float  # RAM used in GB
    memory_available_gb: float  # RAM available in GB
    swap_percent: float  # Swap usage
    load_average: tuple[float, float, float]  # 1, 5, 15 minute load averages
    timestamp: datetime
    gpu: Optional[GPUMetrics] = None  # GPU metrics if available

    @property
    def is_critical(self) -> bool:
        """Check if resources are in critical state (>90%)."""
        cpu_critical = self.cpu_percent > 90
        memory_critical = self.memory_percent > 90
        gpu_critical = self.gpu is not None and (
            self.gpu.memory_percent > 95 or
            self.gpu.temperature_celsius > 85 or
            not self.gpu.is_healthy
        )
        return cpu_critical or memory_critical or gpu_critical

    @property
    def is_warning(self) -> bool:
        """Check if resources are in warning state (>75% CPU or >80% memory)."""
        cpu_warning = self.cpu_percent > 75
        memory_warning = self.memory_percent > 80
        gpu_warning = self.gpu is not None and (
            self.gpu.memory_percent > 85 or
            self.gpu.temperature_celsius > 75
        )
        return cpu_warning or memory_warning or gpu_warning


class ResourceMonitor:
    """
    Monitors system resources including GPU and provides protection thresholds.

    Prevents training jobs from starting when system resources are constrained.
    Supports callbacks for automatic actions when thresholds are exceeded.

    Thresholds (configurable):
    - CPU Warning: 75%
    - CPU Critical: 90%
    - Memory Warning: 80%
    - Memory Critical: 90%
    - GPU Memory Warning: 85%
    - GPU Memory Critical: 95%
    - GPU Temperature Warning: 75°C
    - GPU Temperature Critical: 85°C
    """

    def __init__(
        self,
        cpu_warning: float = 75.0,
        cpu_critical: float = 90.0,
        memory_warning: float = 80.0,
        memory_critical: float = 90.0,
        gpu_memory_warning: float = 85.0,
        gpu_memory_critical: float = 95.0,
        gpu_temp_warning: float = 75.0,
        gpu_temp_critical: float = 85.0,
        poll_interval: float = 5.0,
    ):
        """
        Initialize resource monitor with GPU support.

        Args:
            cpu_warning: CPU percentage threshold for warning state
            cpu_critical: CPU percentage threshold for critical state
            memory_warning: Memory percentage threshold for warning state
            memory_critical: Memory percentage threshold for critical state
            gpu_memory_warning: GPU memory percentage threshold for warning
            gpu_memory_critical: GPU memory percentage threshold for critical
            gpu_temp_warning: GPU temperature threshold for warning (°C)
            gpu_temp_critical: GPU temperature threshold for critical (°C)
            poll_interval: Seconds between background monitoring checks
        """
        self.cpu_warning = cpu_warning
        self.cpu_critical = cpu_critical
        self.memory_warning = memory_warning
        self.memory_critical = memory_critical
        self.gpu_memory_warning = gpu_memory_warning
        self.gpu_memory_critical = gpu_memory_critical
        self.gpu_temp_warning = gpu_temp_warning
        self.gpu_temp_critical = gpu_temp_critical
        self.poll_interval = poll_interval

        self._latest_metrics: Optional[ResourceMetrics] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._callbacks: list[Callable] = []
        self._last_alert_level: Optional[str] = None
        self._nvml_initialized = False

        # Initialize NVML for GPU monitoring
        self._init_nvml()

    def _init_nvml(self) -> None:
        """Initialize NVIDIA Management Library for GPU monitoring."""
        if not PYNVML_AVAILABLE:
            return

        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"NVML initialized - {device_count} GPU(s) detected")
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e}")
            self._nvml_initialized = False

    def _shutdown_nvml(self) -> None:
        """Shutdown NVIDIA Management Library."""
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
            except Exception:
                pass

    def _get_gpu_metrics(self, gpu_index: int = 0) -> Optional[GPUMetrics]:
        """
        Get metrics for specified GPU.

        Args:
            gpu_index: GPU device index (default: 0)

        Returns:
            GPUMetrics or None if not available
        """
        if not self._nvml_initialized:
            return None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

            # Get GPU name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total_mb = mem_info.total / (1024 * 1024)
            memory_used_mb = mem_info.used / (1024 * 1024)
            memory_free_mb = mem_info.free / (1024 * 1024)
            memory_percent = (mem_info.used / mem_info.total) * 100

            # Get utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util.gpu
            except pynvml.NVMLError:
                gpu_utilization = 0.0

            # Get temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError:
                temperature = 0.0

            # Get power usage
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
            except pynvml.NVMLError:
                power = 0.0

            return GPUMetrics(
                gpu_index=gpu_index,
                name=name,
                memory_total_mb=round(memory_total_mb, 2),
                memory_used_mb=round(memory_used_mb, 2),
                memory_free_mb=round(memory_free_mb, 2),
                memory_percent=round(memory_percent, 1),
                gpu_utilization=round(gpu_utilization, 1),
                temperature_celsius=round(temperature, 1),
                power_usage_watts=round(power, 1),
                is_healthy=True,
                error_message=None,
            )

        except pynvml.NVMLError as e:
            error_msg = str(e)
            # Check for specific CUDA errors
            if "illegal memory access" in error_msg.lower():
                logger.error(f"GPU {gpu_index} CUDA error detected: {e}")
            else:
                logger.warning(f"GPU {gpu_index} monitoring error: {e}")

            return GPUMetrics(
                gpu_index=gpu_index,
                is_healthy=False,
                error_message=error_msg,
            )

        except Exception as e:
            logger.warning(f"Unexpected GPU monitoring error: {e}")
            return GPUMetrics(
                gpu_index=gpu_index,
                is_healthy=False,
                error_message=str(e),
            )

    def get_metrics(self) -> ResourceMetrics:
        """
        Get current resource metrics including GPU (synchronous).

        Returns:
            ResourceMetrics with current system state including GPU
        """
        # CPU percentage with short interval for accuracy
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(percpu=True)

        # Memory statistics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # System load average (1, 5, 15 minutes)
        load = psutil.getloadavg()

        # GPU metrics (if available)
        gpu_metrics = self._get_gpu_metrics(gpu_index=0)

        return ResourceMetrics(
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            swap_percent=swap.percent,
            load_average=load,
            timestamp=datetime.now(timezone.utc),
            gpu=gpu_metrics,
        )

    def can_start_training(self) -> tuple[bool, str]:
        """
        Check if system resources allow starting new training.

        Uses warning thresholds (not critical) to prevent training
        from pushing the system into critical state.
        Includes GPU resource checks.

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        metrics = self.get_metrics()

        # Check critical first (hard block)
        if metrics.cpu_percent > self.cpu_critical:
            return False, f"CPU critical: {metrics.cpu_percent:.1f}% (>{self.cpu_critical}%)"

        if metrics.memory_percent > self.memory_critical:
            return False, f"Memory critical: {metrics.memory_percent:.1f}% (>{self.memory_critical}%)"

        # GPU checks (if available)
        if metrics.gpu is not None:
            if not metrics.gpu.is_healthy:
                return False, f"GPU unhealthy: {metrics.gpu.error_message}"

            if metrics.gpu.memory_percent > self.gpu_memory_critical:
                return False, f"GPU memory critical: {metrics.gpu.memory_percent:.1f}% (>{self.gpu_memory_critical}%)"

            if metrics.gpu.temperature_celsius > self.gpu_temp_critical:
                return False, f"GPU temp critical: {metrics.gpu.temperature_celsius:.1f}°C (>{self.gpu_temp_critical}°C)"

            if metrics.gpu.memory_percent > self.gpu_memory_warning:
                return False, f"GPU memory high: {metrics.gpu.memory_percent:.1f}% (>{self.gpu_memory_warning}%)"

            if metrics.gpu.temperature_celsius > self.gpu_temp_warning:
                return False, f"GPU temp high: {metrics.gpu.temperature_celsius:.1f}°C (>{self.gpu_temp_warning}°C)"

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
        """Stop background monitoring loop and cleanup."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Shutdown NVML
        self._shutdown_nvml()
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
        Export current state for API response including GPU metrics.

        Returns:
            Dictionary with all resource metrics and thresholds
        """
        metrics = self._latest_metrics or self.get_metrics()
        can_train, reason = self.can_start_training()

        result = {
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
                "gpu_memory_warning": self.gpu_memory_warning,
                "gpu_memory_critical": self.gpu_memory_critical,
                "gpu_temp_warning": self.gpu_temp_warning,
                "gpu_temp_critical": self.gpu_temp_critical,
            },
            "status": self.get_status(),
            "can_start_training": can_train,
            "training_block_reason": reason if not can_train else None,
            "monitoring_active": self._running,
            "timestamp": metrics.timestamp.isoformat(),
        }

        # Add GPU metrics if available
        if metrics.gpu is not None:
            result["gpu"] = {
                "index": metrics.gpu.gpu_index,
                "name": metrics.gpu.name,
                "memory_total_mb": metrics.gpu.memory_total_mb,
                "memory_used_mb": metrics.gpu.memory_used_mb,
                "memory_free_mb": metrics.gpu.memory_free_mb,
                "memory_percent": metrics.gpu.memory_percent,
                "utilization_percent": metrics.gpu.gpu_utilization,
                "temperature_celsius": metrics.gpu.temperature_celsius,
                "power_usage_watts": metrics.gpu.power_usage_watts,
                "is_healthy": metrics.gpu.is_healthy,
                "error_message": metrics.gpu.error_message,
            }
        else:
            result["gpu"] = None
            result["gpu_monitoring_available"] = PYNVML_AVAILABLE

        return result


# Singleton instance with default thresholds
# Will be reconfigured from settings in main.py
resource_monitor = ResourceMonitor()
