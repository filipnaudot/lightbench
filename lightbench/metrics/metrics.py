"""
Module for measuring time to first token (TTFT), GPU memory usage, and GPU power usage.

This module provides the following classes:
    - GenerationMetrics: Monitors TTFT, GPU VRAM, and power usage.
    - VRAM_TORCH: Measures GPU VRAM usage using PyTorch CUDA utilities.
    - PowerUsage: Measures and tracks GPU power usage via NVML.
"""

import time
from typing import List, Optional

import torch

import pynvml
from pynvml import NVMLError, NVMLError_NotSupported

from transformers.generation.streamers import BaseStreamer




class GenerationMetrics(BaseStreamer):
    """Collects several generation-time metrics in a single place.

    It measures:
        • **TTFT** (Time-To-First-Token)
        • **Average VRAM usage** (either via NVML or PyTorch utilities)
        • **Average GPU power consumption** (via NVML)

    Sampling happens *during* token streaming.  Set ``sample_every`` to decide how often the
    measurements are taken, defaults to ``sample_every = 5``.

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizerBase
        Tokenizer used by your model; required to build a ``TextIteratorStreamer``.
    sample_every : int, default = 5
        Frequency (in tokens) at which VRAM & power are sampled.  Must be > 0.
    device : str, default = "cuda"
        Device string, NVIDIA ``torch``and Apple ``Metal (mps)``are supported.
    use_nvml : bool, default = False
        If ``True`` we try to use NVML for VRAM.  If NVML is unavailable or
        ``use_nvml=False`` we fall back to the PyTorch memory utilities.
    DEBUG : bool, default = False
        Emit verbose messages when something goes wrong.
    """

    def __init__(
        self,
        tokenizer,
        sample_every: int = 1,
        device: str = "cuda",
        DEBUG: bool = False,
    ) -> None:
        if sample_every < 1: raise ValueError("sample_every must be >= 1")
        self.sample_every = sample_every
        self._token_count = 0
        self._vram_samples: List[float] = []
        self._power_samples: List[float] = []
        self._start_time: Optional[float] = None
        self._ttft: Optional[float] = None
        # Tokenizer
        self.tokenizer = tokenizer
        # VRAM
        self.vram_monitor  = VRAM_TORCH(device)
        # Power
        self.power_monitor = PowerUsage()

    @property
    def ttft(self) -> Optional[float]: return self._ttft

    @property
    def avg_vram(self) -> float: return sum(self._vram_samples)/len(self._vram_samples) if self._vram_samples else 0.0

    @property
    def avg_power(self) -> float: return sum(self._power_samples)/len(self._power_samples) if self._power_samples else 0.0

    def put(self, value):
        self._token_count += 1
        if self._start_time is None: raise RuntimeError("Call .set_start_time() before generate()")
        if self._token_count == 1: self._ttft = time.perf_counter() - self._start_time

        if self._token_count % self.sample_every == 0:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            elif torch.backends.mps.is_available(): torch.mps.synchronize()

            self._vram_samples.append(self.vram_monitor.measure_vram())
            self._power_samples.append(self.power_monitor.measure_power())

    def end(self):
        if self._token_count % self.sample_every:
            self._vram_samples.append(self.vram_monitor.measure_vram())
            self._power_samples.append(self.power_monitor.measure_power())
        self.power_monitor.kill()

    def set_start_time(self): self._start_time = time.perf_counter()

    def reset(self):
        self._token_count = 0
        self._vram_samples.clear()
        self._power_samples.clear()
        self._ttft = None
        self.vram_monitor.reset()
        self.power_monitor.power_samples = []


import warnings
# Deprecated
class VRAM_NVML:
    """
    Class to monitor GPU VRAM usage using NVIDIA's NVML.

    .. deprecated:: 0.1.0
       The VRAM_NVML class is deprecated. Use VRAM_TORCH instead.

    Attributes:
        device_handle: NVML handle for the first GPU device.
        _max_memory: Tracks the maximum memory used (in bytes).
    """
    def __init__(self) -> None:
        """
        Initialize NVML and retrieve a handle for the first GPU device.
        """
        warnings.warn(
            "VRAM_NVML is deprecated. Use VRAM_TORCH instead",
            category=DeprecationWarning,
            stacklevel=2
        )
        pynvml.nvmlInit()
        self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._max_memory = 0
        self.reset()

    def reset(self):
        """
        Reset the maximum memory usage by reading the current used memory.
        """
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.device_handle)
        self._max_memory = mem_info.used

    def measure_vram(self):
        """
        Measure and return the peak VRAM usage in gigabytes (GB).

        Returns:
            float: Maximum VRAM usage (in GB) observed so far.
        """
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.device_handle)
        if mem_info.used > self._max_memory:
            self._max_memory = mem_info.used
        return self._max_memory / (1024 ** 3)


class VRAM_TORCH:
    """
    Class to measure GPU VRAM usage using PyTorch's utilities.

    Attributes:
        DEBUG (bool): Flag to enable debug output.
        device (torch.device): The device to monitor ('cuda' or 'mps').
    """
    device: str = "cuda"

    def __init__(self, device: str, DEBUG:bool = False) -> None:
        """
        Initialize the VRAM_TORCH instance by resetting memory stats.
        """
        self.DEBUG = DEBUG
        self.device = torch.device(device)
        self.device_type = self.device.type
        self.reset()

    def reset(self):
        """
        Reset memory usage statistics based on the device type.
        """
        if self.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            if self.DEBUG: print(f"Reset peak memory stats on {self.device}")
        elif self.device_type == "mps":
            pass # MPS backend does not support memory stat reset.
        else:
            if self.DEBUG: print(f"No memory stats to reset for device: {self.device}")

    def measure_vram(self) -> float:
        """
        Measure the memory usage in gigabytes.

        Returns:
            float: Memory usage (in GB), either peak or current depending on backend.
        """
        if self.device_type == "cuda": return torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
        if self.device_type == "mps": return torch.mps.current_allocated_memory() / (1024 ** 3)
        if self.DEBUG: print(f"[VRAM_TORCH] unknown backend '{self.device_type}', returning 0.0")
        return 0.0
  

class PowerUsage:
    """
    Class to measure and track GPU power usage using NVML.

    Attributes:
        DEBUG (bool): Flag to enable debug output.
        handle: NVML handle for the specified GPU.
        power_samples (list): List to store power usage measurements in watts.
    """
    def __init__(self, gpu_index=0, DEBUG:bool = False):
        """
        Initialize the PowerUsage instance.

        Attempts to initialize NVML and obtain a handle for the specified GPU index.
        If initialization fails, the handle is set to None.

        Args:
            gpu_index (int): Index of the GPU to monitor (default is 0).
            DEBUG (bool): Enable debug mode for verbose output (default is False).
        """
        self.DEBUG = DEBUG

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except NVMLError as e:
            if self.DEBUG: print("Failed to initialize NVML: ", e)
            self.handle = None

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.power_samples = []

    def measure_power(self):
        """
        Measure the current GPU power usage and record the sample.

        Returns:
            float: Current GPU power usage in watts. Returns 0 if measurement is unsupported or fails.
        """
        if self.handle:
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW to W
                self.power_samples.append(power)
            except NVMLError_NotSupported:
                if self.DEBUG: print("Power usage measurement not supported on this GPU.")
                return 0
            except NVMLError as e:
                if self.DEBUG: print("NVML error encountered: ", e)
                return 0
        return power

    def get_average(self):
        """
        Calculate and return the average GPU power usage from the recorded samples.

        Returns:
            float: Average power usage in watts. Returns 0.0 if no samples exist.
        """
        if self.power_samples:
            return sum(self.power_samples) / len(self.power_samples)
        return 0.0
    
    def kill(self):
        """
        Shutdown NVML to clean up resources.
        """
        pynvml.nvmlShutdown()

    def __del__(self):
        """
        Destructor to ensure NVML is shutdown properly.
        """
        pynvml.nvmlShutdown()