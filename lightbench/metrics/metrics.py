"""
Module for measuring time to first token (TTFT), GPU memory usage, and GPU power usage.

This module provides the following classes:
    - TTFT: Measures the time to first token during text generation.
    - VRAM_NVML: Monitors GPU VRAM usage using NVIDIA’s NVML library.
    - VRAM_TORCH: Measures GPU VRAM usage using PyTorch CUDA utilities.
    - PowerUsage: Measures and tracks GPU power usage via NVML.

Dependencies:
    - time: For performance timing.
    - torch: For GPU memory measurements.
    - pynvml: For interfacing with NVIDIA Management Library.
    - transformers: For text streaming with transformer models.
"""

import time

import torch
import pynvml
from pynvml import NVMLError, NVMLError_NotSupported

from transformers import TextIteratorStreamer


class TTFT:
    """
    Class for measuring Time To First Token (TTFT) during text generation.

    Attributes:
        tokenizer: A tokenizer instance used for encoding text.
        streamer: A TextIteratorStreamer instance that streams text output.
        ttft: Float representing the time (in seconds) from start to first token.
    """

    def __init__(self, tokenizer) -> None:
        """
        Initialize the TTFT instance with a tokenizer.

        Args:
            tokenizer: A tokenizer instance from a transformer model.
        """
        self.tokenizer = tokenizer
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None)
        self.ttft = 0.00
    
    def measure_ttft(self, start_time):
        """
        Measure the time to first token (TTFT) for a text generation stream.

        This function iterates over the text streamer and sets the TTFT value
        based on the elapsed time since the provided start_time.

        Args:
            start_time: The starting time (from time.perf_counter()) when generation began.
        """
        # IMPORTANT: streamers 'timeout' has to be None for this to work
        for _ in self.streamer:
            self.ttft = time.perf_counter() - start_time
            break


class VRAM_NVML:
    """
    Class to monitor GPU VRAM usage using NVIDIA's NVML.

    Attributes:
        device_handle: NVML handle for the first GPU device.
        _max_memory: Tracks the maximum memory used (in bytes).
    """
    def __init__(self) -> None:
        """
        Initialize NVML and retrieve a handle for the first GPU device.
        """
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
    Class to measure GPU VRAM usage using PyTorch's CUDA utilities.

    Attributes:
        DEBUG (bool): Flag to enable debug output.
        device (torch.device): The device to monitor ('cuda' or 'mps').
    """
    device: str = "cuda"

    def __init__(self, device: str, DEBUG:bool = False) -> None:
        """
        Initialize the VRAM_TORCH instance by resetting CUDA memory stats.
        """
        self.DEBUG = DEBUG
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        """
        Reset memory usage statistics based on the device type.
        """
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            if self.DEBUG: print(f"Reset peak memory stats on {self.device}")
        elif self.device == "mps":
            pass # MPS backend does not support memory stat reset.
        else:
            if self.DEBUG: print(f"No memory stats to reset for device: {self.device}")

    def measure_vram(self):
        """
        Measure the memory usage in gigabytes.

        Returns:
            float: Memory usage (in GB), either peak or current depending on backend.
        """
        if self.device == "mps":
            # TODO: We need to be able to measure max here, similar to VRAM_NVML.
            return torch.mps.current_allocated_memory() / (1024 ** 3)
        elif self.device == "cuda":
            return torch.cuda.max_memory_allocated(device=self.device) / (1024 ** 3)
        else:
            if self.DEBUG: print(f"Unknown device: {self.device}")
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
  

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