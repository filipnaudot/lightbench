import time

import torch
import pynvml
from pynvml import NVMLError, NVMLError_NotSupported

from transformers import TextIteratorStreamer


class TTFT:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None)
        self.ttft = 0.00
    
    def measure_ttft(self, start_time):
        # IMPORTANT: streamers 'timeout' has to be None for this to work
        for _ in self.streamer:
            self.ttft = time.perf_counter() - start_time
            break


class VRAM_NVML:
    def __init__(self) -> None:
        pynvml.nvmlInit()
        self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._max_memory = 0
        self.reset()

    def reset(self):
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.device_handle)
        self._max_memory = mem_info.used

    def measure_vram(self):
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.device_handle)
        if mem_info.used > self._max_memory:
            self._max_memory = mem_info.used
        return self._max_memory / (1024 ** 3)


class VRAM_TORCH:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        torch.cuda.reset_peak_memory_stats()

    def measure_vram(self):
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
  

class PowerUsage:
    def __init__(self, gpu_index=0, DEBUG:bool = False):
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
        if self.power_samples:
            return sum(self.power_samples) / len(self.power_samples)
        return 0.0
    
    def kill(self):
        pynvml.nvmlShutdown()

    def __del__(self):
        pynvml.nvmlShutdown()