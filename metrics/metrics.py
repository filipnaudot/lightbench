import time

import torch
import pynvml

from transformers import TextIteratorStreamer


class TTFT:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None)
        self.ttft = 0.00
    
    def measure_ttft(self, start_time):
        # IMPORTANT: streamers 'timeout' has to be None for this to work
        for _ in self.streamer:
            self.ttft = time.time() - start_time
            break


class VRAM:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        torch.cuda.reset_peak_memory_stats()

    def measure_vram(self):
        # Measure peak GPU memory usage (in GB)
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    

class PowerUsage:
    def __init__(self, gpu_index=0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.power_samples = []

    def measure_power(self):
        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert milliwatts to watts
        self.power_samples.append(power)
        return power

    def get_average(self):
        if self.power_samples:
            return sum(self.power_samples) / len(self.power_samples)
        return 0.0
    
    def kill(self):
        pynvml.nvmlShutdown()

    def __del__(self):
        pynvml.nvmlShutdown()