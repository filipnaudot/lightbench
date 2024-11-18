import time

import torch
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