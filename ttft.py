
import time
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