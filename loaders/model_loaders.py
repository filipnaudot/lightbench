import gc
import time
import threading

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Pipeline,
    pipeline,
)
import torch

from loaders.loader import LLMServiceLoader
from metrics.metrics import TTFT, VRAM, PowerUsage


class LLamaModelLoader(LLMServiceLoader):
    def __init__(self, model_name: str, quantize: bool = False, hf_token: str = None):
        self.model_name = model_name
        self.quantize = quantize
        self.hf_token = hf_token
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.device:str = "cuda" if torch.cuda.is_available() else "cpu"

        if self.quantize:
            self.model = self._load_quantized_model()
        else:
            self.model = model_name

        self.generator: Pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id,
            token=self.hf_token,
        )


    def _load_quantized_model(self):
        print("Loading quantized model...")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        return model
    

    def generate(self, prompt, max_tokens: int = 512):
        vram_handler = VRAM()
        ttft_handler = TTFT(self.tokenizer)
        power_handler = PowerUsage()
        power_handler.measure_power()

        start_time = time.time()
        streaming_thread = threading.Thread(target=ttft_handler.measure_ttft, args=(start_time,))
        streaming_thread.start()
        generation = self.generator(
            prompt,
            streamer=ttft_handler.streamer,
            do_sample=False,
            temperature=1.0,
            top_p=1,
            max_new_tokens=max_tokens,
        )
        end_time = time.time()
        streaming_thread.join()
        power_handler.measure_power()
        peak_memory_usage = vram_handler.measure_vram()
        power_handler.kill()

        response = generation[0]['generated_text'][-1]['content']
        return response, (end_time-start_time), ttft_handler.ttft, peak_memory_usage, power_handler.get_average()


    def cleanup(self):        
        del self.generator
        self.generator = None
        del self.tokenizer
        self.tokenizer = None
        del self.model
        self.model = None
        gc.collect()

        torch.cuda.empty_cache()