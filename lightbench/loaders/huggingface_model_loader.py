import os
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

from lightbench.loaders.loader import LLMServiceLoader
from lightbench.loaders.generation import Generation
from lightbench.metrics.metrics import GenerationMetrics

from dotenv import load_dotenv
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


class HFModelLoader(LLMServiceLoader):
    def __init__(self, model_name: str, quantize: bool = False):
        self.model_name = model_name
        self.quantize = quantize
        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
        self.device:str = "cpu"
        if torch.cuda.is_available():         self.device = "cuda"
        if torch.backends.mps.is_available(): self.device = "mps"

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
            token=HUGGINGFACE_TOKEN,
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
        metric_streamer = GenerationMetrics(tokenizer=self.tokenizer, sample_every=5, device="cuda")
        metric_streamer.set_start_time()

        start_time = time.perf_counter()
        generation = self.generator(
            prompt,
            streamer=metric_streamer,
            do_sample=False,
            temperature=1.0,
            top_p=1,
            max_new_tokens=max_tokens,
        )

        return Generation(response          = generation[0]['generated_text'][-1]['content'],
                          inference_time    = (time.perf_counter() - start_time),
                          ttft              = metric_streamer.ttft,
                          peak_memory_usage = metric_streamer.avg_vram,
                          avg_power_usage   = metric_streamer.avg_power)


    def is_quantized(self): return self.quantize

    def is_local(slef): return True

    def name(self): return self.model_name

    def cleanup(self):        
        del self.generator
        self.generator = None
        del self.tokenizer
        self.tokenizer = None
        del self.model
        self.model = None
        gc.collect()

        if self.device == "cuda": torch.cuda.empty_cache()
        if self.device == "mps": torch.mps.empty_cache()