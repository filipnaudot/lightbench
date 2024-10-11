import os
import time
from queue import Queu
import threading

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer


class CodeEvaluator:
    def __init__(self, model_id, hf_token, quantize=False):
        self.quantize = quantize
        self.model = model_id
        self.hf_token = hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.quantize:
            self.model = self.load_quantized_model(model_id)

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id,
            token=self.hf_token,
        )


    def load_quantized_model(model_id):
        print("Loading quantized model...")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        return model


    def preprocess_data(data):
        if f"```python" in data:
            data = data[data.find(f"```python") + len(f"```python"):]
            data = data[:data.find("```")]
        return data