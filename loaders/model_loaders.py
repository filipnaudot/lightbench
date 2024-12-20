import gc

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Pipeline,
    pipeline,
)
import torch


class LLamaModelLoader:
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


    def cleanup(self):        
        del self.generator
        self.generator = None
        del self.tokenizer
        self.tokenizer = None
        del self.model
        self.model = None
        gc.collect()

        torch.cuda.empty_cache()