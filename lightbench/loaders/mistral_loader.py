import gc
import os
import time

from dotenv import load_dotenv
from mistralai import Mistral

from lightbench.utils import Printer
from lightbench.loaders.loader import LLMServiceLoader
from lightbench.loaders.generation import Generation


class MistralLoader(LLMServiceLoader):
    def __init__(self, model_name: str):
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            Printer.print_red(
                "You need to specify your Mistral API key in a '.env' file in the root directory to use the MistralLoader.\n"
                "Make sure it is defined as: MISTRAL_API_KEY=your_key_here"
            )
            raise RuntimeError("Missing Mistral API key")
            
        self.mistral_client = Mistral(api_key=api_key)
        self.model_name = model_name

    def generate(self,
                 prompt,
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 safe_prompt: bool = False) -> Generation:
        start_time = time.perf_counter()
        response = self.mistral_client.chat.complete(
            model = self.model_name,
            messages = prompt,
            max_tokens = max_tokens,
            temperature = temperature,
            safe_prompt = safe_prompt
        )
        return Generation(response=response.choices[0].message.content,
                          inference_time=(time.perf_counter() - start_time))

    def is_local(slef): return False

    def name(self): return self.model_name

    def cleanup(self):
        del self.mistral_client
        self.mistral_client = None
        del self.model_name
        self.model_name = None
        gc.collect()