import gc
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from lightbench.utils import Printer
from lightbench.loaders.loader import LLMServiceLoader
from lightbench.loaders.generation import Generation

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    Printer.print_red(
        "You need to specify your DeepSeek API key in a '.env' file in the root directory.\n"
        "Make sure it is defined as: DEEPSEEK_API_KEY=your_key_here"
    )
    exit(1)


class DeepSeekLoader(LLMServiceLoader):
    def __init__(self, model_name: str):
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1"
        )
        self.model_name = model_name

    def generate(self, prompt, max_tokens: int = 512) -> Generation:
        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            max_tokens=max_tokens,
        )

        return Generation(
            response=response.choices[0].message.content,
            inferece_time=(time.perf_counter() - start_time)
        )

    def is_local(self): return False

    def name(self): return self.model_name

    def cleanup(self):
        del self.client
        self.client = None
        del self.model_name
        self.model_name = None
        gc.collect()
