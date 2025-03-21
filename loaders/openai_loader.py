import gc
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from utils import Printer
from loaders.loader import LLMServiceLoader
from loaders.generation import Generation

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    Printer.print_red(
        "You need to specify your OpenAI API key in a '.env' file in the root directory to use the OpenAILoader.\n"
        "Make sure it is defined as: OPENAI_API_KEY=your_key_here"
    )
    exit(1) 


class OpenAILoader(LLMServiceLoader):
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name


    def generate(self, prompt, max_tokens: int = 512) -> Generation:
        start_time = time.perf_counter()
        response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=max_tokens,
            )
        return Generation(response=response.choices[0].message.content,
                          inferece_time=(time.perf_counter() - start_time))


    def is_local(slef): return False

    def name(self): return self.model_name

    def cleanup(self):
        del self.client
        self.client = None
        del self.model_name
        self.model_name = None
        gc.collect()