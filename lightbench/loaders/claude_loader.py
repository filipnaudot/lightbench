import gc
import os
import time

from dotenv import load_dotenv
import anthropic

from lightbench.utils import Printer
from lightbench.loaders.loader import LLMServiceLoader
from lightbench.loaders.generation import Generation

load_dotenv()
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not CLAUDE_API_KEY:
    Printer.print_red(
        "You need to specify your Claude API key in a '.env' file in the root directory.\n"
        "Make sure it is defined as: ANTHROPIC_API_KEY=your_key_here"
    )
    exit(1)


class ClaudeLoader(LLMServiceLoader):
    def __init__(self, model_name: str):
        self.client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        self.model_name = model_name

    def generate(self, prompt, max_tokens: int = 512) -> Generation:
        start_time = time.perf_counter()

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=prompt,
            system="",  # optional: system message here
        )

        return Generation(
            response=response.content[0].text,
            inference_time=(time.perf_counter() - start_time)
        )

    def is_local(self): return False

    def name(self): return self.model_name

    def cleanup(self):
        del self.client
        self.client = None
        del self.model_name
        self.model_name = None
        gc.collect()
