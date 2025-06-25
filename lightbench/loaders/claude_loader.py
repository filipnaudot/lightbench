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

    def _separate_system_message(self, prompt):
        system_message = ""
        new_prompt = []
        for msg in prompt:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                new_prompt.append(msg)
        return new_prompt, system_message

    def generate(self, prompt, max_tokens: int = 512) -> Generation:
        start_time = time.perf_counter()

        # Separate system message from prompt
        prompt, system_message = self._separate_system_message(prompt)

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=prompt,
            system=system_message,  #  system message handled correctly
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
