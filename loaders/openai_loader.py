import gc
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



class OpenAILoader:
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name


    def generate(self, prompt, max_tokens: int = 520):
        response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=max_tokens,
            )

        return response.choices[0].message.content


    def cleanup(self):
        del self.client
        self.client = None
        del self.model_name
        self.model_name = None
        gc.collect()