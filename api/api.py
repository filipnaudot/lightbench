import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
from dotenv import load_dotenv
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from loaders.model_loaders import LLamaModelLoader
from utils import Printer


load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
model_id = os.getenv("MODEL_NAME")


app = FastAPI()
model_loader = LLamaModelLoader(model_id, quantize=True, hf_token=hf_token)


class GenerateRequest(BaseModel):
    prompt: list
    max_length: int = 512

@app.post("/generate/")
async def generate_text(request: GenerateRequest):
    Printer.print_yellow(f"RECEIVED", end="") ; print(f": {request.prompt}")
    generation = model_loader.generator(
            request.prompt,
            do_sample=True,
            temperature=0.6,
            top_p=0.6,
            max_new_tokens=request.max_length,
        )
    response = generation[0]['generated_text'][-1]['content']
    Printer.print_yellow(f"RESPONSE", end="") ; print(f": {response}\n")

    return {"generation": generation, "response": response}