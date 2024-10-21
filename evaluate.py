import os
import time
import json

from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from code_evaluator import CodeEvaluator



def load_env_variables():
    load_dotenv()
    
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    model_id = os.getenv("MODEL_NAME")
    
    return model_id, hf_token


def main(stream: bool = False, QUANTIZE: bool = False):

    model_id, hf_token = load_env_variables()
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    start_test_line = 11
    end_test_line = 510
    with open('./data/mbpp/mbpp.jsonl', 'r') as json_file:
        json_list = list(json_file)[start_test_line-1:end_test_line]

    system_command = {
        "role": "system",
        "content": "You are a Python programming assistant. Your task is to write Python functions according to the user's prompt. Respond only with the necessary Python code, including python package imports if needed. Do not provide example usage, only the python function.",
    }
    prompts = []
    for json_str in json_list:
        result = json.loads(json_str)
        promt = (
            [
                system_command,
                {
                    "role": "user",
                    "content": result["text"] + f' The function should pass the following test: {result["test_list"][0]}.',
                }
            ], result["test_list"][1]
        )
        prompts.append(promt)
    
    code_evaluator = CodeEvaluator(model_id=model_id, hf_token=hf_token, quantize=False, few_shot=False, verbose=False)
    code_evaluator.run(prompts)

    code_evaluator.print_summary()



if __name__ == "__main__":
    main(stream=True, QUANTIZE=False)
