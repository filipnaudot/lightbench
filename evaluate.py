import os
import time
import json


from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from code_evaluator import CodeEvaluator




def create_prompts(json_list, system_command):
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
    
    return prompts


def main():

    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    models = ["meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]


    start_test_line = 11
    end_test_line = 510
    with open('./data/mbpp/mbpp.jsonl', 'r') as json_file:
        json_list = list(json_file)[start_test_line-1:end_test_line]

    system_command = {
        "role": "system",
        "content": "You are a Python programming assistant. Your task is to write Python functions according to the user's prompt. Respond only with the necessary Python code, including python package imports if needed. Do not provide example usage, only the python function.",
    }
    
    prompts = create_prompts(json_list, system_command)
    
    #                 MODEL NAME                   QUANT  FEW-SHOT
    models = [("meta-llama/Llama-3.2-1B-Instruct", False, False),
              ("meta-llama/Llama-3.2-1B-Instruct", False, True),
              ("meta-llama/Llama-3.2-1B-Instruct", True,  False),
              ("meta-llama/Llama-3.2-1B-Instruct", True,  True),
              
              ("meta-llama/Llama-3.2-3B-Instruct", False, False),
              ("meta-llama/Llama-3.2-3B-Instruct", False, True),
              ("meta-llama/Llama-3.2-3B-Instruct", True,  False),
              ("meta-llama/Llama-3.2-3B-Instruct", True,  True),
              
              # TODO: Check if model can run on GPU instead of manual remove
              # ("meta-llama/Llama-3.1-8B-Instruct", False, False),
              # ("meta-llama/Llama-3.1-8B-Instruct", False, True),
              ("meta-llama/Llama-3.1-8B-Instruct", True,  False),
              ("meta-llama/Llama-3.1-8B-Instruct", True,  True),
              ]


    for model, quantize, few_shot in models:
        print(f"\n---------- {model} ----------\n    quantize: {str(quantize)}\n    few-shot: {str(few_shot)}\n")
        
        code_evaluator = CodeEvaluator(model, hf_token, quantize=quantize, few_shot=few_shot, verbose=False)
        code_evaluator.run(prompts)

        code_evaluator.print_summary()
        code_evaluator.cleanup()
        time.sleep(3)



if __name__ == "__main__":
    main()
