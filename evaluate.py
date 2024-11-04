import os
import time
import json
import argparse
from dotenv import load_dotenv

from code_evaluator import CodeEvaluator
from text_evaluator import TextEvaluator



#                 MODEL NAME                   QUANT  FEW-SHOT
MODELS = [("meta-llama/Llama-3.2-1B-Instruct", False, False),
          ("meta-llama/Llama-3.2-1B-Instruct", False, True),
          ("meta-llama/Llama-3.2-1B-Instruct", True,  False),
          ("meta-llama/Llama-3.2-1B-Instruct", True,  True),
          ("meta-llama/Llama-3.2-3B-Instruct", False, False),
          ("meta-llama/Llama-3.2-3B-Instruct", False, True),
          ("meta-llama/Llama-3.2-3B-Instruct", True,  False),
          ("meta-llama/Llama-3.2-3B-Instruct", True,  True),
          # TODO: Check if model can run on GPU instead of manual remove
          ("meta-llama/Llama-3.1-8B-Instruct", False, False),
          ("meta-llama/Llama-3.1-8B-Instruct", False, True),
          ("meta-llama/Llama-3.1-8B-Instruct", True,  False),
          ("meta-llama/Llama-3.1-8B-Instruct", True,  True),
          ]




def create_coding_prompts(json_list, system_command):
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


def evaluate_code(hf_token):
    start_test_line = 1
    end_test_line = 450
    with open('./data/mbpp/mbpp.jsonl', 'r') as json_file:
        json_list = list(json_file)[start_test_line-1:end_test_line]

    system_command = {
        "role": "system",
        "content": "You are a Python programming assistant. Your task is to write Python functions according to the user's prompt. Respond only with the necessary Python code, including python package imports if needed. Do not provide example usage, only the python function.",
    }
    
    prompts = create_coding_prompts(json_list, system_command)

    for model, quantize, few_shot in MODELS:
        print(f"\n---------- {model} ----------\n    quantize: {str(quantize)}\n    few-shot: {str(few_shot)}\n")
        
        code_evaluator = CodeEvaluator(model, hf_token, quantize=quantize, few_shot=few_shot, verbose=False)
        code_evaluator.run(prompts)

        code_evaluator.print_summary()
        code_evaluator.cleanup()
        time.sleep(3)


def evaluate_text(hf_token, openai_api_key):
    """ NOT IMPLEMENTED.
    """
    text_evaluator = TextEvaluator(hf_token, openai_api_key, "NONE")
    text_evaluator.run([])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation script with options to run code, text, or all evaluations.",
        usage="python %(prog)s [--code] [--text] [--all]",
        epilog="Examples:\n  python evaluate.py --code\n  python evaluate.py --all\n  python evaluate.py --text",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Define the arguments
    parser.add_argument('--code', action='store_true', help="Run code evaluation")
    parser.add_argument('--text', action='store_true', help="Run text evaluation")
    parser.add_argument('--all', action='store_true', help="Run all evaluations")

    return parser.parse_args()


def main():

    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    args = parse_args()

    if args.code:
        evaluate_code(hf_token)
    elif args.all:
        evaluate_code(hf_token)
        evaluate_text(hf_token, openai_api_key)
    elif args.text:
        evaluate_text(hf_token, openai_api_key)
    else:
        print("Please provide an argument. Use --code, --all, or --text. Use --help for more information.")


if __name__ == "__main__":
    main()