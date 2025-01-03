import time
import json
import argparse

from configurators.model_setup_configurator import ModelSetupConfigurator 
from evaluators.code_evaluator import CodeEvaluator
from evaluators.text_evaluator import TextEvaluator
from metrics.llm_judge import LLMJudge





####################################################################
#                          Code Generation                         #
####################################################################
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


def evaluate_code():
    start_test_line = 1
    end_test_line = 450
    with open('./data/mbpp/mbpp.jsonl', 'r') as json_file:
        json_list = list(json_file)[start_test_line-1:end_test_line]

    system_command = {
        "role": "system",
        "content": "You are a Python programming assistant. Your task is to write Python functions according to the user's prompt. Respond only with the necessary Python code, including python package imports if needed. Do not provide example usage, only the python function.",
    }
    
    prompts = create_coding_prompts(json_list, system_command)

    test_configurator = ModelSetupConfigurator()
    models = test_configurator.generate_list(use_quantization=True, use_few_shot=True)
    for model, quantize, few_shot in models:
        print(f"\n---------- {model} ----------\n    quantize: {str(quantize)}\n    few-shot: {str(few_shot)}\n")
        
        code_evaluator = CodeEvaluator(model, quantize=quantize, few_shot=few_shot, verbose=False)
        code_evaluator.run(prompts)
        code_evaluator.print_summary()
        code_evaluator.cleanup()
        time.sleep(3)



####################################################################
#                        Question Answering                        #
####################################################################
def create_qa_prompts(json_list, system_command):
    prompts = []
    for json_str in json_list:
        result = json.loads(json_str)
        question = result["question"]
        context = "".join(["".join(sentences) for para in result["context"] for sentences in para[1]])
        answer = None # result["answer"]
        
        prompt = (
            [
                system_command,
                {
                    "role": "user",
                    "content": f'Answer the question based on the following context: "{context}". The question is: "{question}"',
                }
            ],
            answer
        )
        prompts.append(prompt)
    
    return prompts


def evaluate_text():
    start_test_line = 1
    end_test_line = 100
    
    with open('./data/hotpotqa/hotpot_test_fullwiki_v1-first-500.jsonl', 'r') as json_file:
        json_list = list(json_file)[start_test_line-1:end_test_line]
    
    system_command = {
        "role": "system",
        "content": "You are a question-answering assistant. Answer the user's question based on the provided context. Respond with only the answer in a single sentence.",
    }
    
    prompts = create_qa_prompts(json_list, system_command)
    
    judge = LLMJudge()

    model_setup_conf = ModelSetupConfigurator()
    models = model_setup_conf.generate_list(use_quantization=True, use_few_shot=False)
    for model, quantize, _ in models:
        print(f"\n---------- {model} ----------\n    quantize: {str(quantize)}\n")
        text_evaluator = TextEvaluator(model, judge, quantize=quantize, verbose=False)
        text_evaluator.run(prompts)
        text_evaluator.print_summary()
        text_evaluator.cleanup()
        time.sleep(3)



####################################################################
#                                Main                              #
####################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation script with options to run code, text, or all evaluations.",
        usage="python %(prog)s [--code] [--text] [--all]",
        epilog="Examples:\n  python evaluate.py --code\n  python evaluate.py --all\n  python evaluate.py --text",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--code', action='store_true', help="Run code evaluation")
    parser.add_argument('--text', action='store_true', help="Run text evaluation")
    parser.add_argument('--all', action='store_true', help="Run all evaluations")

    return parser.parse_args()


def main():

    args = parse_args()
    if args.code:
        evaluate_code()
    elif args.all:
        evaluate_code()
        evaluate_text()
    elif args.text:
        evaluate_text()
    else:
        print("Please provide an argument. Use --code, --all, or --text. Use --help for more information.")


if __name__ == "__main__":
    main()