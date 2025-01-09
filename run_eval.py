import time
import json
import argparse

from configurators.model_setup_configurator import ModelSetupConfigurator 
from evaluators.code_evaluator import CodeEvaluator
from evaluators.text_evaluator import TextEvaluator
from evaluators.cv_bias_evaluator import CVBiasEvaluator

from metrics.llm_judge import LLMJudge





####################################################################
#                          Code Generation                         #
####################################################################
def evaluate_code():
    test_configurator = ModelSetupConfigurator()
    models = test_configurator.generate_list(use_quantization=True, use_few_shot=True)
    for model, quantize, few_shot in models:
        print(f"\n---------- {model} ----------\n    quantize: {str(quantize)}\n    few-shot: {str(few_shot)}\n")
        
        code_evaluator = CodeEvaluator(model, quantize=quantize, few_shot=few_shot, verbose=False)
        code_evaluator.run()
        code_evaluator.print_summary()
        code_evaluator.cleanup()
        time.sleep(3)



####################################################################
#                        Question Answering                        #
####################################################################
def evaluate_text():    
    judge = LLMJudge()
    model_setup_conf = ModelSetupConfigurator()
    models = model_setup_conf.generate_list(use_quantization=True, use_few_shot=False)
    for model, quantize, _ in models:
        print(f"\n---------- {model} ----------\n    quantize: {str(quantize)}\n")
        text_evaluator = TextEvaluator(model, judge, quantize=quantize, verbose=False)
        text_evaluator.run()
        text_evaluator.print_summary()
        text_evaluator.cleanup()
        time.sleep(3)



####################################################################
#                               Bias                               #
####################################################################
def evaluate_bias():
    model_setup_conf = ModelSetupConfigurator()
    models = model_setup_conf.generate_list(use_quantization=True, use_few_shot=False)
    for model, quantize, _ in models:
        print(f"\n---------- {model} ----------\n    quantize: {str(quantize)}\n")
        bias_evaluator = CVBiasEvaluator(model,
                                         quantize=quantize,
                                         dataset_path="./data/CVerse/CVerse.json",
                                         verbose=True)
        bias_evaluator.run()
        bias_evaluator.print_summary()
        bias_evaluator.cleanup()



####################################################################
#                                Main                              #
####################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation script with options to run different evaluations.",
        usage="python %(prog)s [--code] [--text] [--bias]",
        epilog="Examples:\n  python evaluate.py --code\n  python evaluate.py --text\n",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--code', action='store_true', help="Run code evaluation")
    parser.add_argument('--text', action='store_true', help="Run text evaluation")
    parser.add_argument('--bias', action='store_true', help="Run text evaluation")

    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.code:
        evaluate_code()
    elif args.text:
        evaluate_text()
    elif args.bias:
        evaluate_bias()
    else:
        print("Please provide an argument. Use --help for more information.")


if __name__ == "__main__":
    main()