import os
import threading
import time
import gc
import json

import torch

from utils import Printer
from evaluator import Evaluator
from llm_judge import LLMJudge
from metrics import TTFT, VRAM, PowerUsage
from model_loaders import LLamaModelLoader


class TextEvaluator(Evaluator):
    def __init__(self, model_name:str, hf_token:str, judge:LLMJudge, quantize:bool = False, verbose:bool = False):
        super().__init__(verbose)

        self.quantize:bool = quantize
        self.model_name:str = model_name
        self.hf_token:str = hf_token
        self.judge:LLMJudge = judge

        self.num_test:int = 0
        self.inference_time_list:list[float] = []
        self.ttft_list:list[float] = []
        self.memory_usage_list:list[float] = []
        self.power_usage_list:list[float] = []
        self.llm_judge_score_list:list[int] = []

        self.model_loader = LLamaModelLoader(model_name, quantize, hf_token)


    def _clear_last_row(self):
        print(f"\r{' ' * 40}\r", end='', flush=True) # Clear last line
    
    def _print_test_metrics(self, index, inference_time, ttft, memory_usage, power_usage):
        self._clear_last_row()
        print(
            f"\r{index+1} "
            f"({inference_time:.2f}s, "
            f"TTFT: {ttft:.2f}s, "
            f"Memory: {memory_usage:.2f} GB, "
            f"Power: {int(power_usage)}W) ",
            end=''
            )


    def _print_test_status(self):
        if self.num_test > 0: self._clear_last_row()
        avg_score = 0
        if len(self.llm_judge_score_list) > 0: avg_score = (sum(self.llm_judge_score_list) / len(self.llm_judge_score_list))
        Printer.print_cyan(f"\rAverage score: {avg_score:.2f}", end='')


    def _generate_response(self, prompt):
        vram_handler = VRAM()
        ttft_handler = TTFT(self.model_loader.tokenizer)
        power_handler = PowerUsage()
        power_handler.measure_power()

        start_time = time.time()
        streaming_thread = threading.Thread(target=ttft_handler.measure_ttft, args=(start_time,))
        streaming_thread.start()
        generation = self.model_loader.generator(
            prompt,
            streamer=ttft_handler.streamer,
            do_sample=False,
            temperature=1.0,
            top_p=1,
            max_new_tokens=512,
        )
        end_time = time.time()
        streaming_thread.join()
        power_handler.measure_power()

        response = generation[0]['generated_text'][-1]['content']
        peak_memory_usage = vram_handler.measure_vram()

        self.inference_time_list.append((end_time-start_time))
        self.ttft_list.append(ttft_handler.ttft)
        self.memory_usage_list.append(peak_memory_usage)
        self.power_usage_list.append(power_handler.get_average())
        power_handler.kill()


        return response, (end_time-start_time), ttft_handler.ttft, peak_memory_usage, power_handler.get_average()
    

    def _get_llm_judge_score(self, response, prompt):
        # print(f"PROMPT:\n{prompt}\n\nRESPONS:\n{response}\n\n")
        score = self.judge.get_score(prompt, response)

        indent_format = f"\033[{60}G"
        Printer.print_yellow(f"{indent_format} {score}", end='\n')
        
        return score
    

    def run(self, prompts):
        for index, (prompt, refrence_answer) in enumerate(prompts):
            response, inference_time, ttft, memory_usage, power_usage = self._generate_response(prompt)
            self._print_test_metrics(index, inference_time, ttft, memory_usage, power_usage)
            
            score = self._get_llm_judge_score(response, prompt)
            self.llm_judge_score_list.append(score)

            self.num_test += 1
            self._print_test_status()
        print()


    def print_summary(self):
        if len(self.inference_time_list) > 0:
            avg_inference_time = sum(self.inference_time_list) / len(self.inference_time_list)
        else:
            avg_inference_time = 0.0

        if len(self.ttft_list) > 0:
            avg_ttft = sum(self.ttft_list) / len(self.ttft_list)
        else:
            avg_ttft = 0.0

        if len(self.memory_usage_list) > 0:
            avg_memory_usage = sum(self.memory_usage_list) / len(self.memory_usage_list)
        else:
            avg_memory_usage = 0.0
        
        if len(self.power_usage_list) > 0:
            avg_power_usage = sum(self.power_usage_list) / len(self.power_usage_list)
        else:
            avg_power_usage = 0.0
        
        if len(self.llm_judge_score_list) > 0:
            avg_score = (sum(self.llm_judge_score_list) / len(self.llm_judge_score_list))
        else:
            avg_score = 0.0


        summary = {
            "quantize": str(self.quantize),
            "average_inference_time_sec": round(avg_inference_time, 2),
            "average_ttft_sec": round(avg_ttft, 2),
            "average_mem_usage_GB": round(avg_memory_usage, 2),
            "average_power_usage_W": round(avg_power_usage, 2),
            "total_tests": self.num_test,
            "avg_score": round(avg_score, 2)
        }

        print(json.dumps(summary, indent=4))

        os.makedirs("./results/question_answering", exist_ok=True)
        with open(f"./results/question_answering/{self.model_name.replace('/','-')}---quantize={str(self.quantize)}.json", "w") as file:
            file.write(json.dumps(summary, indent=4))


    def cleanup(self):        
        self.model_loader.cleanup()
        del self.model_loader
        self.model_loader = None
        gc.collect()

        torch.cuda.empty_cache()