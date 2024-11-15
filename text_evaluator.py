import os
import threading
import time
import random
import gc
import json

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer

from utils import Printer
from evaluator import Evaluator
from llm_judge import LLMJudge


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
        self.llm_judge_score_list:list[int] = []

        self.tokenizer:AutoTokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.device:str = "cuda" if torch.cuda.is_available() else "cpu"

        self.streamer:TextIteratorStreamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

        if self.quantize:
            self.model = self._load_quantized_model()
        else:
            self.model = model_name

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id,
            token=self.hf_token,
        )


    def _handle_stream_output(self, streamer, start_time, ttft_list, print_stream=False):
        first_token = True
        for token in streamer:
            if first_token:
                ttft = time.time() - start_time
                ttft_list.append(ttft)
                first_token = False
            
            if print_stream:
                print(f"{token}", end='', flush=True)


    def _load_quantized_model(self):
        print("Loading quantized model...")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        return model


    def _clear_last_row(self):
        print(f"\r{' ' * 40}\r", end='', flush=True) # Clear last line
    
    def _print_test_metrics(self, index, inference_time, ttft, memory_usage):
        self._clear_last_row()
        print(f"\r{index+1} ({inference_time:.2f}s, TTFT: {ttft:.2f}s, Memory: {memory_usage:.2f} GB) ", end='')


    def _print_test_status(self):
        if self.num_test > 0: self._clear_last_row()
        avg_score = 0
        if len(self.llm_judge_score_list) > 0: avg_score = (sum(self.llm_judge_score_list) / len(self.llm_judge_score_list))
        Printer.print_cyan(f"\rAverage score: {avg_score:.2f}", end='')


    def _generate_response(self, prompt):
        ttft_list = []
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()

        streaming_thread = threading.Thread(target=self._handle_stream_output, args=(self.streamer, start_time, ttft_list))
        streaming_thread.start()

        generation = self.generator(
            prompt,
            streamer=self.streamer,
            do_sample=False,
            temperature=1.0,
            top_p=1,
            max_new_tokens=512,
        )
        end_time = time.time()

        streaming_thread.join()
        
        response = generation[0]['generated_text'][-1]['content']

        # Measure peak GPU memory usage (in GB)
        peak_memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 3)

        self.inference_time_list.append((end_time-start_time))
        self.ttft_list.append(ttft_list[-1])
        self.memory_usage_list.append(peak_memory_usage)

        return response, (end_time-start_time), ttft_list[-1], peak_memory_usage
    

    def _get_llm_judge_score(self, response, prompt):
        # print(f"PROMPT:\n{prompt}\n\nRESPONS:\n{response}\n\n")
        score = self.judge.get_score(prompt, response)

        indent_format = f"\033[{45}G"
        Printer.print_yellow(f"{indent_format} {score}", end='\n')
        
        return score
    

    def run(self, prompts):
        for index, (prompt, refrence_answer) in enumerate(prompts):
            response, inference_time, ttft, memory_usage = self._generate_response(prompt)
            self._print_test_metrics(index, inference_time, ttft, memory_usage)
            
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
        
        if len(self.llm_judge_score_list) > 0:
            avg_score = (sum(self.llm_judge_score_list) / len(self.llm_judge_score_list))
        else:
            avg_score = 0.0


        summary = {
            "quantize": str(self.quantize),
            "average_inference_time": round(avg_inference_time, 2),
            "average_ttft": round(avg_ttft, 2),
            "average_mem_usage": round(avg_memory_usage, 2),
            "total_tests": self.num_test,
            "avg_score": round(avg_score, 2)
        }

        print(json.dumps(summary, indent=4))

        os.makedirs("./results/question_answering", exist_ok=True)
        with open(f"./results/question_answering/{self.model_name.replace('/','-')}---quantize={str(self.quantize)}.json", "w") as file:
            file.write(json.dumps(summary, indent=4))


    def cleanup(self):
        del self.generator
        del self.tokenizer
        del self.model
        del self.streamer
        gc.collect()

        torch.cuda.empty_cache()