import os
import json

from lightbench.utils import Printer
from lightbench.evaluators.evaluator import Evaluator
from lightbench.metrics.llm_judge import LLMJudge
from lightbench.loaders.loader import LLMServiceLoader
from lightbench.loaders.generation import Generation
from lightbench.data.hotpotqa.hotpotqa import HotpotQA


class TextEvaluator(Evaluator):
    def __init__(self, model_loader: LLMServiceLoader, judge:LLMJudge, num_test_limit: int | None = None, verbose:bool = False):
        super().__init__(verbose)

        self.judge:LLMJudge = judge
        self.num_test_limit: int | None = num_test_limit

        self.num_test:int = 0
        self.inference_time_list:list[float] = []
        self.ttft_list:list[float] = []
        self.memory_usage_list:list[float] = []
        self.power_usage_list:list[float] = []
        self.llm_judge_score_list:list[int] = []

        self.model_loader = model_loader


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


    def _create_qa_prompts(self):
        prompts = []
        system_command = {
            "role": "system",
            "content": "You are a question-answering assistant. Answer the user's question \
                        based on the provided context. Respond with only the answer in a single sentence.",
        }
        dataset = HotpotQA(start=0, end=self.num_test_limit)
        for question, context in dataset:
            prompt = (
                [
                    system_command,
                    {
                        "role": "user",
                        "content": f'Answer the question based on the following context: "{context}". The question is: "{question}"',
                    }
                ]
            )
            prompts.append(prompt)
        return prompts


    def _generate_response(self, prompt) -> Generation:
        generation: Generation = self.model_loader.generate(prompt)

        self.inference_time_list.append(generation.inference_time)
        self.ttft_list.append(generation.ttft)
        self.memory_usage_list.append(generation.peak_memory_usage)
        self.power_usage_list.append(generation.avg_power_usage)
        
        return generation
    

    def _get_llm_judge_score(self, response, prompt):
        return self.judge.get_score(prompt, response)
    

    def run(self):
        prompts = self._create_qa_prompts()
        for index, prompt in enumerate(prompts):
            generation: Generation = self._generate_response(prompt)
            self._print_test_metrics(index,
                                     generation.inference_time,
                                     generation.ttft,
                                     generation.peak_memory_usage,
                                     generation.avg_power_usage)
            
            score = self._get_llm_judge_score(generation.response, prompt)
            self.llm_judge_score_list.append(score)
            Printer.print_yellow(f"\t{score}")
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
            "total_tests": self.num_test,
            "avg_score": round(avg_score, 2),
            "average_inference_time_sec": round(avg_inference_time, 2),
        }
        if self.model_loader.is_local():
            summary.update({
                "average_ttft_sec": round(avg_ttft, 2),
                "quantize": str(self.model_loader.quantize),
                "average_mem_usage_GB": round(avg_memory_usage, 2),
                "average_power_usage_W": round(avg_power_usage, 2),
            })

        print(json.dumps(summary, indent=4))

        os.makedirs("./results/question_answering", exist_ok=True)
        with open(f"./results/question_answering/{self.model_loader.name().replace('/','-')}.json", "w") as file:
            file.write(json.dumps(summary, indent=4))


    def cleanup(self): pass