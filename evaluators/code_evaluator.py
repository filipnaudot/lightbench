import os
import gc
import io
from contextlib import redirect_stdout
import time
import threading
import signal
import json

from dotenv import load_dotenv

from utils import Printer
from evaluators.evaluator import Evaluator
from metrics.metrics import TTFT, VRAM, PowerUsage
from loaders.model_loaders import LLamaModelLoader

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


class CodeEvaluator(Evaluator):
    def __init__(self, model_name, quantize=False, few_shot=False, verbose=False):
        super().__init__(verbose)
        self.few_shot:bool = few_shot
        self.quantize:bool = quantize
        
        self.model_name:str = model_name
        self.hf_token:str = HUGGINGFACE_TOKEN

        self.inference_time_list:list[float] = []
        self.ttft_list:list[float] = []
        self.memory_usage_list:list[float] = []
        self.power_usage_list:list[float] = []

        self.num_test:int = 0
        self.passed_test:int = 0

        self.model_loader = LLamaModelLoader(model_name, quantize, HUGGINGFACE_TOKEN)

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
        percentage = 0
        if self.num_test > 0: percentage = (self.passed_test / self.num_test) * 100
        Printer.print_cyan(f"\rTests Passed: {self.passed_test}/{self.num_test} ({percentage:.2f}%)", end='')


    def _preprocess_data(self, data):
        if f"```python" in data:
            data = data[data.find(f"```python") + len(f"```python"):]
            data = data[:data.find("```")]
        return data


    def _timeout_handler(self, signum, frame):
        raise TimeoutError("Execution timed out!")


    def _validate_code(self, code, test, shots):
        indent_format = f"\033[{65}G" if shots > 0 else f"\033[{60}G"
        end = "" if self.verbose else "\n"

        full_code_to_execute = f"{code}\n\n{test}"
        Printer.print_yellow(f"{indent_format}{shots} ", end='')
        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(10)
        try:
            # Redirect stdout for cleaner terminal since executed code might have prints
            with io.StringIO() as buffer, redirect_stdout(buffer):
                exec(full_code_to_execute, globals())
        except AssertionError:
            Printer.print_red(f"FAILED", end=end)
            if self.verbose: print(f": test {str(test)} FAILED")
            return 0, f"There is a logical error in the code. TEST: {str(test)} FAILED"
        except Exception as error:
            Printer.print_red(f"FAILED", end=end)
            if self.verbose: print(f": An error occurred: {error}")
            return 0, f"ERROR: {error}"
        except TimeoutError as error:
            Printer.print_red(f"FAILED", end=end)
            if self.verbose: print(f": {error}")
            return 0, f"FAILED: {error}"
        finally:
            signal.alarm(0)
        
        Printer.print_green(f"PASSED")
        return 1, "PASSED"
    

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
            temperature=1.0, # Set to 1 since we are NOT using sample
            top_p=1,         # Set to 1 since we are NOT using sample
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
    
    
    def _create_few_shot_prompt(self, prompt, response, message):
        return [
            *prompt,
            {
                "role": "assistant",
                "content": response
            },
            {
                "role": "user",
                "content": f' While running that code I received the following: {message}. Can you update the code and fix the problem?',
            }
        ]
    

    def run(self, prompts):
        for index, (prompt, test) in enumerate(prompts):
            response, inference_time, ttft, memory_usage, power_usage = self._generate_response(prompt)
            self._print_test_metrics(index, inference_time, ttft, memory_usage, power_usage)
            extracted_code = self._preprocess_data(response).strip()
            passed, message = self._validate_code(extracted_code, test, 0)

            if self.few_shot and not passed:
                prompt = self._create_few_shot_prompt(prompt, response, message)
                shots = 1

                while not passed and shots <= 2:
                    self._print_test_status()

                    response, inference_time, ttft, memory_usage, power_usage = self._generate_response(prompt)
                    self._print_test_metrics(index, inference_time, ttft, memory_usage, power_usage)
                    extracted_code = self._preprocess_data(response).strip()
                    passed, message = self._validate_code(extracted_code, test, shots)

                    prompt = self._create_few_shot_prompt(prompt, response, message)
                    shots += 1

            self.num_test += 1
            if passed:
                self.passed_test += 1

            self._print_test_status()
        print()
    

    def cleanup(self):        
        self.model_loader.cleanup()
        del self.model_loader
        self.model_loader = None
        gc.collect()


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

        if self.num_test > 0:
            passed_percentage = (self.passed_test / self.num_test) * 100
        else:
            passed_percentage = 0.0

        summary = {
            "quantize": str(self.quantize),
            "few_shot": str(self.few_shot),
            "average_inference_time_sec": round(avg_inference_time, 2),
            "average_ttft_sec": round(avg_ttft, 2),
            "average_mem_usage_GB": round(avg_memory_usage, 2),
            "average_power_usage_W": round(avg_power_usage, 2),
            "passed_tests": self.passed_test,
            "total_tests": self.num_test,
            "passed_percentage": round(passed_percentage, 2)
        }

        print(json.dumps(summary, indent=4))

        os.makedirs("./results/code_evaluation", exist_ok=True)
        with open(f"./results/code_evaluation/{self.model_name.replace('/','-')}---quantize={str(self.quantize)}--few_shot={str(self.few_shot)}.json", "w") as file:
            file.write(json.dumps(summary, indent=4))