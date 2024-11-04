import os
import gc
import io
from contextlib import redirect_stdout
import time
import threading
import signal
import json


import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer

from utils import Printer
from evaluator import Evaluator

class CodeEvaluator(Evaluator):
    def __init__(self, model_name, hf_token, quantize=False, few_shot=False, verbose=False):
        super().__init__(verbose)
        self.few_shot:bool = few_shot
        self.quantize:bool = quantize
        
        self.model_name:str = model_name
        self.hf_token:str = hf_token

        self.inference_time_list:list[float] = []
        self.ttft_list:list[float] = []
        self.memory_usage_list:list[float] = []

        self.num_test:int = 0
        self.passed_test:int = 0

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
        self._clear_last_row()
        percentage = 0
        if self.num_test > 0: percentage = (self.passed_test / self.num_test) * 100
        Printer.print_cyan(f"\rTests Passed: {self.passed_test}/{self.num_test} ({percentage:.2f}%)", end='')


    def _handle_stream_output(self, streamer, start_time, ttft_list, print_stream=False):
        first_token = True
        for token in streamer:
            if first_token:
                ttft = time.time() - start_time
                ttft_list.append(ttft)
                first_token = False
            
            if print_stream:
                print(f"{token}", end='', flush=True)


    def _preprocess_data(self, data):
        if f"```python" in data:
            data = data[data.find(f"```python") + len(f"```python"):]
            data = data[:data.find("```")]
        return data


    def _timeout_handler(self, signum, frame):
        raise TimeoutError("Execution timed out!")


    def _validate_code(self, code, test, shots):
        indent_format = f"\033[{50}G" if shots > 0 else f"\033[{45}G"
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
            response, inference_time, ttft, memory_usage = self._generate_response(prompt)
            self._print_test_metrics(index, inference_time, ttft, memory_usage)
            extracted_code = self._preprocess_data(response).strip()
            passed, message = self._validate_code(extracted_code, test, 0)

            if self.few_shot and not passed:
                prompt = self._create_few_shot_prompt(prompt, response, message)
                shots = 1

                while not passed and shots <= 2:
                    self._print_test_status()

                    response, inference_time, ttft, memory_usage = self._generate_response(prompt)
                    self._print_test_metrics(index, inference_time, ttft, memory_usage)
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
        del self.generator
        del self.tokenizer
        del self.model
        del self.streamer
        gc.collect()

        torch.cuda.empty_cache()


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

        if self.num_test > 0:
            passed_percentage = (self.passed_test / self.num_test) * 100
        else:
            passed_percentage = 0.0

        summary = {
            "quantize": str(self.quantize),
            "few_shot": str(self.few_shot),
            "average_inference_time": round(avg_inference_time, 2),
            "average_ttft": round(avg_ttft, 2),
            "average_mem_usage": round(avg_memory_usage, 2),
            "passed_tests": self.passed_test,
            "total_tests": self.num_test,
            "passed_percentage": round(passed_percentage, 2)
        }

        print(json.dumps(summary, indent=4))

        os.makedirs("./results", exist_ok=True)
        with open(f"./results/{self.model_name.replace('/','-')}---quantize={str(self.quantize)}--few_shot={str(self.few_shot)}.json", "w") as file:
            file.write(json.dumps(summary, indent=4))