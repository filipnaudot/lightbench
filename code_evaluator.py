import os
import time
from queue import Queue
import threading

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer

from utils import Printer

class CodeEvaluator:
    def __init__(self, model_id, hf_token, quantize=False):
        self.quantize = quantize
        self.model = model_id
        self.hf_token = hf_token

        self.num_test = 0
        self.passed_test = 0

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.quantize:
            self.model = self.load_quantized_model()

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            pad_token_id=self.tokenizer.eos_token_id,
            token=self.hf_token,
        )


    def load_quantized_model(self):
        print("Loading quantized model...")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        return model


    def handle_stream_output(self, streamer, queue, start_time, ttft_list, verbose=False):
        first_token = True
        for token in streamer:
            if first_token:
                ttft = time.time() - start_time
                ttft_list.append(ttft)
                first_token = False
            
            if verbose:
                print(f"{token}", end='', flush=True)
            queue.put(token)


    def preprocess_data(self, data):
        if f"```python" in data:
            data = data[data.find(f"```python") + len(f"```python"):]
            data = data[:data.find("```")]
        return data


    def validate_code(self, code, test, attempts):
        full_code_to_execute = f"{code}\n\n{test}"
        Printer.print_yellow(f"{attempts}", end=' ')
        try:
            exec(full_code_to_execute, globals())
            Printer.print_green("PASSED")
            return 1, "PASSED"
        except AssertionError:
            Printer.print_red("FAILED: ", end='')
            print(f"test {str(test)} FAILED")
            return 0, f"TEST {str(test)} FAILED"
        except Exception as error:
            Printer.print_red("FAILED: ", end='')
            print(f"An error occurred: {error}")
            return 0, f"ERROR: {error}"


    def run(self, prompts, attempts=0, few_shot=False):
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        queue = Queue()

        
        for index, (prompt, test) in enumerate(prompts):
            ttft_list = []
            start_time = time.time()

            streaming_thread = threading.Thread(target=self.handle_stream_output, args=(streamer, queue, start_time, ttft_list))
            streaming_thread.start()

            generation = self.generator(
                prompt,
                streamer=streamer,
                do_sample=False,
                temperature=1.0,
                top_p=1,
                max_new_tokens=512,
            )
            end_time = time.time()

            streaming_thread.join()
            
            response = generation[0]['generated_text'][-1]['content']
            # print(f"{response}")
            
            extracted_code = self.preprocess_data(response).strip()

            print(f"\r{' ' * 40}", end='') # Clear last line            
            if attempts == 0:
                print(f"\r{index+1} ({end_time - start_time:.2f}s TTFT: {ttft_list[-1]:.2f}s)", end='\t')

            else:
                print(f"\r\t({end_time - start_time:.2f}s TTFT: {ttft_list[-1]:.2f}s)", end='\t')
                
            passed, message = self.validate_code(extracted_code, test, attempts)

            if few_shot and not passed and attempts < 2:
                prompts = [(
                    [
                        *prompt,  # Unpacking the list
                        {
                            "role": "assistant",
                            "content": response
                        },
                        {
                            "role": "user",
                            "content": f' While running that code I received the following: {message}. Can you update the code and fix the problem?',
                        }
                    ], test)
                ]
                self.run(prompts, attempts=attempts+1, few_shot=few_shot)
            
            self.num_test += 1
            if passed: self.passed_test += 1

            percentage = (self.passed_test / self.num_test) * 100
            Printer.print_cyan(f"\rTests Passed: {self.passed_test}/{self.num_test} ({percentage:.2f}%)", end='')