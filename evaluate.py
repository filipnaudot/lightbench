import os
import time
from queue import Queue
import threading

from dotenv import load_dotenv
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer

from utils import Colors


def load_env_variables():
    load_dotenv()
    
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    model_id = os.getenv("MODEL_NAME")
    
    return model_id, hf_token


def load_quantized_model(model_id):
    print("Loading quantized model...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    return model


def print_stream_output(streamer, queue, start_time, ttft_list):
    first_token = True
    for token in streamer:
        if first_token:
            ttft = time.time() - start_time
            ttft_list.append(ttft)
            first_token = False
        
        # TODO: Add verbose mode here
        # print(f"{token}", end='', flush=True)
        queue.put(token)


def preprocess_data(data):
    if f"```python" in data:
        data = data[data.find(f"```python") + len(f"```python"):]
        data = data[:data.find("```")]
    return data


def main(stream: bool = False, QUANTIZE: bool = False):

    model_id, hf_token = load_env_variables()
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_id
    if QUANTIZE:
        model = load_quantized_model(model_id)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id,
        token=hf_token,
    )

    prompt = [
        {
            "role": "system",
            "content": "You are a Python programming assistant. Your task is to write Python functions according to the user's prompt. Respond only with the necessary Python code, including any imports if needed. Do not provide example usage, only the python function."
        }
    ]
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    queue = Queue()

    #################
    ### Chat loop ###
    #################
    while True:
        user_input = input(f"\n{Colors.WHITE_TEXT + Colors.BACKGROUND_LIGHT_BLUE}You:{Colors.RESET} ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        prompt.append({
            "role": "user",
            "content": user_input
        })
        
        print(f"\n{Colors.WHITE_TEXT + Colors.BACKGROUND_GRAY}Bot:{Colors.RESET} ", end='')
        
        ttft_list = []
        start_time = time.time()
        if stream:
            streaming_thread = threading.Thread(target=print_stream_output, args=(streamer, queue, start_time, ttft_list))
            streaming_thread.start()

        generation = generator(
            prompt,
            streamer=streamer if stream else None,
            do_sample=False,
            temperature=1.0,
            top_p=1,
            max_new_tokens=512,
        )
        end_time = time.time()

        if stream:
            streaming_thread.join()
        
        response = generation[0]['generated_text'][-1]['content']
        

        print(f"{response}")
        print("\n\n--------- EXTRACTED CODE ---------\n")
        print(preprocess_data(response))
        
        ttft = ttft_list[-1] if stream else 0
        print(f"\n\n({end_time - start_time:.2f}s TTFT: {ttft:.2f}s)\n")

        


if __name__ == "__main__":
    main(stream=True, QUANTIZE=False)
