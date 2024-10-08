import os
import time
from queue import Queue
import threading

from dotenv import load_dotenv
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer

WHITE_TEXT = '\033[97m'
BACKGROUND_LIGHT_BLUE = '\033[104m'
BACKGROUND_GRAY = '\033[100m'
RESET = '\033[0m'



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
        
        print(f"{token}", end='', flush=True)
        queue.put(token)


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

    conversation_history = [
        {
            "role": "system",
            "content": "You are a helpful assistant, that responds as a pirate."
        }
    ]
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    queue = Queue()

    #################
    ### Chat loop ###
    #################
    while True:
        user_input = input(f"\n{WHITE_TEXT + BACKGROUND_LIGHT_BLUE}You:{RESET} ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        print(f"\n{WHITE_TEXT + BACKGROUND_GRAY}Bot:{RESET} ", end='')
        
        ttft_list = []
        start_time = time.time()
        if stream:
            streaming_thread = threading.Thread(target=print_stream_output, args=(streamer, queue, start_time, ttft_list))
            streaming_thread.start()

        generation = generator(
            conversation_history,
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
        
        if not stream:
            print(f"{response}")
        
        ttft = ttft_list[-1] if stream else 0
        print(f"\n\n({end_time - start_time:.2f}s TTFT: {ttft:.2f}s)\n")

        conversation_history.append({
            "role": "assistant", 
            "content": response
        })


if __name__ == "__main__":
    main(stream=True, QUANTIZE=False)
