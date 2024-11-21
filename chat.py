import os
import time
import threading

from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer

from utils import Colors
from model_loaders import LLamaModelLoader



class ChatStreamHandler:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None)
        self.ttft_list = [0]
    
    def stream(self, start_time):
        first_token = True
        # IMPORTANT: streamers 'timeout' has to be None for this to work
        for token in self.streamer:
            if first_token: self.ttft_list.append(time.time() - start_time) ; first_token = False
            print(f"{token}", end='', flush=True)
    
    def get_ttft(self):
        return self.ttft_list[-1]


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


def main(stream: bool, quantize: bool):

    model_id, hf_token = load_env_variables()
    model_loader = LLamaModelLoader(model_id, quantize, hf_token)
    stream_handler = ChatStreamHandler(model_loader.tokenizer)


    conversation_history = [
        {
            "role": "system",
            "content": "You are a helpful assistant, that responds as a pirate."
        }
    ]
    
    #################
    ### Chat loop ###
    #################
    while True:
        user_input = input(f"\n{Colors.WHITE_TEXT + Colors.BACKGROUND_LIGHT_BLUE}You:{Colors.RESET} ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        print(f"\n{Colors.WHITE_TEXT + Colors.BACKGROUND_GRAY}Bot:{Colors.RESET} ", end='')
        
        start_time = time.time()
        if stream:
            streaming_thread = threading.Thread(target=stream_handler.stream, args=(start_time,))
            streaming_thread.start()

        generation = model_loader.generator(
            conversation_history,
            streamer=stream_handler.streamer if stream else None,
            do_sample=True,
            temperature=0.6,
            top_p=0.6,
            max_new_tokens=512,
        )
        end_time = time.time()
        if stream: streaming_thread.join()
        
        response = generation[0]['generated_text'][-1]['content']
        if not stream: print(f"{response}")
        
        print(f"\n\n({end_time - start_time:.2f}s TTFT: {stream_handler.get_ttft():.2f}s)\n")

        conversation_history.append({
            "role": "assistant", 
            "content": response
        })


if __name__ == "__main__":
    main(stream=True, quantize=False)
