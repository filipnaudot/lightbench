import os
import time
from dotenv import load_dotenv

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

WHITE_TEXT = '\033[97m'
BACKGROUND_LIGHT_BLUE = '\033[104m'
BACKGROUND_GRAY = '\033[100m'
RESET = '\033[0m'



def load_env():
    load_dotenv()
    
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    model_id = os.getenv("MODEL_NAME")
    
    return model_id, hf_token


def load_quantized_model(model_id):
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    return model


def main(stream: bool = False, QUANTIZE: bool = False):

    model_id, hf_token = load_env()
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
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    #################
    ### Chat loop ###
    #################
    while True:
        user_input = input(f"{WHITE_TEXT + BACKGROUND_LIGHT_BLUE}You:{RESET} ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        print(f"\n{WHITE_TEXT + BACKGROUND_GRAY}Bot:{RESET} ", end='')
        
        start_time = time.time()
        generation = generator(
            conversation_history,
            streamer=streamer if stream else None,
            do_sample=False,
            temperature=1.0,
            top_p=1,
            max_new_tokens=512,
        )
        end_time = time.time()
        
        response = generation[0]['generated_text'][-1]['content']
        
        if not stream:
            print(f"{response}")
        
        print(f"\n({end_time - start_time:.2f}s)\n")

        conversation_history.append({
            "role": "assistant", 
            "content": response
        })


if __name__ == "__main__":
    main(stream=True, QUANTIZE=False)
