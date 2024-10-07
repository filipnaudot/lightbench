import os
import time
from dotenv import load_dotenv

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig



QUANTIZE = False

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
model_id = os.getenv("MODEL_NAME")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = model_id
if QUANTIZE:
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

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

def main():
    while True:
        user_input = input("Enter: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        conversation_history.append(
            {
                "role": "user",
                "content": user_input
            })
        
        start_time = time.time()
        generation = generator(
            conversation_history,
            do_sample=False,
            temperature=1.0,
            top_p=1,
            max_new_tokens=256,
        )
        end_time = time.time()

        response = generation[0]['generated_text'][-1]['content']
        
        print(f"\n({end_time - start_time:.2f}) Bot: {response}\n\n")
        
        conversation_history.append(
            {
                "role": "assistant", 
                "content": response
            })

if __name__ == "__main__":
    main()
