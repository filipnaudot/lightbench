import os
from dotenv import load_dotenv

import torch
# Hugging face
from transformers import AutoModelForCausalLM, AutoTokenizer


load_dotenv()



def generate_response(input_text, model, tokenizer, device, max_length=100):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate a response using the model
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=max_length, 
            num_return_sequences=1,
            do_sample=True,   # Use sampling to create a more conversational response
            temperature=0.7,  # Adjust temperature for more randomness
            top_p=0.9,        # Nucleus sampling
        )
    
    # Decode the generated tokens into text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    model_name = os.getenv("MODEL_NAME")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
    model.eval()

    # Move the model to GPU
    device = torch.device("cuda")
    model.to(device)
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        bot_response = generate_response(user_input, model, tokenizer, device)
        print(f"Bot: {bot_response}")


if __name__ == "__main__":
    main()
