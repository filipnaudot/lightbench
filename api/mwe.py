###################################
# Simple minimum working example. #
###################################


import requests

url = "http://127.0.0.1:8000/generate/"
payload = {
    "prompt": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_length": 512
}
api_response = requests.post(url, json=payload).json()
full_context_response = api_response["generation"]
most_recent_model_response = api_response["response"]


print(f"API response: {api_response}")
print(f"Full context: {full_context_response}")
print(f"Latest response: {most_recent_model_response}")