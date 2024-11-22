# API Documentation

## Overview

This API provides an interface for interacting with a LLM capable of generating responses based on structured conversational prompts.

---

## Features
- **Flexible Input Format**: Accepts a structured list of messages with `role` and `content` fields.
- **Customizable Output**: Supports parameters like `max_length` for controlling response length.
- **Lightweight and Fast**: Built with FastAPI, ensuring high performance and ease of deployment.

---

## Getting Started

### Prerequisites
Run the installation script in the root directory:
```bash
bash install_dependencies.sh
```
---

### Starting the API

To launch the API server, run the following command:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

This will start the server on `http://127.0.0.1:8000`.

---

### Endpoint Details

#### **Generate Text**
- **URL**: `/generate/`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### **Request Body**
The request body must include:
1. `prompt`: A list of message objects, each containing:
   - `role` (string): Specifies the role of the message sender (e.g., `system`, `user`, `assistant`).
   - `content` (string): The text content of the message.
2. `max_length`: (optional, integer): Maximum number of tokens to generate in the response. Default is `512`.

**Example Request**:
```json
{
    "prompt": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_length": 512
}
```

---

#### **Response**
The API responds with a JSON object containing the generated text:

**Example Response**:
```json
{
    "generation":
        [
            {
                "generated_text":
                    [
                        {"role":"system","content":"You are a helpful assistant."},
                        {"role":"user","content":"What is the capital of France?"},
                        {"role":"assistant","content":"The capital of France is Paris."}
                    ]
            }
        ],
    "response": "The capital of France is Paris."
}
```
Here, `generation` contains the formatted full-context response. And `response` contains the string version of the latest model response.

---

## Example Usage

### cURL Example
You can test the API using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/generate/" -H "Content-Type: application/json" -d '{
    "prompt": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_length": 512
}'
```

---

## Prompt Format

The API utilizes a structured prompt format. Each prompt is a list of message objects, where each object includes a `role` and `content` field:

- **`role`**: Specifies the sender of the message. Supported roles include:
  - `"system"`: Used to define the behavior or persona of the assistant.
  - `"user"`: Represents the user's input or query.
  - `"assistant"`: Contains the assistant's response (used in historical context).

- **`content`**: Contains the actual text of the message.

### Example Prompt
```python
prompt = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
]
```


---

### Python Example
A Python example using the API is available in `mwe.py`

---