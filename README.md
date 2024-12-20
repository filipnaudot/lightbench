<div align='center'>
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="/readme_assets/lightbench_logo_lightmode.png">
        <img alt="lighbench logo" src="/readme_assets/lightbench_logo_darkmode.png" width="50%" height="50%">
    </picture>
    <p>
        <img src="https://img.shields.io/badge/Ubuntu-20.04-orange">
        <img src="https://img.shields.io/badge/python->=3.11.3-blue">
        <br>
        <img src="https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black">
        <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
    </p>
</div>


# LLM Benchmark Framework

## Human Evaluation
Human evaluation is straightforward with the interactive chat interface, enabling users to interact with the model in real-time and assess its responses in a conversational setting.

### Chat Example
Example of a chat using [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), running on a GTX 1080 TI.

![Demo of Terminal Chat Interface](./readme_assets/demo.gif)


## Automatic Evaluations
This framework supports automatic evaluation for:
1. **Code Generation**: Measure the model's ability to generate syntactically correct and functional code.
2. **Text Generation (Question Answering)**: Evaluate the model's performance in answering questions using the LLM-as-a-judge technique.


## Getting Started
**1. Install the required dependencies by running the following commands:**
```bash
bash install_dependencies.sh
```
This script **creates a Python virtual environment** and installs all required dependencies within it.

**2. Activating the Virtual Environment**

After running the installation script, you can activate the virtual environment using the command provided at the end of the script.

**3. Creating the .env File**

To enable the framework to function correctly, you need to create a .env file in the root directory of the project. The file should include the following keys:
```bash
OPENAI_API_KEY=
HUGGINGFACE_TOKEN=
MODEL_NAME=
```
`OPENAI_API_KEY`: Your OpenAI API key, required for text evaluation.

`HUGGINGFACE_TOKEN`: Your Hugging Face token, used for all evaluations if you are not using your own model loader.

`MODEL_NAME`: Specifies the model to use in the **chat.py** script for interactive chat.


## Paper
***LLMs and Efficiency: A Study in Computational Demands*** (Work in progress)