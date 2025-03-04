<div align='center'>
    <picture>
        <source media="(prefers-color-scheme: light)" srcset="/readme_assets/lightbench_logo_lightmode.png">
        <img alt="lighbench logo" src="/readme_assets/lightbench_logo_darkmode.png" width="50%" height="50%">
    </picture>
    <p>
        <img src="https://img.shields.io/badge/Ubuntu-20.04-orange">
        <img src="https://img.shields.io/badge/python->=3.11.3-blue">
        <img src="https://img.shields.io/badge/License-MIT-blue.svg">
        <!-- <img src="https://img.shields.io/badge/version-1.0.0-blue)"> -->
        <br>
        <img src="https://img.shields.io/badge/-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black">
        <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
    </p>
</div>


# lightbench
*A lightweight benchmarking framework for LLMs.*

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
**lightbench** is designed to offer both interactive and automated benchmarking for large language models, enabling comprehensive evaluation of code generation and question answering capabilities.

## Key Features
- **Human Evaluation:** Interactive chat interface.
- **Automatic Evaluations:** Automated tests for code and text outputs.
- **Extensible Architecture:** Easy integration of new evaluators and metrics.

## Installation
1. **Dependencies:**  
   Ensure you have Python 3.8+ installed.
2. **Setup Environment:**  
   Run the installation script:
   ```bash
   bash install_dependencies.sh
   ```
3. **Configure Environment:**  
   Create a `.env` file with your `OPENAI_API_KEY`, `HUGGINGFACE_TOKEN`, and `MODEL_NAME`.

## Usage
- **Interactive Chat:**  
  Run `chat.py` to start the chat interface. This will use the model specified by `MODEL_NAME` in the `.env` file. Below is an example of a chat using [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), running on a GTX 1080 TI.
  ![Demo of Terminal Chat Interface](./readme_assets/demo.gif)

- **Automated Evaluations:**  
  See examples in [`examples.ipynb`](examples.ipynb).

## Project Structure
- **data**: Benchmark datasets.
- **evaluators**: Modules for both code and text evaluation.
- **loaders**: Tools to load and manage models.
- **metric**: Available metrics for local and API based models.

<!-- ## Contributing
We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting issues, pull requests, and feature ideas.  
Questions or suggestions? Open an issue. -->

## Citation
**Paper.** If you refer to the research paper related to this project, please cite:
```bibtex
@inproceedings{naudot2025performance,
  author    = {Filip Naudot},
  title     = {Performance and Computational Demands of LLMs: Impact of Model Size and Quantization},
  booktitle = {Proceedings of Umeå’s 28th Student Conference in Computing Science (USCCS 2025)},
  editor    = {Thomas Hellström},
  year      = {2025},
  publisher = {Umeå University, Sweden},
  note      = {Branch \texttt{conf-paper} used for paper results},
}
```

**Repository.**
If you use **lightbench** in your research, please cite the repository:
```bibtex
@misc{lightbench2025,
  author    = {Filip Naudot},
  title     = {lightbench},
  year      = {2025},
  howpublished = {\url{https://github.com/filipnaudot/lightbench}},
}
```

## License
Distributed under the MIT License. See `LICENSE` for more information.
