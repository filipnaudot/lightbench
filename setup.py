from setuptools import setup, find_packages

setup(
    name="lightbench",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "python-dotenv",
        "transformers",
        "bitsandbytes",
        "accelerate>=0.26.0",
        "nvidia-ml-py",
        "openai",
        "mistralai",
        "anthropic",
        "pynvml",
    ],
    extras_require={
        "test":     ["pytest"],
        "api":      ["fastapi", "uvicorn"],
        "notebook": ["ipywidgets"],
    },
)
