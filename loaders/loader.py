from abc import ABC, abstractmethod


class LLMServiceLoader(ABC):
    """
    Abstract class for LLM loaders. Provides a common interface
    for generating text, and cleaning up resources.
    """

    @abstractmethod
    def generate(self, prompt, max_tokens: int) -> str:
        """
        Generate text based on the given prompt and max tokens.
        """
        pass

    def name(self) -> str:
        """
        Returns the name of the model.
        """
        pass

    def is_local(slef) -> bool:
        """
        Returns true if using a local model, false if using an API.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Perform cleanup of model resources and release memory.
        """
        pass