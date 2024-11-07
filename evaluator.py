from abc import ABC, abstractmethod

class Evaluator(ABC):
    def __init__(self, verbose=False):
        self.verbose: bool = verbose

    @abstractmethod
    def run(self, prompts):
        raise NotImplementedError
    
    @abstractmethod
    def print_summary(self):
        raise NotImplementedError
    
    @abstractmethod
    def cleanup(self):
        raise NotImplementedError