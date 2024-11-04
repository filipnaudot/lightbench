from abc import ABC, abstractmethod

class Evaluator(ABC):
    def __init__(self, verbose=False):
        self.verbose: bool = verbose

    @abstractmethod
    def run(self, prompts):
        pass
    
    @abstractmethod
    def print_summary(self):
        pass
    
    @abstractmethod
    def cleanup(self):
        pass