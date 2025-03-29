from abc import ABC, abstractmethod

class DataSet(ABC):
    """
    Abstract DataSet class.
    Subclasses must implement the __iter__ method.
    """
    @abstractmethod
    def __iter__(self):
        """Return an iterator over the dataset."""
        pass