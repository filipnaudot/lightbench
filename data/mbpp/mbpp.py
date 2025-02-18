import json

from data.data_set import DataSet


class MBPPDataset(DataSet):
    """
    A dataset for managing raw MBPP data.
    Each data point is a dictionary loaded from a jsonl file.
    """

    def __init__(self, file_path='./data/mbpp/mbpp.jsonl', start_line=1, end_line=450):
        """
        Initialize the dataset.

        Args:
            file_path (str): Path to the MBPP jsonl file.
            start_line (int): Starting line number (1-indexed).
            end_line (int): Ending line number (inclusive).
        """
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self._data = None  # Lazy-loaded data

    def _load_data(self):
        """Load data points from the jsonl file."""
        with open(self.file_path, 'r') as json_file:
            # Adjust indices: start_line is 1-indexed while list indexing is 0-indexed.
            lines = list(json_file)[self.start_line - 1:self.end_line]
        data = [json.loads(line) for line in lines]
        return data

    def __iter__(self):
        """
        Iterate over the dataset.
        Data is loaded lazily on the first iteration.
        """
        if self._data is None:
            self._data = self._load_data()
        return iter(self._data)

    def __len__(self):
        """
        Return the number of data points in the dataset.
        """
        if self._data is None:
            self._data = self._load_data()
        return len(self._data)