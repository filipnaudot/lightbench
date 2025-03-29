from pathlib import Path
import json

from lightbench.data.data_set import DataSet


class MBPPDataset(DataSet):
    """
    A dataset for managing raw MBPP data.
    Each data point is a dictionary loaded from a jsonl file.
    """

    def __init__(self,
                 file_path: str = None,
                 start: int | None = None,
                 end: int | None = None):
        """
        Initialize the dataset.

        Args:
            file_path (str): Path to the MBPP jsonl file.
            start_line (int): Starting line number (1-indexed).
            end_line (int): Ending line number (inclusive).
        """
        if file_path is None:
            self.file_path = Path(__file__).parent / 'mbpp.jsonl'
        self.start: int | None = start
        self.end: int | None = end
        self._data = None  # Lazy-loaded data

    def _load_data(self):
        """Load data points from the jsonl file."""
        with open(self.file_path, 'r') as json_file:
            lines = list(json_file)[self.start:self.end]
        data = []
        for line in lines:
            json_line = json.loads(line)
            data.append((json_line["text"],
                         json_line["test_list"][0],
                         json_line["test_list"][1],
                         json_line["test_list"][2]))
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