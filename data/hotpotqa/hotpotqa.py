import json

from data.data_set import DataSet


class HotpotQA(DataSet):
    def __init__(self, file_path: str = './data/hotpotqa/hotpot_test_fullwiki_v1-first-500.jsonl',
                 start: int | None = None,
                 end: int | None = None):
        """
        Initialize the dataset.

        Args:
            file_path (str): Path to the HotpotQA jsonl file.
            start_line (int): Starting line number (1-indexed).
            end_line (int): Ending line number (inclusive).
        """
        self.file_path: str = file_path
        self.start: int | None = start
        self.end: int | None = end
        self._data = None  # Lazy-loaded data
    
    def _load_data(self):
        """Load data points from the jsonl file."""
        with open(self.file_path, 'r') as json_file:
            lines = list(json_file)[self.start:self.end]
        data = []
        for line in lines:
            result = json.loads(line)
            question = result["question"]
            context = "".join(["".join(sentences) for para in result["context"] for sentences in para[1]])
            data.append((question, context))
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