import pandas as pd
from typing import List, Optional, Any

class pacDataset:
    """
    Base class for PAC-style datasets. 
    Subclasses should implement extract_data() and create_prompt().
    """

    def __init__(self, dimension: str, sample_size: Optional[int] = None):
        """
        Args:
            dimension (str): The scoring dimension to evaluate (e.g., "Relevance", "Coherence").
            sample_size (Optional[int]): Number of samples to use (if None, use all).
        """
        self.dimension = dimension
        self.sample_size = sample_size

    def extract_data(self) -> pd.DataFrame:
        """
        Subclasses must implement this to load data into a pandas DataFrame.
        Must return DataFrame with at least: ['text', 'source_text', 'original_score', 'text_id'].
        """
        raise NotImplementedError("Subclasses must implement extract_data()")

    def create_prompt(self, row: pd.Series) -> str:
        """
        Subclasses must implement this to define how evaluation prompts are built.
        """
        raise NotImplementedError("Subclasses must implement create_prompt()")

    def get_available_dimensions(self) -> List[str]:
        """
        Subclasses must return the dimensions available for this dataset.
        """
        raise NotImplementedError("Subclasses must implement get_available_dimensions()")

    def get_data(self) -> pd.DataFrame:
        """
        Wrapper for extracting data with optional sampling applied.
        Ensures a consistent DataFrame format.
        """
        data = self.extract_data()

        required_cols = {"text", "source_text", "original_score", "text_id"}
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

        if self.sample_size is not None:
            data = data.sample(n=min(self.sample_size, len(data)), random_state=42)

        return data

    def build_prompts(self, data: pd.DataFrame) -> List[str]:
        """
        Given a dataset, return a list of prompts for evaluation.
        """
        return [self.create_prompt(row) for _, row in data.iterrows()]
