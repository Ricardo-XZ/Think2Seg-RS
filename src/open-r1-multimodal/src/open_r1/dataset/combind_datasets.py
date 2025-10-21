import torch
from torch.utils.data import Dataset
import random
import numpy as np
from PIL import Image
import os

class CombinedRefSegDataset(Dataset):
    """
    A dataset class that combines multiple reference segmentation datasets.
    This allows for training on multiple datasets simultaneously.
    """

    def __init__(self, datasets):
        """
        Initialize the combined dataset.

        Args:
            datasets (list): List of dataset objects to combine
        """
        self.datasets = datasets
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = [0]

        # Calculate cumulative lengths for index mapping
        cumulative_length = 0
        for length in self.dataset_lengths:
            cumulative_length += length
            self.cumulative_lengths.append(cumulative_length)

        print(f"Created combined dataset with {len(self)} total samples:")
        for i, dataset in enumerate(datasets):
            print(f"  - Dataset {i+1}: {len(dataset)} samples")

    def __len__(self):
        """Return the total length of the combined dataset."""
        return sum(self.dataset_lengths)

    def _get_dataset_index(self, idx):
        """
        Map a global index to the corresponding dataset and local index.

        Args:
            idx (int): Global index

        Returns:
            tuple: (dataset_idx, local_idx) - The dataset index and the local index within that dataset
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for combined dataset of length {len(self)}")

        # Find which dataset the index belongs to
        dataset_idx = 0
        while dataset_idx < len(self.cumulative_lengths) - 1 and idx >= self.cumulative_lengths[dataset_idx + 1]:
            dataset_idx += 1

        # Calculate the local index within the dataset
        local_idx = idx - self.cumulative_lengths[dataset_idx]

        return dataset_idx, local_idx

    def __getitem__(self, idx):
        """
        Get an item from the combined dataset.

        Args:
            idx (int): Global index

        Returns:
            dict: Sample data with consistent format across all datasets
        """
        dataset_idx, local_idx = self._get_dataset_index(idx)

        try:
            # Get the sample from the appropriate dataset
            sample = self.datasets[dataset_idx][local_idx]

            # Add a dataset identifier for debugging/analysis
            sample['dataset_idx'] = dataset_idx

            return sample
        except Exception as e:
            print(f"Error loading sample {local_idx} from dataset {dataset_idx}: {str(e)}")
            # Try a random index as fallback
            fallback_idx = random.randint(0, len(self) - 1)
            return self[fallback_idx]

def create_combined_dataset(*datasets):
    """
    Create a combined dataset from multiple datasets.

    Args:
        *datasets: Variable number of dataset instances to combine

    Returns:
        CombinedRefSegDataset: The combined dataset

    Example:
        # Combine two datasets
        combined = create_combined_dataset(dataset1, dataset2)

        # Combine multiple datasets
        combined = create_combined_dataset(dataset1, dataset2, dataset3, dataset4)
    """
    if not datasets:
        raise ValueError("At least one dataset must be provided")

    return CombinedRefSegDataset(list(datasets))
