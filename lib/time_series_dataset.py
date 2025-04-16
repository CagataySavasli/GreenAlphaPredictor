from torch.utils.data import Dataset
import torch
import numpy as np

class TimeSeriesDataset(Dataset):
    """
    Custom Dataset for sequence data.
    Each sample is a tuple (sequence, target) where:
        - sequence: sequence of ESG features (shape: seq_length x n_features)
        - target: corresponding scaled Price value.
    """

    def __init__(self, sequences: np.array, targets: np.array):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return sequence, target
