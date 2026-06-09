import pytest
import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


@pytest.fixture
def dummy_iid_dataset():
    data = [torch.zeros(1, 28, 28) for _ in range(12)]
    targets = [i % 3 for i in range(12)]
    return DummyDataset(data, targets)


@pytest.fixture
def dummy_non_iid_dataset():
    data = [torch.zeros(1, 28, 28) for _ in range(12)]
    targets = [0] * 4 + [1] * 4 + [2] * 4
    return DummyDataset(data, targets)
