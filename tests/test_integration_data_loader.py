import pytest
import torch
from client.dataset import create_single_client_loader


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.targets = [0, 1, 1, 0]
        self.data = [torch.zeros(1, 28, 28) for _ in range(len(self.targets))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def test_create_single_client_loader_with_monkeypatch(monkeypatch):
    def fake_load_dataset(dataset_name, data_dir, train=True):
        return FakeDataset()

    monkeypatch.setattr('client.dataset.load_dataset', fake_load_dataset)
    loader = create_single_client_loader('MNIST', client_id=0, num_clients=2, batch_size=2)
    batches = list(loader)
    assert len(batches) > 0
    data, targets = batches[0]
    assert data.shape[0] <= 2
    assert targets.shape[0] <= 2
