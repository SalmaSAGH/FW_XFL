import pytest
from client.dataset import DatasetPartitioner


def test_iid_partition_equal_sizes(dummy_iid_dataset):
    partitioner = DatasetPartitioner(dummy_iid_dataset, num_clients=3, distribution="iid", seed=123)
    assert len(partitioner.client_indices) == 3
    sizes = [len(indices) for indices in partitioner.client_indices]
    assert sizes == [4, 4, 4]


def test_non_iid_partition_respects_class_groups(dummy_non_iid_dataset):
    partitioner = DatasetPartitioner(dummy_non_iid_dataset, num_clients=3, distribution="non_iid", seed=123)
    assert len(partitioner.client_indices) == 3

    # Each client should receive samples from exactly two classes
    for client_indices in partitioner.client_indices:
        labels = {dummy_non_iid_dataset.targets[idx] for idx in client_indices}
        assert len(labels) == 2


def test_client_id_wrapping(dummy_iid_dataset):
    partitioner = DatasetPartitioner(dummy_iid_dataset, num_clients=3, distribution="iid", seed=123)
    dataset0 = partitioner.get_client_dataset(3)
    dataset1 = partitioner.get_client_dataset(0)
    assert len(dataset0) == len(dataset1)
