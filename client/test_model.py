import pytest
import torch
from client.model import create_model, DATASET_CONFIG


def test_all_models_creation():
    """Test that all models can be created successfully"""
    for dataset, config in DATASET_CONFIG.items():
        model_name, num_classes, in_channels, input_size = config

        print(f"Testing {model_name} for {dataset} (classes: {num_classes}, channels: {in_channels}, size: {input_size})")

        try:
            model = create_model(model_name, num_classes, in_channels, input_size)
            assert model is not None, f"Failed to create {model_name}"

            # Test forward pass with a batch
            batch_size = 2  # Small batch to avoid memory issues
            x = torch.randn(batch_size, in_channels, input_size, input_size)
            with torch.no_grad():
                output = model(x)
                assert output.shape == (batch_size, num_classes), f"Wrong output shape for {model_name}: {output.shape}"

            print(f"✓ {model_name} passed")

        except Exception as e:
            pytest.fail(f"Model {model_name} failed: {str(e)}")


def test_model_forward_pass():
    """Test forward pass for specific model-dataset combinations"""
    test_cases = [
        ('MNIST', 'TinyCNN'),
        ('FashionMNIST', 'MicroLeNet'),
        ('CIFAR10', 'DepthwiseCNN'),
        ('CIFAR100', 'CIFAR100CNN'),
        ('EMNIST', 'EMNISTCNN'),
    ]

    for dataset, model_name in test_cases:
        config = DATASET_CONFIG[dataset]
        _, num_classes, in_channels, input_size = config

        print(f"Testing forward pass: {model_name} on {dataset}")

        model = create_model(model_name, num_classes, in_channels, input_size)

        # Test with different batch sizes
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, in_channels, input_size, input_size)
            with torch.no_grad():
                output = model(x)
                assert output.shape == (batch_size, num_classes), f"Batch size {batch_size} failed for {model_name}"


if __name__ == "__main__":
    test_all_models_creation()
    test_model_forward_pass()
    print("All model tests passed!")