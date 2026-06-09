import torch
from client.model import create_model, DATASET_CONFIG


def test_create_models_forward_pass():
    for dataset_name, config in DATASET_CONFIG.items():
        model_name, num_classes, in_channels, input_size = config
        model = create_model(model_name, num_classes, in_channels, input_size)
        assert model is not None

        batch = torch.randn(2, in_channels, input_size, input_size)
        with torch.no_grad():
            output = model(batch)
        assert output.shape == (2, num_classes)


def test_model_forward_with_different_batch_sizes():
    model_name, num_classes, in_channels, input_size = DATASET_CONFIG['MNIST']
    model = create_model(model_name, num_classes, in_channels, input_size)

    for batch_size in [1, 4, 8]:
        batch = torch.randn(batch_size, in_channels, input_size, input_size)
        with torch.no_grad():
            output = model(batch)
        assert output.shape == (batch_size, num_classes)
