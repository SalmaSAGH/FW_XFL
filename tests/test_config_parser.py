import yaml
from pathlib import Path
import pytest
from config.config_parser import Config, FederatedLearningConfig, DatasetConfig, ModelConfig
from config.config_parser import TrainingConfig, StrategyConfig, NetworkConfig
from config.config_parser import EnergyConfig, MonitoringConfig, ServerConfig, ClientConfig
from config.config_parser import PathsConfig, ExperimentConfig, load_config


def test_load_config_creates_directories(tmp_path):
    config_data = {
        "experiment": {"name": "test", "description": "desc", "seed": 123},
        "federated_learning": {"num_rounds": 1, "num_clients": 4, "clients_per_round": 2, "local_epochs": 1},
        "dataset": {"name": "MNIST", "data_distribution": "iid", "train_test_split": 0.8, "batch_size": 32},
        "model": {"name": "TinyCNN", "input_shape": [1, 28, 28], "num_classes": 10},
        "training": {"optimizer": "adam", "learning_rate": 0.001, "momentum": 0.9, "weight_decay": 0.0},
        "strategy": {"type": "all_layers", "num_layers": None},
        "network": {"simulate_constraints": False, "latency_ms": 0.0, "latency_std_ms": 0.0, "bandwidth_mbps": 10.0, "packet_loss_rate": 0.0, "jitter_ms": 0.0},
        "energy": {"cpu_idle_watts": 10.0, "cpu_peak_watts": 50.0, "network_energy_per_bit_joules": 1e-6, "power_exponent": 1.0, "use_frequency_scaling": True},
        "monitoring": {"enabled": True, "metrics_interval_sec": 10, "log_level": "INFO"},
        "server": {"host": "localhost", "port": 5000, "aggregation_method": "fedavg"},
        "client": {"timeout_sec": 30, "max_retries": 3},
        "paths": {"data_dir": str(tmp_path / "data"), "models_dir": str(tmp_path / "models"), "logs_dir": str(tmp_path / "logs"), "results_dir": str(tmp_path / "results")}
    }

    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    config = load_config(str(config_file))

    assert isinstance(config, Config)
    assert config.experiment.name == "test"
    assert config.federated_learning.num_clients == 4
    assert config.paths.data_dir == str(tmp_path / "data")
    assert Path(config.paths.data_dir).exists()
    assert Path(config.paths.models_dir).exists()
    assert Path(config.paths.logs_dir).exists()
    assert Path(config.paths.results_dir).exists()


def test_invalid_clients_per_round_raises():
    with pytest.raises(ValueError):
        FederatedLearningConfig(num_rounds=1, num_clients=2, clients_per_round=3, local_epochs=1)
