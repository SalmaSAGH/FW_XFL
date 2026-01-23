"""
Configuration parser with validation using Pydantic
"""

import yaml
from pathlib import Path
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
import os


class ExperimentConfig(BaseModel):
    """Experiment configuration"""
    name: str
    description: str
    seed: int = 42


class FederatedLearningConfig(BaseModel):
    """Federated Learning parameters"""
    num_rounds: int = Field(gt=0, description="Number of FL rounds")
    num_clients: int = Field(gt=0, description="Total number of clients")
    clients_per_round: int = Field(gt=0, description="Clients per round")
    local_epochs: int = Field(gt=0, description="Local training epochs")
    
    @field_validator('clients_per_round')
    @classmethod
    def validate_clients_per_round(cls, v, info):
        num_clients = info.data.get('num_clients')
        if num_clients is not None and v > num_clients:
           raise ValueError("clients_per_round cannot exceed num_clients")
        return v



class DatasetConfig(BaseModel):
    """Dataset configuration"""
    name: Literal["MNIST", "CIFAR10", "FashionMNIST"]
    data_distribution: Literal["iid", "non_iid"] = "iid"
    train_test_split: float = Field(gt=0, lt=1)
    batch_size: int = Field(gt=0)


class ModelConfig(BaseModel):
    """Model configuration"""
    name: str
    input_shape: List[int]
    num_classes: int = Field(gt=0)


class TrainingConfig(BaseModel):
    """Training parameters"""
    optimizer: Literal["sgd", "adam"]
    learning_rate: float = Field(gt=0)
    momentum: float = Field(ge=0, le=1, default=0.9)
    weight_decay: float = Field(ge=0, default=0.0001)


class StrategyConfig(BaseModel):
    """Layer-wise strategy configuration"""
    type: Literal["all_layers", "first_n_layers", "last_n_layers", "random_layers"]
    num_layers: Optional[int] = None


class NetworkConfig(BaseModel):
    """Network constraints simulation"""
    simulate_constraints: bool = True
    latency_ms: float = Field(ge=0)
    latency_std_ms: float = Field(ge=0)
    bandwidth_mbps: float = Field(gt=0)
    packet_loss_rate: float = Field(ge=0, le=1)
    jitter_ms: float = Field(ge=0)


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    enabled: bool = True
    metrics_interval_sec: int = Field(gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ServerConfig(BaseModel):
    """Server configuration"""
    host: str = "localhost"
    port: int = Field(gt=0, lt=65536)
    aggregation_method: Literal["fedavg"] = "fedavg"


class ClientConfig(BaseModel):
    """Client configuration"""
    timeout_sec: int = Field(gt=0)
    max_retries: int = Field(ge=0)


class PathsConfig(BaseModel):
    """Paths configuration"""
    data_dir: str = "./data"
    models_dir: str = "./models"
    logs_dir: str = "./logs"
    results_dir: str = "./results"


class Config(BaseModel):
    """Main configuration class"""
    experiment: ExperimentConfig
    federated_learning: FederatedLearningConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    strategy: StrategyConfig
    network: NetworkConfig
    monitoring: MonitoringConfig
    server: ServerConfig
    client: ClientConfig
    paths: PathsConfig
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.paths.data_dir,
            self.paths.models_dir,
            self.paths.logs_dir,
            self.paths.results_dir
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directories created/verified")


def load_config(config_path: str = "config/config.yaml") -> Config:
    """
    Load and validate configuration from YAML file
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config: Validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Validate and create Config object
    config = Config(**config_dict)
    
    # Create necessary directories
    config.create_directories()
    
    print(f"✅ Configuration loaded successfully from {config_path}")
    print(f"   Experiment: {config.experiment.name}")
    print(f"   Clients: {config.federated_learning.num_clients}")
    print(f"   Rounds: {config.federated_learning.num_rounds}")
    print(f"   Dataset: {config.dataset.name}")
    print(f"   Model: {config.model.name}")
    print(f"   Strategy: {config.strategy.type}")
    
    return config


# Test function
if __name__ == "__main__":
    """Test the configuration parser"""
    try:
        config = load_config("config/config.yaml")
        print("\n✅ Configuration validation successful!")
        print(f"\n📊 Full configuration:")
        print(config.model_dump_json(indent=2))
    except Exception as e:
        print(f"\n❌ Configuration validation failed: {e}")