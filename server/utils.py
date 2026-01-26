"""
Utility functions for the FL server
"""

import logging
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"server_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def parse_client_list(client_str: str) -> list:
    """
    Parse client list from string
    
    Args:
        client_str: String like "0,1,2,3,4"
        
    Returns:
        List of client IDs
    """
    if not client_str:
        return []
    
    return [int(x.strip()) for x in client_str.split(',')]


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human-readable string
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def validate_config(config):
    """
    Validate configuration parameters
    
    Args:
        config: Configuration object
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate FL parameters
    if config.federated_learning.clients_per_round > config.federated_learning.num_clients:
        raise ValueError(
            f"clients_per_round ({config.federated_learning.clients_per_round}) "
            f"cannot exceed num_clients ({config.federated_learning.num_clients})"
        )
    
    if config.federated_learning.num_rounds <= 0:
        raise ValueError("num_rounds must be positive")
    
    # Validate dataset
    valid_datasets = ["MNIST", "CIFAR10", "FashionMNIST"]
    if config.dataset.name not in valid_datasets:
        raise ValueError(f"Invalid dataset: {config.dataset.name}. Must be one of {valid_datasets}")
    
    # Validate model
    valid_models = ["SimpleCNN", "LeNet5"]
    if config.model.name not in valid_models:
        raise ValueError(f"Invalid model: {config.model.name}. Must be one of {valid_models}")
    
    print("✅ Configuration validation passed")


# Test function
if __name__ == "__main__":
    """Test utility functions"""
    print("🧪 Testing server utilities...\n")
    
    # Test logging
    print("📊 Testing logging...")
    logger = setup_logging(log_dir="logs", log_level="INFO")
    logger.info("This is a test log message")
    logger.warning("This is a test warning")
    
    # Test client list parsing
    print("\n📊 Testing client list parsing...")
    clients = parse_client_list("0,1,2,3,4")
    print(f"   Parsed: {clients}")
    
    # Test byte formatting
    print("\n📊 Testing byte formatting...")
    test_bytes = [100, 1024, 1024*1024, 1024*1024*1024]
    for b in test_bytes:
        print(f"   {b} bytes = {format_bytes(b)}")
    
    # Test time formatting
    print("\n📊 Testing time formatting...")
    test_times = [30, 90, 3600, 7265]
    for t in test_times:
        print(f"   {t} seconds = {format_time(t)}")
    
    print("\n✅ All utility tests passed!")