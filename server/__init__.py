"""
Server package for XFL-RPiLab
"""

from .strategy import FedAvg, create_aggregation_strategy
from .metrics import ServerMetricsCollector
from .server import FLServer, create_server, run_server
from .utils import setup_logging, validate_config

__all__ = [
    'FedAvg',
    'create_aggregation_strategy',
    'ServerMetricsCollector',
    'FLServer',
    'create_server',
    'run_server',
    'setup_logging',
    'validate_config'
]