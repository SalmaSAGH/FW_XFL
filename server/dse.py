"""
Design Space Exploration (DSE) — Real federated training sweep
"""

import os
import glob
import time
from datetime import datetime
import yaml
import json
import hashlib
import gc
import traceback
import random
import threading
import torch
from itertools import product
from typing import Dict, List, Any

from .strategy import create_aggregation_strategy
from client.model import create_model_for_dataset, DATASET_CONFIG
from client.dataset import load_dataset, create_single_client_loader
from client.trainer import LocalTrainer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DSE_RESULTS_DIR = os.path.join(ROOT_DIR, 'results', 'dse')

DSE_JOB_STATUS: Dict[str, str] = {}
DSE_JOB_LOCK = threading.Lock()


def list_dse_sessions() -> List[Dict[str, Any]]:
    """List all saved DSE sessions."""
    if not os.path.exists(DSE_RESULTS_DIR):
        return []

    sessions = []
    for session_path in glob.glob(os.path.join(DSE_RESULTS_DIR, "dse_*")):
        if os.path.isdir(session_path):
            results_file = os.path.join(session_path, "results.json")
            if not os.path.exists(results_file):
                continue
            basename = os.path.basename(session_path)
            try:
                ts_str = basename[4:]
                try:
                    ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S_%f")
                except ValueError:
                    ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                config_dir = os.path.join(session_path, "temp_configs")
                num_configs = len(glob.glob(os.path.join(config_dir, "*.yaml"))) if os.path.exists(config_dir) else 0
                sessions.append({
                    "id": basename,
                    "timestamp": ts.isoformat(),
                    "num_configs": num_configs
                })
            except Exception:
                pass

    return sorted(sessions, key=lambda x: x["timestamp"], reverse=True)


def load_dse_session(session_id: str) -> Any:
    """Load a saved DSE session's results."""
    session_dir = os.path.join(DSE_RESULTS_DIR, session_id)
    if not os.path.exists(session_dir):
        return None

    results_file = os.path.join(session_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            return {"session_id": session_id, "results": json.load(f)}
    return {"session_id": session_id, "results": []}


def load_all_dse_results() -> List[Dict[str, Any]]:
    """Load and merge results from all DSE sessions."""
    all_results = []
    for session in list_dse_sessions():
        loaded = load_dse_session(session.get('id'))
        if loaded and isinstance(loaded.get('results'), list):
            all_results.extend(loaded['results'])
    return all_results


def reset_dse_data() -> bool:
    """Reset all DSE data by deleting the results directory."""
    try:
        if os.path.exists(DSE_RESULTS_DIR):
            import shutil
            shutil.rmtree(DSE_RESULTS_DIR)
        # Recreate the directory
        os.makedirs(DSE_RESULTS_DIR, exist_ok=True)
        # Clear job statuses
        global DSE_JOB_STATUS
        DSE_JOB_STATUS.clear()
        return True
    except Exception as e:
        print(f"Error resetting DSE data: {e}")
        return False


def _run_dse_background(sweep_config: Dict[str, Any], session_id: str):
    try:
        with DSE_JOB_LOCK:
            DSE_JOB_STATUS[session_id] = "running"

        run_dse_sweep(sweep_config, session_id=session_id)

        with DSE_JOB_LOCK:
            DSE_JOB_STATUS[session_id] = "completed"
    except Exception as e:
        with DSE_JOB_LOCK:
            DSE_JOB_STATUS[session_id] = "failed"
        print(f"[DSE] Background sweep {session_id} failed: {e}")
        traceback.print_exc()


def start_dse_sweep(sweep_config: Dict[str, Any]) -> str:
    session_id = f"dse_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    with DSE_JOB_LOCK:
        DSE_JOB_STATUS[session_id] = "queued"

    thread = threading.Thread(target=_run_dse_background, args=(sweep_config, session_id), daemon=True)
    thread.start()
    return session_id


def get_dse_job_status(session_id: str) -> str:
    with DSE_JOB_LOCK:
        return DSE_JOB_STATUS.get(session_id, "unknown")


def get_dse_job_progress(session_id: str) -> Dict[str, Any]:
    session_dir = os.path.join(DSE_RESULTS_DIR, session_id)
    if not os.path.exists(session_dir):
        return {
            "session_id": session_id,
            "status": get_dse_job_status(session_id),
            "completed_configs": 0,
            "total_configs": 0,
            "best_accuracy": 0.0
        }

    temp_configs_dir = os.path.join(session_dir, 'temp_configs')
    total_configs = len(glob.glob(os.path.join(temp_configs_dir, '*.yaml')))

    results_file = os.path.join(session_dir, 'results.json')
    completed_configs = 0
    best_accuracy = 0.0
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            completed_configs = len(results)
            best_accuracy = max(
                (float(r.get('metrics', {}).get('final_accuracy', 0.0)) for r in results),
                default=0.0
            )
        except Exception:
            completed_configs = 0
            best_accuracy = 0.0

    return {
        "session_id": session_id,
        "status": get_dse_job_status(session_id),
        "completed_configs": completed_configs,
        "total_configs": total_configs,
        "best_accuracy": round(best_accuracy, 2)
    }


def _evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> (float, float):
    model.eval()
    device = next(model.parameters()).device
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def _build_sweep_configs(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    params = sweep_config.get('params', {}) or {}
    base_config = sweep_config.get('baseConfig', {}) or {}
    max_configs = int(sweep_config.get('maxConfigs', 6))

    param_names = []
    param_values = []
    for name, values in params.items():
        if values is None:
            continue
        param_names.append(name)
        if isinstance(values, (list, tuple)):
            param_values.append(list(values))
        else:
            param_values.append([values])

    if not param_names:
        param_names = ['learningRate', 'localEpochs', 'batchSize']
        param_values = [[0.001, 0.01, 0.1], [1, 2, 3], [32, 64]]

    # Build cartesian product of parameter values
    all_combinations = []
    for values in product(*param_values):
        combo = dict(base_config)
        combo.update({name: value for name, value in zip(param_names, values)})
        all_combinations.append(combo)

    search_strategy = sweep_config.get('searchStrategy', 'grid').lower()
    if search_strategy == 'random':
        rng = random.Random(int(sweep_config.get('randomSeed', 42)))
        if len(all_combinations) <= max_configs:
            selected = all_combinations
        else:
            selected = rng.sample(all_combinations, max_configs)
    else:
        if len(all_combinations) <= max_configs:
            selected = all_combinations
        else:
            step = len(all_combinations) / max_configs
            indices = sorted({min(len(all_combinations) - 1, int(round(i * step))) for i in range(max_configs)})
            selected = [all_combinations[i] for i in indices]

    return selected


def _run_federated_config(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset_name = config.get('dataset', 'MNIST')
    batch_size = max(1, int(config.get('batchSize', 64)))
    clients_per_round = max(1, int(config.get('clientsPerRound', 3)))
    num_rounds = max(1, int(config.get('numRounds', 1)))
    local_epochs = max(1, int(config.get('localEpochs', 1)))
    learning_rate = float(config.get('learningRate', 0.01))
    strategy_name = config.get('strategy', 'fedavg')
    xfl_strategy = config.get('xflStrategy', 'all_layers')
    xfl_param = int(config.get('xflParam', 3))
    distribution = config.get('dataDistribution', 'iid')
    total_clients = max(clients_per_round, int(config.get('numClients', clients_per_round)))

    if total_clients < clients_per_round:
        total_clients = clients_per_round

    strategy = create_aggregation_strategy(strategy_name, xfl_strategy=xfl_strategy, xfl_param=xfl_param)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = create_model_for_dataset(dataset_name)
    global_model.to(device)
    global_model.train()

    # Load full test set for final evaluation
    test_dataset = load_dataset(dataset_name, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    rng = random.Random(42)
    strategy_meta = strategy.get_xfl_info()

    for round_idx in range(1, num_rounds + 1):
        client_weights = []
        client_samples = []

        selected_clients = rng.sample(list(range(total_clients)), clients_per_round)
        print(f"[DSE] Round {round_idx}/{num_rounds} - selected clients: {selected_clients}")

        for client_id in selected_clients:
            client_model = create_model_for_dataset(dataset_name)
            client_model.load_state_dict(global_model.state_dict())
            client_model.to(device)

            trainer = LocalTrainer(
                client_model,
                learning_rate=learning_rate,
                optimizer_name='sgd'
            )

            client_loader = create_single_client_loader(
                dataset_name,
                client_id,
                total_clients,
                batch_size=batch_size,
                distribution=distribution,
                data_dir='./data'
            )

            trainer.train(client_loader, num_epochs=local_epochs)
            client_state = {name: tensor.cpu().clone() for name, tensor in trainer.get_model_weights().items()}
            client_weights.append(client_state)
            client_samples.append(len(client_loader.dataset))

        if len(client_weights) == 0:
            raise RuntimeError('No client updates collected during federated sweep')

        aggregated_weights = strategy.aggregate(client_weights, client_samples)
        current_state = global_model.state_dict()
        current_state.update(aggregated_weights)
        global_model.load_state_dict(current_state, strict=False)

    final_loss, final_accuracy = _evaluate_model(global_model, test_loader)

    return {
        'final_accuracy': round(final_accuracy, 2),
        'final_loss': round(final_loss, 4),
        'total_time': round(0.0, 1)
    }


def run_dse_sweep(sweep_config: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
    print('[DSE] Starting real federated sweep...')
    if session_id is None:
        session_id = f"dse_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = os.path.join(DSE_RESULTS_DIR, session_id)
    config_dir = os.path.join(session_dir, 'temp_configs')

    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(DSE_RESULTS_DIR, exist_ok=True)

    sweep_configs = _build_sweep_configs(sweep_config)
    results = []

    results_file = os.path.join(session_dir, 'results.json')
    for idx, config in enumerate(sweep_configs):
        config = dict(config)
        if 'dataset' not in config or config['dataset'] is None:
            config['dataset'] = sweep_config.get('dataset', 'MNIST')
        if 'numRounds' not in config or config['numRounds'] is None:
            config['numRounds'] = int(sweep_config.get('numShortRounds', 1))

        config_id = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        config_yaml = os.path.join(config_dir, f"config_{config_id}.yaml")
        with open(config_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

        try:
            start_time = time.time()
            stats = _run_federated_config(config)
            elapsed = time.time() - start_time
            stats['total_time'] = round(elapsed, 1)

            results.append({
                'config_id': config_id,
                'config': config,
                'metrics': stats
            })
            print(f"[DSE] Completed config {config_id}: acc={stats['final_accuracy']}%, loss={stats['final_loss']}, time={stats['total_time']}s")

        except Exception as e:
            print(f"[DSE] Config {config_id} failed: {e}")
            traceback.print_exc()
            results.append({
                'config_id': config_id,
                'config': config,
                'metrics': {
                    'final_accuracy': 0.0,
                    'final_loss': 0.0,
                    'total_time': 0.0
                }
            })

        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"[DSE] Failed to save intermediate results: {e}")

        gc.collect()

    results_file = os.path.join(session_dir, 'results.json')
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"[DSE] Saved results to {results_file}")
    except Exception as e:
        print(f"[DSE] Failed to save results: {e}")

    return {
        'session_id': session_id,
        'num_configs': len(results),
        'results': results
    }
