import hashlib
import json
import pytest
from server.server import app
import server.server as server_module


class DummyAggregationStrategy:
    def get_xfl_info(self):
        return {
            "strategy": "all_layers",
            "param": 3,
            "sparsification_threshold": 0.01,
            "quantization_bits": 8
        }


class DummyFLServer:
    def __init__(self):
        self.current_round = 0
        self.current_dataset_name = "MNIST"
        self.current_model_name = "TinyCNN"
        self.aggregation_strategy = DummyAggregationStrategy()

    def get_server_status(self):
        return {"status": "ok", "round_in_progress": False}

    def get_global_model(self):
        return {}

    def set_xfl_strategy(self, strategy, param):
        return {"status": "success", "strategy": strategy, "param": param}


def test_get_config_endpoint_returns_config():
    """Test endpoint /api/config renvoie la configuration active."""
    with app.test_client() as client:
        response = client.get('/api/config')

    assert response.status_code == 200
    assert "config" in response.json
    assert isinstance(response.json["config"], dict)


def test_register_endpoint_creates_user(monkeypatch):
    """Test endpoint /api/register crée un nouvel utilisateur lorsque les données sont valides."""
    monkeypatch.setattr(server_module, "get_user_from_db", lambda username: None)
    monkeypatch.setattr(server_module, "create_user_in_db", lambda username, password: True)

    with app.test_client() as client:
        response = client.post('/api/register', json={"username": "student", "password": "mypassword"})

    assert response.status_code == 201
    assert response.json["status"] == "success"
    assert "message" in response.json


def test_register_endpoint_requires_credentials():
    """Test endpoint /api/register échoue lorsque le nom d'utilisateur ou le mot de passe est manquant."""
    with app.test_client() as client:
        response = client.post('/api/register', json={"username": "student"})

    assert response.status_code == 400
    assert "error" in response.json


def test_login_and_verify_token_endpoints(monkeypatch):
    """Test endpoints /api/login et /api/verify_token avec session simulée."""
    user_password_hash = hashlib.sha256("mypassword".encode()).hexdigest()

    def fake_get_user_from_db(username):
        return {"password": user_password_hash}

    def fake_create_session_in_db(token, username):
        return True

    def fake_get_session_from_db(token):
        return {"username": "student"}

    monkeypatch.setattr(server_module, "get_user_from_db", fake_get_user_from_db)
    monkeypatch.setattr(server_module, "create_session_in_db", fake_create_session_in_db)
    monkeypatch.setattr(server_module, "get_session_from_db", fake_get_session_from_db)

    with app.test_client() as client:
        login_response = client.post('/api/login', json={"username": "student", "password": "mypassword"})
        assert login_response.status_code == 200
        assert login_response.json["status"] == "success"
        token = login_response.json["token"]

        verify_response = client.post('/api/verify_token', json={"token": token})
        assert verify_response.status_code == 200
        assert verify_response.json["status"] == "valid"
        assert verify_response.json["username"] == "student"


def test_xfl_get_strategy_endpoint_returns_strategy(monkeypatch):
    """Test endpoint /api/xfl/get_strategy retourne les informations de stratégie XFL."""
    server_module.fl_server = DummyFLServer()

    with app.test_client() as client:
        response = client.get('/api/xfl/get_strategy')

    assert response.status_code == 200
    assert response.json["strategy"] == "all_layers"
    assert response.json["param"] == 3
