"""
Server API endpoints for real Raspberry Pi clients
Add these routes to server.py to enable physical client support
"""

from flask import Blueprint, request, jsonify
import time
import threading

# Create blueprint
physical_client_bp = Blueprint('physical_clients', __name__, url_prefix='/api/physical')

# Global storage (will be imported from server)
_physical_client_manager = None
_server_metrics_collector = None


def init_physical_client_routes(app, physical_manager, metrics_collector):
    """Initialize physical client routes with required dependencies"""
    global _physical_client_manager, _server_metrics_collector
    _physical_client_manager = physical_manager
    _server_metrics_collector = metrics_collector
    
    # Register blueprint
    app.register_blueprint(physical_client_bp)
    
    print("✅ Physical client routes registered")


@physical_client_bp.route('/physical/register', methods=['POST'])
def register_physical_client():
    """
    Register a new physical Raspberry Pi client
    
    Request body:
    {
        "client_id": int,
        "ip_address": str,
        "hostname": str (optional),
        "username": str (optional)
    }
    """
    try:
        data = request.get_json()
        
        client_id = data.get('client_id')
        ip_address = data.get('ip_address')
        hostname = data.get('hostname', '')
        username = data.get('username', 'pi')
        
        if client_id is None or not ip_address:
            return jsonify({'error': 'client_id and ip_address required'}), 400
        
        success = _physical_client_manager.register_client(
            client_id=client_id,
            ip_address=ip_address,
            hostname=hostname,
            username=username
        )
        
        if success:
            return jsonify({
                'status': 'registered',
                'client_id': client_id,
                'ip_address': ip_address
            }), 200
        else:
            return jsonify({'error': 'Registration failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@physical_client_bp.route('/physical/unregister/<int:client_id>', methods=['POST'])
def unregister_physical_client(client_id):
    """Unregister a physical client"""
    try:
        success = _physical_client_manager.unregister_client(client_id)
        
        if success:
            return jsonify({'status': 'unregistered', 'client_id': client_id}), 200
        else:
            return jsonify({'error': 'Client not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@physical_client_bp.route('/physical/status/<int:client_id>', methods=['GET'])
def get_physical_client_status(client_id):
    """Get status of a specific physical client"""
    try:
        client_info = _physical_client_manager.get_client_info(client_id)
        
        if client_info:
            return jsonify(client_info), 200
        else:
            return jsonify({'error': 'Client not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@physical_client_bp.route('/physical/clients', methods=['GET'])
def get_all_physical_clients():
    """Get all registered physical clients"""
    try:
        clients = _physical_client_manager.get_all_clients()
        return jsonify({'clients': clients, 'count': len(clients)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@physical_client_bp.route('/physical/active', methods=['GET'])
def get_active_physical_clients():
    """Get list of active physical clients"""
    try:
        active = _physical_client_manager.get_active_clients()
        return jsonify({'active_clients': active, 'count': len(active)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@physical_client_bp.route('/physical/<int:client_id>/metrics', methods=['POST'])
def receive_physical_client_metrics(client_id):
    """
    Receive metrics from a physical Raspberry Pi client
    
    Request body: Full metrics from RealHardwareMetricsCollector
    """
    try:
        metrics = request.get_json()
        
        # Update client metrics
        _physical_client_manager.update_client_metrics(client_id, metrics)
        
        # Record in server metrics collector
        if _server_metrics_collector:
            _server_metrics_collector.record_client_update(
                client_id=client_id,
                round_number=metrics.get('round_number', 0),
                metrics=metrics
            )
        
        return jsonify({'status': 'metrics_received'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@physical_client_bp.route('/physical/<int:client_id>/latency', methods=['GET'])
def measure_client_latency(client_id):
    """Measure latency to a specific client"""
    try:
        quality = _physical_client_manager.measure_connection_quality(client_id)
        return jsonify(quality), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@physical_client_bp.route('/physical/network-stats', methods=['GET'])
def get_server_network_stats():
    """Get server network statistics"""
    try:
        if _server_metrics_collector:
            stats = _server_metrics_collector.get_network_stats()
            return jsonify(stats), 200
        else:
            return jsonify({'error': 'Metrics collector not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@physical_client_bp.route('/physical/<int:client_id>/latency-stats', methods=['GET'])
def get_client_latency_stats(client_id):
    """Get latency statistics for a client"""
    try:
        if _server_metrics_collector:
            stats = _server_metrics_collector.get_client_latency_stats(client_id)
            return jsonify(stats), 200
        else:
            return jsonify({'error': 'Metrics collector not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Additional utility endpoints
@physical_client_bp.route('/physical/heartbeat', methods=['POST'])
def client_heartbeat():
    """
    Client heartbeat endpoint - clients should call this periodically
    
    Request body:
    {
        "client_id": int,
        "status": str ("connected", "training", "idle"),
        "round_number": int (optional)
    }
    """
    try:
        data = request.get_json()
        
        client_id = data.get('client_id')
        status = data.get('status', 'connected')
        round_number = data.get('round_number', 0)
        
        if client_id is None:
            return jsonify({'error': 'client_id required'}), 400
        
        _physical_client_manager.update_client_status(
            client_id=client_id,
            status=status,
            round_number=round_number
        )
        
        return jsonify({'status': 'heartbeat_received', 'timestamp': time.time()}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@physical_client_bp.route('/physical/config', methods=['GET'])
def get_client_config():
    """
    Get configuration for physical clients
    Clients should call this on startup to get their configuration
    """
    try:
        # This would typically come from the FL config
        config = {
            'server_url': request.host_url.rstrip('/'),
            'metrics_interval': 1.0,
            'heartbeat_interval': 30,
            'connection_timeout': 60
        }
        return jsonify(config), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    # Test the routes
    print("Physical client API routes:")
    print("  POST /api/physical/register - Register a client")
    print("  POST /api/physical/unregister/<id> - Unregister a client")
    print("  GET  /api/physical/status/<id> - Get client status")
    print("  GET  /api/physical/clients - Get all clients")
    print("  GET  /api/physical/active - Get active clients")
    print("  POST /api/physical/<id>/metrics - Receive metrics")
    print("  GET  /api/physical/<id>/latency - Measure latency")
    print("  GET  /api/physical/network-stats - Server network stats")
    print("  GET  /api/physical/<id>/latency-stats - Client latency stats")
    print("  POST /api/physical/heartbeat - Client heartbeat")
    print("  GET  /api/physical/config - Get client config")