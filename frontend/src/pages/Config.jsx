import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { setXflStrategy, startRound, saveConfig, getConfig } from '../services/api';

function Config() {
  const navigate = useNavigate();
  const [config, setConfig] = useState({
    // FL Settings
    numClients: 40,
    clientsPerRound: 5,
    numRounds: 50,
    // Data Parameters
    dataset: 'MNIST',
    model: 'SimpleCNN',
    dataDistribution: 'iid',
    // Algorithmic Parameters
    strategy: 'all_layers',
    xflParam: 3,
    // Training Parameters
    localEpochs: 2,
    batchSize: 512,
    learningRate: 0.01,
    // Network Parameters
    networkLatency: 0,
    networkBandwidth: 10,
    networkPacketLoss: 0,
    // System Parameters
    cpuLimit: 100,
    ramLimit: 2048,
  });
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });

  // Load config from server on mount
  useEffect(() => {
    const loadConfig = async () => {
      try {
        const response = await getConfig();
        if (response.data && response.data.config) {
          setConfig(prev => ({
            ...prev,
            ...response.data.config
          }));
        }
      } catch (error) {
        console.log('Using default configuration');
      }
    };
    loadConfig();
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('username');
    navigate('/login');
  };

  const handleConfigChange = (field, value) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const handleConfirm = async () => {
    setLoading(true);
    setMessage({ type: '', text: '' });
    
    try {
      // Save full configuration to server
      await saveConfig(config);
      setMessage({ type: 'success', text: 'Configuration confirmed and saved!' });
    } catch (error) {
      setMessage({ type: 'error', text: 'Error confirming configuration: ' + error.message });
    }
    
    setLoading(false);
  };

  const handleStartRound = async () => {
    setLoading(true);
    setMessage({ type: '', text: '' });
    
    try {
      // Start a round
      const response = await startRound();
      if (response.data.status === 'started') {
        setMessage({ type: 'success', text: `Round ${response.data.round} started successfully!` });
        // Navigate to dashboard after a short delay
        setTimeout(() => {
          navigate('/dashboard');
        }, 1500);
      } else {
        setMessage({ type: 'warning', text: response.data.message || 'Could not start round' });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Error starting round: ' + error.message });
    }
    
    setLoading(false);
  };

  // Strategies for algorithmic parameters
  const strategies = [
    { value: 'all_layers', label: 'All Layers (FedAvg)' },
    { value: 'xfl_cyclic', label: 'XFL - Cyclic' },
    { value: 'xfl_sparsification', label: 'XFL - Sparsification' },
    { value: 'xfl_quantization', label: 'XFL - Quantization' },
  ];

  // Dataset to model mapping
  const datasetModelMap = {
    'MNIST': { model: 'SimpleCNN', numClasses: 10 },
    'FashionMNIST': { model: 'SimpleCNN', numClasses: 10 },
    'CIFAR10': { model: 'SimpleCNN', numClasses: 10 },
    'CIFAR100': { model: 'CIFAR100CNN', numClasses: 100 },
    'EMNIST': { model: 'EMNISTCNN', numClasses: 47 },
  };

  const datasets = [
    { value: 'MNIST', label: 'MNIST (10 classes, 28x28 grayscale)' },
    { value: 'FashionMNIST', label: 'FashionMNIST (10 classes, 28x28 grayscale)' },
    { value: 'CIFAR10', label: 'CIFAR10 (10 classes, 32x32 RGB)' },
    { value: 'CIFAR100', label: 'CIFAR100 (100 classes, 32x32 RGB)' },
    { value: 'EMNIST', label: 'EMNIST (47 classes, 28x28 grayscale)' },
  ];

  const models = [
    { value: 'SimpleCNN', label: 'SimpleCNN (MNIST, FashionMNIST, CIFAR10)' },
    { value: 'CIFAR100CNN', label: 'CIFAR100CNN (CIFAR-100)' },
    { value: 'EMNISTCNN', label: 'EMNISTCNN (EMNIST)' },
    { value: 'LeNet5', label: 'LeNet5 (Legacy)' },
  ];

  // Network parameter options
  const latencyOptions = [
    { value: 0, label: '0ms (No delay)' },
    { value: 50, label: '50ms (LAN)' },
    { value: 100, label: '100ms (WAN)' },
    { value: 200, label: '200ms (Slow WAN)' },
  ];

  const bandwidthOptions = [
    { value: 10, label: '10 Mbps (Fast)' },
    { value: 5, label: '5 Mbps (Medium)' },
    { value: 1, label: '1 Mbps (Slow)' },
  ];

  const packetLossOptions = [
    { value: 0, label: '0% (Perfect)' },
    { value: 1, label: '1% (Low)' },
    { value: 5, label: '5% (High)' },
  ];

  // System parameter options
  const cpuLimitOptions = [
    { value: 100, label: '100% (No limit)' },
    { value: 75, label: '75%' },
    { value: 50, label: '50%' },
    { value: 25, label: '25%' },
  ];

  const ramLimitOptions = [
    { value: 2048, label: '2048 MB (2 GB)' },
    { value: 1024, label: '1024 MB (1 GB)' },
    { value: 512, label: '512 MB' },
    { value: 256, label: '256 MB' },
  ];

  return (
    <div className="page-container">
      {/* Header */}
      <header className="header">
        <h1>🔷 XFL-RPiLab - Configuration</h1>
        <div className="header-nav">
          <a href="#" onClick={() => navigate('/config')} className="active">Config</a>
          <a href="#" onClick={() => navigate('/dashboard')}>Dashboard</a>
          <a href="#" onClick={() => navigate('/dse')}>DSE</a>
          <a href="#" onClick={() => navigate('/history')}>History</a>
          <button className="logout-btn" onClick={handleLogout}>Logout</button>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div style={{ maxWidth: '900px', margin: '0 auto' }}>
          {/* Message Display */}
          {message.text && (
            <div style={{ 
              padding: '15px', 
              marginBottom: '20px', 
              borderRadius: '4px',
              background: message.type === 'success' ? '#66bb6a' : message.type === 'error' ? '#ef5350' : '#ffa726',
              color: message.type === 'warning' ? '#1a1d2e' : 'white'
            }}>
              {message.text}
            </div>
          )}

          {/* ============================================ */}
          {/* CATEGORY 1: ALGORITHMIC PARAMETERS */}
          {/* ============================================ */}
          <div className="card" style={{ marginBottom: '20px', borderLeft: '4px solid #ab47bc' }}>
            <h2 className="panel-title">
              🔧 Algorithmic Parameters
              <span style={{ fontSize: '12px', fontWeight: 'normal', color: '#888', marginLeft: '10px' }}>
                (XFL Strategy)
              </span>
            </h2>
            
            <div className="grid-2">
              <div className="form-group">
                <label>Aggregation Strategy</label>
                <select
                  value={config.strategy}
                  onChange={(e) => handleConfigChange('strategy', e.target.value)}
                >
                  {strategies.map(s => (
                    <option key={s.value} value={s.value}>{s.label}</option>
                  ))}
                </select>
                <div style={{ marginTop: '5px', fontSize: '12px', color: '#888' }}>
                  Select the layer-wise aggregation strategy
                </div>
              </div>

              <div className="form-group">
                <label>XFL Parameter (Top-K)</label>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={config.xflParam}
                  onChange={(e) => handleConfigChange('xflParam', parseInt(e.target.value))}
                />
                <div style={{ marginTop: '5px', fontSize: '12px', color: '#888' }}>
                  Top-K layers for importance-based, layers for layerwise
                </div>
              </div>
            </div>
          </div>

          {/* ============================================ */}
          {/* CATEGORY 2: NETWORK PARAMETERS */}
          {/* ============================================ */}
          <div className="card" style={{ marginBottom: '20px', borderLeft: '4px solid #4fc3f7' }}>
            <h2 className="panel-title">
              🌐 Network Parameters
              <span style={{ fontSize: '12px', fontWeight: 'normal', color: '#888', marginLeft: '10px' }}>
                (Latency, Bandwidth, Packet Loss)
              </span>
            </h2>
            
            <div className="grid-3">
              <div className="form-group">
                <label>Latency</label>
                <select
                  value={config.networkLatency}
                  onChange={(e) => handleConfigChange('networkLatency', parseInt(e.target.value))}
                >
                  {latencyOptions.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Bandwidth</label>
                <select
                  value={config.networkBandwidth}
                  onChange={(e) => handleConfigChange('networkBandwidth', parseInt(e.target.value))}
                >
                  {bandwidthOptions.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Packet Loss</label>
                <select
                  value={config.networkPacketLoss}
                  onChange={(e) => handleConfigChange('networkPacketLoss', parseInt(e.target.value))}
                >
                  {packetLossOptions.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* ============================================ */}
          {/* CATEGORY 3: SYSTEM PARAMETERS */}
          {/* ============================================ */}
          <div className="card" style={{ marginBottom: '20px', borderLeft: '4px solid #ffa726' }}>
            <h2 className="panel-title">
              💻 System Parameters
              <span style={{ fontSize: '12px', fontWeight: 'normal', color: '#888', marginLeft: '10px' }}>
                (CPU, RAM, Clients)
              </span>
            </h2>
            
            <div className="grid-2">
              <div className="form-group">
                <label>CPU Limitation</label>
                <select
                  value={config.cpuLimit}
                  onChange={(e) => handleConfigChange('cpuLimit', parseInt(e.target.value))}
                >
                  {cpuLimitOptions.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
                <div style={{ marginTop: '5px', fontSize: '12px', color: '#888' }}>
                  Maximum CPU usage per client
                </div>
              </div>

              <div className="form-group">
                <label>RAM Limitation</label>
                <select
                  value={config.ramLimit}
                  onChange={(e) => handleConfigChange('ramLimit', parseInt(e.target.value))}
                >
                  {ramLimitOptions.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
                <div style={{ marginTop: '5px', fontSize: '12px', color: '#888' }}>
                  Maximum RAM usage per client
                </div>
              </div>

              <div className="form-group">
                <label>Total Number of Clients</label>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={config.numClients}
                  onChange={(e) => handleConfigChange('numClients', parseInt(e.target.value))}
                />
                <div style={{ marginTop: '5px', fontSize: '12px', color: '#888' }}>
                  Total clients available in the system (1-100)
                </div>
              </div>

              <div className="form-group">
                <label>Clients per Round</label>
                <input
                  type="number"
                  min="1"
                  max="40"
                  value={config.clientsPerRound}
                  onChange={(e) => handleConfigChange('clientsPerRound', parseInt(e.target.value))}
                />
                <div style={{ marginTop: '5px', fontSize: '12px', color: '#888' }}>
                  Number of clients to use in each training round (1-40)
                </div>
              </div>
            </div>
          </div>

          {/* ============================================ */}
          {/* CATEGORY 4: DATA PARAMETERS */}
          {/* ============================================ */}
          <div className="card" style={{ marginBottom: '20px', borderLeft: '4px solid #66bb6a' }}>
            <h2 className="panel-title">
              📊 Data Parameters
              <span style={{ fontSize: '12px', fontWeight: 'normal', color: '#888', marginLeft: '10px' }}>
                (Dataset, Distribution)
              </span>
            </h2>
            
            <div className="grid-2">
              <div className="form-group">
                <label>Dataset</label>
                <select
                  value={config.dataset}
                  onChange={(e) => handleConfigChange('dataset', e.target.value)}
                >
                  {datasets.map(ds => (
                    <option key={ds.value} value={ds.value}>{ds.label}</option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Data Distribution</label>
                <select
                  value={config.dataDistribution}
                  onChange={(e) => handleConfigChange('dataDistribution', e.target.value)}
                >
                  <option value="iid">IID (Independent and Identically Distributed)</option>
                  <option value="non_iid">Non-IID</option>
                </select>
              </div>
            </div>
          </div>

          {/* ============================================ */}
          {/* TRAINING PARAMETERS */}
          {/* ============================================ */}
          <div className="card" style={{ marginBottom: '20px' }}>
            <h2 className="panel-title">
              ⚙️ Training Parameters
              <span style={{ fontSize: '12px', fontWeight: 'normal', color: '#888', marginLeft: '10px' }}>
                (Rounds, Epochs, Learning Rate)
              </span>
            </h2>
            
            <div className="grid-3">
              <div className="form-group">
                <label>Number of Rounds</label>
                <input
                  type="number"
                  min="1"
                  max="500"
                  value={config.numRounds}
                  onChange={(e) => handleConfigChange('numRounds', parseInt(e.target.value))}
                />
              </div>

              <div className="form-group">
                <label>Local Epochs</label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={config.localEpochs}
                  onChange={(e) => handleConfigChange('localEpochs', parseInt(e.target.value))}
                />
              </div>

              <div className="form-group">
                <label>Batch Size</label>
                <input
                  type="number"
                  min="8"
                  max="2048"
                  value={config.batchSize}
                  onChange={(e) => handleConfigChange('batchSize', parseInt(e.target.value))}
                />
              </div>
            </div>

            <div className="form-group" style={{ marginTop: '15px' }}>
              <label>Learning Rate</label>
              <input
                type="number"
                step="0.001"
                min="0.001"
                max="1"
                value={config.learningRate}
                onChange={(e) => handleConfigChange('learningRate', parseFloat(e.target.value))}
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="card">
            <h2 className="panel-title">Actions</h2>
            
            <div style={{ display: 'flex', gap: '15px', flexWrap: 'wrap' }}>
              <button 
                className="btn btn-primary"
                onClick={handleConfirm}
                disabled={loading}
              >
                ✓ Confirm Configuration
              </button>
              
              <button 
                className="btn btn-success"
                onClick={handleStartRound}
                disabled={loading}
              >
                ▶ Start Round
              </button>
              
              <button 
                className="btn"
                onClick={() => navigate('/dashboard')}
                style={{ background: '#252942', border: '1px solid #2d3348', color: '#e0e0e0' }}
              >
                View Dashboard
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default Config;

