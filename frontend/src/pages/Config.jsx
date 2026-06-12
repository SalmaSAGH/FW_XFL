import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { setXflStrategy, startExperiment, saveConfig, getConfig } from '../services/api';

function Config() {
  const navigate = useNavigate();
  const [config, setConfig] = useState({
    // FL Settings
    numClients: 40,
    clientsPerRound: 5,
    numRounds: 50,
    experimentRounds: 5,
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
    networkSimulationShare: 0.5,
    simulateConstraints: false,
    // System Parameters
    cpuLimit: 100,
    ramLimit: 2048,
    systemConstraintsEnabled: false,
    systemConstraintsShare: 1.0,
    // Round Timeout
    roundTimeout: 300,
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
    setConfig(prev => {
      const newConfig = { ...prev, [field]: value };
      
      // Auto-update model and batch_size when dataset changes
      if (field === 'dataset' && datasetModelMap[value]) {
        newConfig.model = datasetModelMap[value].model;
        // Set appropriate batch size and epochs for different datasets
        if (value.includes('CIFAR')) {
          newConfig.batchSize = 64;  // Smaller batch for CIFAR on RPi
          newConfig.localEpochs = 1;  // Fewer epochs for CIFAR
        } else {
          newConfig.batchSize = 512;  // Larger batch for MNIST/FashionMNIST
          newConfig.localEpochs = 2;  // More epochs for simpler datasets
        }
      }
      
      return newConfig;
    });
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

  const handleStartExperiment = async () => {
    setLoading(true);
    setMessage({ type: '', text: '' });
    
    try {
      // Start an experiment with multiple automatic rounds
      const response = await startExperiment(config.experimentRounds);
      if (response.data.status === 'experiment_started') {
        setMessage({ type: 'success', text: `Experiment started with ${response.data.rounds_requested} rounds scheduled automatically.` });
        setTimeout(() => {
          navigate('/dashboard');
        }, 1500);
      } else {
        setMessage({ type: 'warning', text: response.data.message || 'Could not start experiment' });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Error starting experiment: ' + error.message });
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
    'MNIST': { model: 'TinyCNN', numClasses: 10 },
    'FashionMNIST': { model: 'MicroLeNet', numClasses: 10 },
    'CIFAR10': { model: 'TinyCNN', numClasses: 10 },
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
    { value: 'SimpleCNN', label: 'SimpleCNN (Basic CNN)' },
    { value: 'LeNet5', label: 'LeNet5 (Classic CNN)' },
    { value: 'TinyCNN', label: 'TinyCNN (Ultra-lightweight for RPi)' },
    { value: 'MicroLeNet', label: 'MicroLeNet (Compact LeNet-style)' },
    { value: 'DepthwiseCNN', label: 'DepthwiseCNN (Efficient separable)' },
    { value: 'MobileNetV2', label: 'MobileNetV2 (Mobile-optimized)' },
    { value: 'ResNet8', label: 'ResNet8 (Small ResNet)' },
    { value: 'ShuffleNetV2', label: 'ShuffleNetV2 (Very efficient)' },
    { value: 'CIFAR100CNN', label: 'CIFAR100CNN (CIFAR-100 specialized)' },
    { value: 'EMNISTCNN', label: 'EMNISTCNN (EMNIST specialized)' },
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
    { value: 0.0, label: '0% (Perfect)' },
    { value: 0.01, label: '1% (Low)' },
    { value: 0.05, label: '5% (High)' },
  ];

  // System parameter options
  const cpuLimitOptions = [
    { value: 999999, label: '∞ (No Alert)' },
    { value: 200, label: '200%' },
    { value: 150, label: '150%' },
    { value: 100, label: '100%' },
    { value: 75, label: '75%' },
    { value: 50, label: '50%' },
    { value: 25, label: '25%' },
  ];

  const ramLimitOptions = [
    { value: 999999, label: '∞ (No Limit)' },
    { value: 4096, label: '4096 MB (4 GB)' },
    { value: 2048, label: '2048 MB (2 GB)' },
    { value: 1024, label: '1024 MB (1 GB)' },
    { value: 512, label: '512 MB' },
    { value: 256, label: '256 MB' },
  ];

  const constraintShareOptions = [
    { value: 1.0, label: '100% (All clients)' },
    { value: 0.75, label: '75% (3 out of 4)' },
    { value: 0.5, label: '50% (Half)' },
    { value: 0.25, label: '25% (Quarter)' },
  ];

  return (
    <div className="page-container">
      {/* Header */}
      <header className="header">
        <h1>🔷 XFL-FW - Configuration</h1>
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
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '20px', flexWrap: 'wrap' }}>
              <div>
                <h2 className="panel-title" style={{ marginBottom: '6px' }}>
                  🌐 Network Parameters
                </h2>
                <div style={{ fontSize: '12px', fontWeight: '400', color: '#666' }}>
                  Configure network delay, bandwidth and packet loss for training rounds.
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px', borderRadius: '12px', background: '#e3f2fd', border: '1px solid #b3e5fc' }}>
                <input
                  type="checkbox"
                  checked={config.simulateConstraints}
                  onChange={(e) => handleConfigChange('simulateConstraints', e.target.checked)}
                  style={{ width: '18px', height: '18px', accentColor: '#0288d1' }}
                />
                <div>
                  <div style={{ fontWeight: '700', color: '#0d47a1' }}>Enable Network Simulation</div>
                  <div style={{ fontSize: '12px', color: '#555', marginTop: '2px' }}>
                    Toggle network simulation on or off for the next training round.
                  </div>
                </div>
              </div>
            </div>
            
            <div className="grid-2" style={{ marginTop: '24px' }}>
              <div className="form-group">
                <label>Latency</label>
                <select
                  value={config.networkLatency}
                  disabled={!config.simulateConstraints}
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
                  disabled={!config.simulateConstraints}
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
                  disabled={!config.simulateConstraints}
                  onChange={(e) => handleConfigChange('networkPacketLoss', parseFloat(e.target.value))}
                >
                  {packetLossOptions.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Network Simulation Share</label>
                <select
                  value={config.networkSimulationShare}
                  disabled={!config.simulateConstraints}
                  onChange={(e) => handleConfigChange('networkSimulationShare', parseFloat(e.target.value))}
                >
                  {constraintShareOptions.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
                <div style={{ marginTop: '5px', fontSize: '12px', color: '#888' }}>
                  What fraction of clients get network simulation per round
                </div>
              </div>
            </div>
          </div>

          {/* ============================================ */}
          {/* CATEGORY 3: SYSTEM PARAMETERS */}
          {/* ============================================ */}
          <div className="card" style={{ marginBottom: '20px', borderLeft: '4px solid #ffa726' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '20px', flexWrap: 'wrap' }}>
              <div>
                <h2 className="panel-title" style={{ marginBottom: '6px' }}>
                  💻 System Parameters
                </h2>
                <div style={{ fontSize: '12px', fontWeight: '400', color: '#666' }}>
                  Configure CPU and RAM constraints for selected clients per round.
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px 16px', borderRadius: '12px', background: '#fff3e0', border: '1px solid #ffe0b2' }}>
                <input
                  type="checkbox"
                  checked={config.systemConstraintsEnabled}
                  onChange={(e) => handleConfigChange('systemConstraintsEnabled', e.target.checked)}
                  style={{ width: '18px', height: '18px', accentColor: '#f57c00' }}
                />
                <div>
                  <div style={{ fontWeight: '700', color: '#e65100' }}>Enable System Constraints</div>
                  <div style={{ fontSize: '12px', color: '#555', marginTop: '2px' }}>
                    Toggle CPU/RAM constraints on or off for the next training round.
                  </div>
                </div>
              </div>
            </div>
            
            <div className="grid-2" style={{ marginTop: '24px' }}>
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
                <label>CPU/RAM Constraints Share</label>
                <select
                  value={config.systemConstraintsShare}
                  disabled={!config.systemConstraintsEnabled}
                  onChange={(e) => handleConfigChange('systemConstraintsShare', parseFloat(e.target.value))}
                >
                  {constraintShareOptions.map(opt => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
                <div style={{ marginTop: '5px', fontSize: '12px', color: '#888' }}>
                  What fraction of clients get CPU/RAM constraints per round
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
                (Dataset, Model, Distribution)
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
                <label>Model</label>
                <select
                  value={config.model}
                  onChange={(e) => handleConfigChange('model', e.target.value)}
                >
                  {models.map(m => (
                    <option key={m.value} value={m.value}>{m.label}</option>
                  ))}
                </select>
                <div style={{ marginTop: '5px', fontSize: '12px', color: '#888' }}>
                  Select the model architecture for training
                </div>
              </div>
            </div>

            <div className="form-group" style={{ marginTop: '15px' }}>
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

          {/* ============================================ */}
          {/* TRAINING PARAMETERS */}
          {/* ============================================ */}
          <div className="card" style={{ marginBottom: '20px', borderLeft: '4px solid #66bb6a' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '15px', flexWrap: 'wrap', marginBottom: '20px' }}>
              <div>
                <h2 className="panel-title" style={{ marginBottom: '8px' }}>
                  ⚙️ Training Parameters
                </h2>
                <div style={{ color: '#aaa', fontSize: '13px', maxWidth: '720px', lineHeight: 1.5 }}>
                  Configure the experiment parameters: number of rounds, local epochs, batch size, learning rate, and timeout.
                </div>
              </div>

            </div>

            <div className="grid-2" style={{ gap: '20px' }}>
              <div className="form-group" style={{ padding: '15px', background: 'rgba(102, 187, 106, 0.08)', borderRadius: '8px' }}>
                <label style={{ display: 'block', marginBottom: '8px' }}>Number of Rounds</label>
                <input
                  type="number"
                  min="1"
                  max="500"
                  value={config.numRounds}
                  onChange={(e) => handleConfigChange('numRounds', parseInt(e.target.value))}
                />
              </div>

              <div className="form-group" style={{ padding: '15px', background: 'rgba(102, 187, 106, 0.08)', borderRadius: '8px' }}>
                <label style={{ display: 'block', marginBottom: '8px' }}>Experiment Rounds</label>
                <input
                  type="number"
                  min="1"
                  max="100"
                  value={config.experimentRounds}
                  onChange={(e) => handleConfigChange('experimentRounds', parseInt(e.target.value))}
                />
                <div style={{ marginTop: '8px', fontSize: '12px', color: '#888' }}>
                  Number of rounds automatically started when the experiment begins.
                </div>
              </div>

              <div className="form-group" style={{ padding: '15px', background: 'rgba(102, 187, 106, 0.08)', borderRadius: '8px' }}>
                <label style={{ display: 'block', marginBottom: '8px' }}>Local Epochs</label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={config.localEpochs}
                  onChange={(e) => handleConfigChange('localEpochs', parseInt(e.target.value))}
                />
              </div>

              <div className="form-group" style={{ padding: '15px', background: 'rgba(102, 187, 106, 0.08)', borderRadius: '8px' }}>
                <label style={{ display: 'block', marginBottom: '8px' }}>Batch Size</label>
                <input
                  type="number"
                  min="8"
                  max="2048"
                  value={config.batchSize}
                  onChange={(e) => handleConfigChange('batchSize', parseInt(e.target.value))}
                />
              </div>
            </div>

            <div className="grid-2" style={{ gap: '20px', marginTop: '20px' }}>
              <div className="form-group" style={{ padding: '15px', background: 'rgba(102, 187, 106, 0.08)', borderRadius: '8px' }}>
                <label style={{ display: 'block', marginBottom: '8px' }}>Learning Rate</label>
                <input
                  type="number"
                  step="0.001"
                  min="0.001"
                  max="1"
                  value={config.learningRate}
                  onChange={(e) => handleConfigChange('learningRate', parseFloat(e.target.value))}
                />
              </div>

              <div className="form-group" style={{ padding: '15px', background: 'rgba(102, 187, 106, 0.08)', borderRadius: '8px' }}>
                <label style={{ display: 'block', marginBottom: '8px' }}>Round Timeout</label>
                <input
                  type="number"
                  min="0"
                  max="3600"
                  value={config.roundTimeout}
                  onChange={(e) => handleConfigChange('roundTimeout', parseInt(e.target.value))}
                />
                <div style={{ marginTop: '8px', fontSize: '12px', color: '#888' }}>
                  Maximum waiting time for clients. 0 = no timeout.
                </div>
              </div>
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
                onClick={handleStartExperiment}
                disabled={loading}
              >
                ▶ Start Experiment
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

