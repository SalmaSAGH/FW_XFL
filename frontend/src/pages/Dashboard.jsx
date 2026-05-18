import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { 
  getStatus, 
  getConfig,
  getAccuracyData,
  getLossData,
  getCpuData,
  getClientsData, 
  getBandwidthData,
  getThroughputData,
  getLatencyData,
  getEnergyData,
  getNetworkMetricsData,
  getRoundsHistory,
  startRound,
  setXflStrategy,
  exportData,
  resetMetrics
} from '../services/api';

function Dashboard() {
  const navigate = useNavigate();
const [status, setStatus] = useState({});
  const [config, setConfig] = useState({});
  const [clients, setClients] = useState([]);
  const [accuracyData, setAccuracyData] = useState({ rounds: [], accuracy: [] });
  const [lossData, setLossData] = useState({ rounds: [], loss: [] });
  const [cpuData, setCpuData] = useState({ rounds: [], cpu: [] });
  const [bandwidthData, setBandwidthData] = useState({ rounds: [], bandwidth_mb: [] });
  const [throughputData, setThroughputData] = useState({ rounds: [], throughput_mbps: [] });
  const [latencyData, setLatencyData] = useState({ rounds: [], latency: [] });
  const [energyData, setEnergyData] = useState({ rounds: [], energy: [] });
  const [networkMetrics, setNetworkMetrics] = useState({ rounds: [], packet_loss: [], jitter: [] });
  const [history, setHistory] = useState([]);
  const [resourceAlerts, setResourceAlerts] = useState([]);

  // Strategy is now fetched from server status, no local state needed
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  const [dataError, setDataError] = useState(false);
  const [serverAvailable, setServerAvailable] = useState(true);
  const [resetting, setResetting] = useState(false);
  const [resetStatus, setResetStatus] = useState('');
  const [consecutiveErrors, setConsecutiveErrors] = useState(0);
  const [lastSuccessfulFetch, setLastSuccessfulFetch] = useState(null);
  const [initialLoading, setInitialLoading] = useState(true);

  // Get number of devices from config or use default
  const NUM_DEVICES = status.total_clients || 40;

  useEffect(() => {
    // Initial fetch
    fetchAllData();
    
    // Set up polling interval
    const interval = setInterval(fetchAllData, 2000);
    
    // Safety timeout to exit initial loading after 10 seconds max
    const safetyTimeout = setTimeout(() => {
      console.warn("Safety timeout: exiting initial loading state");
      setInitialLoading(false);
    }, 10000);

    return () => {
      clearInterval(interval);
      clearTimeout(safetyTimeout);
    };
  }, []);

  const fetchAllData = async () => {
    // If server was unavailable, add a small delay before retrying
    if (!serverAvailable) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    try {
      setDataError(false);
      
// Fetch all data with individual error handling to preserve data on partial failures
      let statusRes, accuracyRes, lossRes, clientsRes, cpuRes, bandwidthRes, latencyRes, energyRes, networkRes, historyRes;
      
      try {
        statusRes = await getStatus();
      } catch (e) {
        console.warn('Status fetch failed:', e.message);
        statusRes = { data: null };
      }
      
      try {
        accuracyRes = await getAccuracyData();
      } catch (e) {
        console.warn('Accuracy fetch failed:', e.message);
        accuracyRes = { data: null };
      }

      try {
        lossRes = await getLossData();
      } catch (e) {
        console.warn('Loss fetch failed:', e.message);
        lossRes = { data: null };
      }
      
      try {
        cpuRes = await getCpuData();
      } catch (e) {
        console.warn('CPU fetch failed:', e.message);
        cpuRes = { data: null };
      }
      

      
      try {
        clientsRes = await getClientsData();
      } catch (e) {
        console.warn('Clients fetch failed:', e.message);
        clientsRes = { data: null };
      }
      
      try {
        bandwidthRes = await getBandwidthData();
      } catch (e) {
        console.warn('Bandwidth fetch failed:', e.message);
        bandwidthRes = { data: null };
      }
      
      let throughputRes;
      try {
        throughputRes = await getThroughputData();
      } catch (e) {
        console.warn('Throughput fetch failed:', e.message);
        throughputRes = { data: null };
      }
      
      try {
        latencyRes = await getLatencyData();
      } catch (e) {
        console.warn('Latency fetch failed:', e.message);
        latencyRes = { data: null };
      }
      
      try {
        energyRes = await getEnergyData();
      } catch (e) {
        console.warn('Energy fetch failed:', e.message);
        energyRes = { data: null };
      }
      
      try {
        networkRes = await getNetworkMetricsData();
      } catch (e) {
        console.warn('Network metrics fetch failed:', e.message);
        networkRes = { data: null };
      }
      
      try {
        historyRes = await getRoundsHistory();
      } catch (e) {
        console.warn('History fetch failed:', e.message);
        historyRes = { data: null };
      }
      
      try {
        const configRes = await getConfig();
        if (configRes && configRes.data && configRes.data.config) {
          setConfig(configRes.data.config);
        }
      } catch (e) {
        console.warn('Config fetch failed:', e.message);
      }

      // Server is available if at least one API call succeeded
      const anySuccess = [statusRes, accuracyRes, lossRes, clientsRes, bandwidthRes, latencyRes, energyRes, networkRes, historyRes]
        .some(res => res && res.data);
      
      setServerAvailable(anySuccess);
      // Always exit initial loading state after first successful fetch attempt
      // This prevents getting stuck on "Loading..." even if no data exists yet
      setInitialLoading(false);
      
      if (anySuccess) {
        setConsecutiveErrors(0);
        setLastSuccessfulFetch(new Date());
      } else {
        setConsecutiveErrors(prev => prev + 1);
      }

      // Update status only if we have valid status data
      if (statusRes && statusRes.data && Object.keys(statusRes.data).length > 0) {
        setStatus(statusRes.data);
        if (Array.isArray(statusRes.data.resource_alerts)) {
          setResourceAlerts(statusRes.data.resource_alerts);
        }
        // DEBUG: Log status to console for debugging round progress bar and client colors
        console.log('[DEBUG] Status updated:', {
          round_in_progress: statusRes.data.round_in_progress,
          current_round: statusRes.data.current_round,
          submissions_received: statusRes.data.submissions_received,
          clients_expected: statusRes.data.clients_expected,
          xfl_strategy: statusRes.data.xfl_strategy,
          selected_clients: statusRes.data.selected_clients
        });
        // Strategy is now read directly from status.xfl_strategy in the render
      }

// FIXED: Only update metrics data if we have valid data with actual content - preserve existing data

      
      // Accuracy data - check for non-empty rounds array
      if (accuracyRes && accuracyRes.data && accuracyRes.data.rounds && accuracyRes.data.rounds.length > 0) {
        setAccuracyData(accuracyRes.data);
      }
      // Loss data - check for non-empty rounds array
      if (lossRes && lossRes.data && lossRes.data.rounds && lossRes.data.rounds.length > 0) {
        setLossData(lossRes.data);
      }
      
      // CPU data - check for non-empty rounds array
      if (cpuRes && cpuRes.data && cpuRes.data.rounds && cpuRes.data.rounds.length > 0) {
        setCpuData(cpuRes.data);
      }
      
      // Clients data - check for non-empty clients array
      if (clientsRes && clientsRes.data && clientsRes.data.clients && clientsRes.data.clients.length > 0) {
        setClients(clientsRes.data.clients);
      }
      
      // Bandwidth data - check for non-empty rounds array
      if (bandwidthRes && bandwidthRes.data && bandwidthRes.data.rounds && bandwidthRes.data.rounds.length > 0) {
        setBandwidthData(bandwidthRes.data);
      }
      
      // Throughput data - check for non-empty rounds array
      if (throughputRes && throughputRes.data && throughputRes.data.rounds && throughputRes.data.rounds.length > 0) {
        setThroughputData(throughputRes.data);
      }
      
      // Latency data - check for non-empty rounds array
      
      if (latencyRes && latencyRes.data && latencyRes.data.rounds && latencyRes.data.rounds.length > 0) {
        setLatencyData(latencyRes.data);
      }
      
      // Energy data - check for non-empty rounds array
      if (energyRes && energyRes.data && energyRes.data.rounds && energyRes.data.rounds.length > 0) {
        setEnergyData(energyRes.data);
      }
      
      // Network metrics data - check for non-empty rounds array
      if (networkRes && networkRes.data && networkRes.data.rounds && networkRes.data.rounds.length > 0) {
        setNetworkMetrics(networkRes.data);
      }
      
      // History data - check for non-empty rounds array
      if (historyRes && historyRes.data && historyRes.data.rounds && historyRes.data.rounds.length > 0) {
        setHistory(historyRes.data.rounds);
      }
    } catch (error) {
      console.error('Error fetching data:', error);
      setDataError(true);
      setServerAvailable(false);
      setConsecutiveErrors(prev => prev + 1);
      setInitialLoading(false);
      // Don't clear existing data on error - preserve it
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('username');
    navigate('/login');
  };

  const handleStartRound = async () => {
    setLoading(true);
    setMessage({ type: '', text: '' });
    
    try {
      const response = await startRound();
      if (response.data.status === 'started') {
        setMessage({ type: 'success', text: `Round ${response.data.round} started successfully!` });
      } else {
        setMessage({ type: 'warning', text: response.data.message || 'Could not start round' });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Error starting round: ' + error.message });
    }
    
    setLoading(false);
  };

  const handleApplyStrategy = async (selectedStrategy) => {
    setLoading(true);
    setMessage({ type: '', text: '' });
    
    try {
      await setXflStrategy(selectedStrategy, 3);
      setMessage({ type: 'success', text: 'Strategy updated successfully!' });
      // Force refresh to get updated strategy from server
      fetchAllData();
    } catch (error) {
      setMessage({ type: 'error', text: 'Error updating strategy: ' + error.message });
    }
    
    setLoading(false);
  };

  const handleExport = async () => {
    setLoading(true);
    try {
      const response = await exportData();
      setMessage({ type: 'success', text: 'Data exported successfully!' });
    } catch (error) {
      setMessage({ type: 'error', text: 'Error exporting data: ' + error.message });
    }
    setLoading(false);
  };

  const handleResetData = async () => {
    setResetting(true);
    setResetStatus('');
    setMessage({ type: '', text: '' });

    try {
      const response = await resetMetrics();
      if (response.data && response.data.status === 'ok') {
        setResetStatus('Données réinitialisées.');
        fetchAllData();
      } else {
        setMessage({ type: 'error', text: 'Impossible de réinitialiser les données.' });
      }
    } catch (error) {
      console.error('Reset failed:', error);
      setMessage({ type: 'error', text: 'Échec de la réinitialisation : ' + (error.response?.data?.error || error.message) });
    } finally {
      setResetting(false);
    }
  };

const accuracyChartData = accuracyData.rounds.map((round, idx) => ({
    round,
    accuracy: accuracyData.accuracy[idx]
  }));

  const lossChartData = lossData.rounds.map((round, idx) => ({
    round,
    loss: lossData.loss[idx]
  }));

  const cpuChartData = cpuData.rounds.map((round, idx) => ({
    round,
    cpu: cpuData.cpu[idx]
  }));

const bandwidthChartData = bandwidthData.rounds.map((round, idx) => ({
    round,
    bandwidth: bandwidthData.bandwidth_mb[idx]
  }));

  const throughputChartData = throughputData.rounds.map((round, idx) => ({
    round,
    throughput: throughputData.throughput_mbps[idx]
  }));

  const latencyChartData = latencyData.rounds.map((round, idx) => ({
    round,
    latency: latencyData.latency[idx]
  }));

  const energyChartData = energyData.rounds.map((round, idx) => ({
    round,
    energy: energyData.energy[idx]
  }));

  const networkChartData = networkMetrics.rounds.map((round, idx) => ({
    round,
    packet_loss: networkMetrics.packet_loss[idx],
    jitter: networkMetrics.jitter[idx]
  }));

  const effectiveAccuracySeries = (accuracyData.accuracy && accuracyData.accuracy.length > 0)
    ? accuracyData.accuracy.filter(v => v != null)
    : history.map(h => h.accuracy).filter(v => v != null);

  const convergenceValue = (() => {
    if (effectiveAccuracySeries.length >= 2) {
      const recent = effectiveAccuracySeries.slice(-3);
      if (recent.length >= 2) {
        const improvement = recent[recent.length - 1] - recent[0];
        return `${improvement >= 0 ? '+' : ''}${improvement.toFixed(2)}%`;
      }
    }
    return 'N/A';
  })();

  const completedRoundsCount = history.filter(h => h.accuracy != null).length;
  const targetRounds = status.total_rounds || config.numRounds || 0;
  const successRateValue = targetRounds > 0
    ? `${Math.round((completedRoundsCount / targetRounds) * 100)}%`
    : 'N/A';

  const validRoundTimes = history
    .map(h => Number(h.agg_time))
    .filter(t => !Number.isNaN(t) && t > 0);
  const avgRoundTimeValue = validRoundTimes.length > 0
    ? `${(validRoundTimes.reduce((sum, t) => sum + t, 0) / validRoundTimes.length).toFixed(1)}s`
    : 'N/A';

  // Get Pi card state - map server states to UI states
  // Server returns: "idle" (not selected), "training" (selected but not submitted), "active" (submitted)
  const getPiState = (client) => {
    if (status.round_in_progress) {
      if (client.state === 'active') return 'active';     // Green - already submitted
      if (client.state === 'training') return 'training'; // Blue - selected, training in progress
      return 'idle';                                    // Gray - not selected for this round
    }
    if (client.state === 'active') return 'active';
    return 'idle';
  };

  // Show loading indicator during initial load
  if (initialLoading) {
    return (
      <div className="page-container">
        <header className="header">
          <h1>🔷 XFL-FW - Dashboard</h1>
          <div className="header-nav">
            <a href="#" onClick={() => navigate('/config')}>Config</a>
            <a href="#" onClick={() => navigate('/dashboard')} className="active">Dashboard</a>
            <a href="#" onClick={() => navigate('/history')}>History</a>
            <button className="btn btn-secondary" onClick={handleResetData} disabled={resetting} style={{ marginRight: '10px' }}>
              {resetting ? 'Réinitialisation...' : 'Réinitialiser'}
            </button>
            <button className="logout-btn" onClick={handleLogout}>Logout</button>
          </div>
        </header>
        <main className="main-content">
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: '50vh',
            flexDirection: 'column',
            gap: '20px'
          }}>
            <div className="loading-spinner"></div>
            <p style={{ color: '#888' }}>(...) Loading dashboard data...</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="page-container">
      {/* Header */}
      <header className="header">
        <h1>🔷 XFL-FW - Dashboard</h1>
        <div className="header-nav">
          <a href="#" onClick={() => navigate('/config')}>Config</a>
          <a href="#" onClick={() => navigate('/dashboard')} className="active">Dashboard</a>
          <a href="#" onClick={() => navigate('/dse')}>DSE</a>
          <a href="#" onClick={() => navigate('/history')}>History</a>
          <button className="btn btn-secondary" onClick={handleResetData} disabled={resetting} style={{ marginRight: '10px' }}>
            {resetting ? 'Réinitialisation...' : 'Réinitialiser'}
          </button>
          <button className="logout-btn" onClick={handleLogout}>Logout</button>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Server Unavailable Warning Banner */}
        {!serverAvailable && (
          <div style={{ 
            padding: '15px', 
            marginBottom: '20px', 
            borderRadius: '4px',
            background: '#ef5350',
            color: 'white',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <div>
              <strong>⚠️ Server Unavailable</strong> - Data from last successful fetch is displayed. Attempting to reconnect...
            </div>
            <div style={{ fontSize: '12px' }}>
              {consecutiveErrors > 0 && `Failed attempts: ${consecutiveErrors}`}
            </div>
          </div>
        )}

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

{/* Status Cards Row */}
        <div className="grid-4" style={{ marginBottom: '20px' }}>
          <div className="info-card">
            <div className="info-label">Current Round</div>
            <div className="info-value">{status.current_round || 0} / {status.total_rounds || '?'}</div>
          </div>
          <div className="info-card">
            <div className="info-label">Active Clients</div>
            <div className="info-value">{status.total_clients || 0}</div>
          </div>
          <div className="info-card">
            <div className="info-label">Latest Accuracy</div>
            <div className="info-value">{status.latest_accuracy ? status.latest_accuracy.toFixed(1) + '%' : '-'}</div>
          </div>
          <div className="info-card">
            <div className="info-label">XFL Strategy</div>
            <div className="info-value" style={{ fontSize: '16px' }}>
              {status.xfl_strategy || 'all_layers'}
            </div>
          </div>
        </div>

{/* Current Configuration Parameters Display */}
        <div className="card" style={{ marginBottom: '20px', borderLeft: '4px solid #ab47bc' }}>
          <h2 className="panel-title">⚙️ Current Configuration</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '15px', marginTop: '15px' }}>
            {/* Algorithmic Parameters */}
            <div style={{ padding: '10px', background: 'rgba(171, 71, 188, 0.1)', borderRadius: '4px' }}>
              <div style={{ fontSize: '12px', fontWeight: '600', color: '#ab47bc', marginBottom: '8px' }}>🔧 Algorithmic</div>
              <div style={{ fontSize: '11px', color: '#888' }}>Strategy: <span style={{ color: '#fff' }}>{config.strategy || 'all_layers'}</span></div>
              <div style={{ fontSize: '11px', color: '#888' }}>XFL Param: <span style={{ color: '#fff' }}>{config.xflParam || 3}</span></div>
            </div>
            {/* Network Parameters */}
            <div style={{ padding: '10px', background: 'rgba(79, 195, 247, 0.1)', borderRadius: '4px' }}>
              <div style={{ fontSize: '12px', fontWeight: '600', color: '#4fc3f7', marginBottom: '8px' }}>🌐 Network</div>
              <div style={{ fontSize: '11px', color: '#888' }}>Latency: <span style={{ color: '#fff' }}>{config.networkLatency || 0}ms</span></div>
              <div style={{ fontSize: '11px', color: '#888' }}>Bandwidth: <span style={{ color: '#fff' }}>{config.networkBandwidth || 10}Mbps</span></div>
              <div style={{ fontSize: '11px', color: '#888' }}>Packet Loss: <span style={{ color: '#fff' }}>{config.networkPacketLoss || 0}%</span></div>
            </div>
            {/* System Parameters */}
            <div style={{ padding: '10px', background: 'rgba(255, 167, 38, 0.1)', borderRadius: '4px' }}>
              <div style={{ fontSize: '12px', fontWeight: '600', color: '#ffa726', marginBottom: '8px' }}>💻 System</div>
              <div style={{ fontSize: '11px', color: '#888' }}>CPU Limit: <span style={{ color: '#fff' }}>{config.cpuLimit || 100}%</span></div>
              <div style={{ fontSize: '11px', color: '#888' }}>RAM Limit: <span style={{ color: '#fff' }}>{config.ramLimit || 2048}MB</span></div>
              <div style={{ fontSize: '11px', color: '#888' }}>Clients: <span style={{ color: '#fff' }}>{config.numClients || 40}</span></div>
            </div>
            {/* Data Parameters */}
            <div style={{ padding: '10px', background: 'rgba(102, 187, 106, 0.1)', borderRadius: '4px' }}>
              <div style={{ fontSize: '12px', fontWeight: '600', color: '#66bb6a', marginBottom: '8px' }}>📊 Data</div>
              <div style={{ fontSize: '11px', color: '#888' }}>Dataset: <span style={{ color: '#fff' }}>{config.dataset || 'MNIST'}</span></div>
              <div style={{ fontSize: '11px', color: '#888' }}>Distribution: <span style={{ color: '#fff' }}>{config.dataDistribution === 'iid' ? 'IID' : 'Non-IID'}</span></div>
              <div style={{ fontSize: '11px', color: '#888' }}>Model: <span style={{ color: '#fff' }}>{config.model || SimpleCNN}</span></div>
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        {status.round_in_progress && (
          <div className="progress-container" style={{ marginBottom: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
              <span className="panel-title">Round Progress</span>
              <span style={{ color: '#4fc3f7', fontWeight: '600' }}>
                {status.submissions_received || 0} / {status.clients_expected || 0} clients
              </span>
            </div>
            <div className="progress-bar-bg">
              <div 
                className="progress-bar-fill" 
                style={{ width: `${status.clients_expected ? Math.round((status.submissions_received / status.clients_expected) * 100) : 0}%` }}
              ></div>
              <div className="progress-text">
                {status.clients_expected ? Math.round((status.submissions_received / status.clients_expected) * 100) : 0}%
              </div>
            </div>
          </div>
        )}

        {/* Testbed Grid - Full Width */}
        <div className="card" style={{ marginBottom: '20px' }}>
          <h2 className="panel-title">Testbed Monitoring ({NUM_DEVICES} Raspberry Pi)</h2>
          
          {/* Legend */}
          <div style={{ display: 'flex', gap: '15px', marginBottom: '15px', flexWrap: 'wrap', fontSize: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <div style={{ width: '12px', height: '12px', background: '#66bb6a', borderRadius: '2px' }}></div>
              <span>Active</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <div style={{ width: '12px', height: '12px', background: '#4fc3f7', borderRadius: '2px' }}></div>
              <span>Training</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <div style={{ width: '12px', height: '12px', background: '#252942', borderRadius: '2px' }}></div>
              <span>Idle</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
              <div style={{ width: '12px', height: '12px', background: '#ef5350', borderRadius: '2px' }}></div>
              <span>Error</span>
            </div>
          </div>

          {/* Pi Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: '8px' }}>
            {Array.from({ length: NUM_DEVICES }, (_, i) => {
              const client = clients.find(c => c.client_id === i);
              const state = client ? getPiState(client) : 'idle';
              return (
                <div 
                  key={i}
                  style={{
                    background: state === 'active' ? '#66bb6a' : state === 'training' ? '#4fc3f7' : state === 'error' ? '#ef5350' : '#252942',
                    borderRadius: '4px',
                    padding: '8px',
                    border: '1px solid #2d3348',
                    minHeight: '60px'
                  }}
                >
                  <div style={{ fontSize: '10px', fontWeight: '600', color: 'white', marginBottom: '4px' }}>
                    Pi {String(i).padStart(2, '0')}
                  </div>
                  <div style={{ fontSize: '8px', color: 'rgba(255,255,255,0.7)' }}>
                    {state.charAt(0).toUpperCase() + state.slice(1)}
                  </div>
                  {client && (
                    <div style={{ fontSize: '7px', color: 'rgba(255,255,255,0.6)', marginTop: '4px' }}>
                      CPU: {client.avg_cpu || 0}% | RAM: {client.avg_memory || 0} MB
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>



{/* Charts Row 2 */}
        <div className="grid-3" style={{ marginBottom: '20px' }}>
          <div className="card">
            <h2 className="panel-title">📊 Bandwidth Usage (MB)</h2>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={bandwidthChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                <XAxis dataKey="round" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip 
                  contentStyle={{ background: '#252942', border: '1px solid #2d3348' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="bandwidth" 
                  stroke="#4fc3f7" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h2 className="panel-title">🚀 Network Throughput (Mbps)</h2>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={throughputChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                <XAxis dataKey="round" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip 
                  contentStyle={{ background: '#252942', border: '1px solid #2d3348' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="throughput" 
                  stroke="#66bb6a" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h2 className="panel-title">⏱️ Network Latency (ms)</h2>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={latencyChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                <XAxis dataKey="round" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip 
                  contentStyle={{ background: '#252942', border: '1px solid #2d3348' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="latency" 
                  stroke="#ffa726" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Charts Row 3 - Model Accuracy, CPU Usage and Energy */}
        <div className="grid-3" style={{ marginBottom: '20px' }}>
          <div className="card">
            <h2 className="panel-title">📉 Global Test Loss</h2>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={lossChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                <XAxis dataKey="round" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip 
                  contentStyle={{ background: '#252942', border: '1px solid #2d3348' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="loss" 
                  stroke="#ef5350" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h2 className="panel-title">💻 CPU Usage (%)</h2>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={cpuChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                <XAxis dataKey="round" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{ background: '#252942', border: '1px solid #2d3348' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="cpu" 
                  stroke="#ffa726" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h2 className="panel-title">⚡ Energy Consumption (Wh)</h2>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={energyChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                <XAxis dataKey="round" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip 
                  contentStyle={{ background: '#252942', border: '1px solid #2d3348' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="energy" 
                  stroke="#ab47bc" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Network Metrics Chart */}
        <div className="card" style={{ marginBottom: '20px' }}>
          <h2 className="panel-title">📡 Network Metrics (Packet Loss & Jitter)</h2>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={networkChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
              <XAxis dataKey="round" stroke="#888" fontSize={10} />
              <YAxis stroke="#888" fontSize={10} />
              <Tooltip 
                contentStyle={{ background: '#252942', border: '1px solid #2d3348' }}
                labelStyle={{ color: '#fff' }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="packet_loss" 
                name="Packet Loss (%)"
                stroke="#ef5350" 
                strokeWidth={2}
                dot={false}
              />
              <Line 
                type="monotone" 
                dataKey="jitter" 
                name="Jitter (ms)"
                stroke="#ab47bc" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Resource Alert Panel */}
        <div className="card" style={{ marginBottom: '20px', borderLeft: '4px solid #ef5350' }}>
          <h2 className="panel-title">🚨 Resource Alerts</h2>
          
          {resourceAlerts.length === 0 ? (
            <div style={{ padding: '12px', color: '#fff', background: '#2e3b55', borderRadius: '4px', textAlign: 'center' }}>
              ✅ Aucun client n'a dépassé les seuils CPU/RAM enregistrés.
            </div>
          ) : (
            <>
              {/* Summary Stats */}
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', 
                gap: '10px', 
                marginBottom: '15px',
                padding: '10px',
                background: 'rgba(255, 87, 34, 0.1)',
                borderRadius: '4px'
              }}>
                <div style={{ fontSize: '12px' }}>
                  <span style={{ color: '#888' }}>Clients Affectés:</span>
                  <span style={{ color: '#ff6b6b', fontWeight: '700', marginLeft: '5px' }}>{resourceAlerts.length}</span>
                </div>
                <div style={{ fontSize: '12px' }}>
                  <span style={{ color: '#888' }}>CPU Dépasse:</span>
                  <span style={{ color: '#ffa726', fontWeight: '700', marginLeft: '5px' }}>
                    {resourceAlerts.filter(a => a.cpu_limit_exceeded).length}
                  </span>
                </div>
                <div style={{ fontSize: '12px' }}>
                  <span style={{ color: '#888' }}>RAM Dépasse:</span>
                  <span style={{ color: '#ffa726', fontWeight: '700', marginLeft: '5px' }}>
                    {resourceAlerts.filter(a => a.ram_limit_exceeded).length}
                  </span>
                </div>
              </div>

              {/* Alert Items */}
              <div style={{ display: 'grid', gap: '12px' }}>
                {resourceAlerts.map((alert, index) => {
                  const cpuExcess = alert.cpu_limit_exceeded ? ((alert.cpu_percent - alert.cpu_limit) / alert.cpu_limit) * 100 : 0;
                  const ramExcess = alert.ram_limit_exceeded ? ((alert.memory_mb - alert.ram_limit) / alert.ram_limit) * 100 : 0;
                  
                  // Determine severity colors
                  const getCpuColor = () => {
                    if (!alert.cpu_limit_exceeded) return '#66bb6a';
                    if (alert.cpu_percent < alert.cpu_limit * 1.2) return '#ffa726';
                    if (alert.cpu_percent < alert.cpu_limit * 1.5) return '#ff7043';
                    return '#ef5350';
                  };
                  
                  const getRamColor = () => {
                    if (!alert.ram_limit_exceeded) return '#66bb6a';
                    if (alert.memory_mb < alert.ram_limit * 1.2) return '#ffa726';
                    if (alert.memory_mb < alert.ram_limit * 1.5) return '#ff7043';
                    return '#ef5350';
                  };

                  return (
                    <div 
                      key={`${alert.client_id}-${alert.round_number}-${index}`} 
                      style={{ 
                        padding: '12px', 
                        background: '#2d3d52',
                        borderRadius: '6px',
                        border: `1px solid ${alert.cpu_limit_exceeded || alert.ram_limit_exceeded ? '#ef5350' : '#4caf50'}`,
                        display: 'grid',
                        gap: '8px'
                      }}
                    >
                      {/* Header */}
                      <div style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between',
                        alignItems: 'center'
                      }}>
                        <div style={{ fontWeight: '700', color: '#fff' }}>
                          Client {alert.client_id} • Last Round
                        </div>
                        <div style={{ fontSize: '11px', color: '#888' }}>
                          {alert.timestamp ? new Date(alert.timestamp).toLocaleTimeString() : 'N/A'}
                        </div>
                      </div>

                      {/* CPU Metric */}
                      {alert.cpu_limit_exceeded && (
                        <div style={{ display: 'grid', gap: '4px' }}>
                          <div style={{ 
                            display: 'flex', 
                            justifyContent: 'space-between',
                            alignItems: 'center'
                          }}>
                            <span style={{ fontSize: '12px', color: getCpuColor(), fontWeight: '600' }}>
                              💻 CPU: {alert.cpu_percent.toFixed(1)}%
                            </span>
                            <span style={{ 
                              fontSize: '11px', 
                              background: getCpuColor(), 
                              color: '#fff',
                              padding: '2px 8px',
                              borderRadius: '3px',
                              fontWeight: '600'
                            }}>
                              +{cpuExcess.toFixed(0)}% 🔥
                            </span>
                          </div>
                          <div style={{ 
                            height: '6px', 
                            background: '#1a2332', 
                            borderRadius: '3px',
                            overflow: 'hidden'
                          }}>
                            <div style={{ 
                              height: '100%', 
                              width: `${Math.min(100, (alert.cpu_percent / (alert.cpu_limit * 2)) * 100)}%`,
                              background: getCpuColor(),
                              transition: 'width 0.3s ease'
                            }} />
                          </div>
                          <div style={{ fontSize: '11px', color: '#888', textAlign: 'right' }}>
                            Limite: {alert.cpu_limit}%
                          </div>
                        </div>
                      )}

                      {/* RAM Metric */}
                      {alert.ram_limit_exceeded && (
                        <div style={{ display: 'grid', gap: '4px' }}>
                          <div style={{ 
                            display: 'flex', 
                            justifyContent: 'space-between',
                            alignItems: 'center'
                          }}>
                            <span style={{ fontSize: '12px', color: getRamColor(), fontWeight: '600' }}>
                              🧠 RAM: {alert.memory_mb.toFixed(0)}MB
                            </span>
                            <span style={{ 
                              fontSize: '11px', 
                              background: getRamColor(), 
                              color: '#fff',
                              padding: '2px 8px',
                              borderRadius: '3px',
                              fontWeight: '600'
                            }}>
                              +{ramExcess.toFixed(0)}% 🔥
                            </span>
                          </div>
                          <div style={{ 
                            height: '6px', 
                            background: '#1a2332', 
                            borderRadius: '3px',
                            overflow: 'hidden'
                          }}>
                            <div style={{ 
                              height: '100%', 
                              width: `${Math.min(100, (alert.memory_mb / (alert.ram_limit * 2)) * 100)}%`,
                              background: getRamColor(),
                              transition: 'width 0.3s ease'
                            }} />
                          </div>
                          <div style={{ fontSize: '11px', color: '#888', textAlign: 'right' }}>
                            Limite: {alert.ram_limit}MB
                          </div>
                        </div>
                      )}

                      {/* OK Status */}
                      {!alert.cpu_limit_exceeded && !alert.ram_limit_exceeded && (
                        <div style={{ 
                          padding: '8px', 
                          background: 'rgba(102, 187, 106, 0.1)',
                          borderRadius: '4px',
                          color: '#66bb6a',
                          fontSize: '12px',
                          textAlign: 'center'
                        }}>
                          ✅ Ressources dans les limites
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>

        {/* System Health & Alerts */}
        <div className="card">
          <h2 className="panel-title">🩺 System Health & Alerts</h2>

          {/* Resource Alerts */}
          {resourceAlerts && resourceAlerts.length > 0 && (
            <div style={{ marginBottom: '20px' }}>
              <h3 style={{ color: '#ef5350', marginBottom: '10px', fontSize: '14px' }}>⚠️ Active Alerts</h3>
              <div style={{ display: 'grid', gap: '8px' }}>
                {resourceAlerts.map((alert, idx) => {
                  const clientName = alert.client_name || (alert.client_id ? `Client ${alert.client_id}` : 'System');
                  const cpuText = alert.cpu_percent != null ? `CPU ${alert.cpu_percent.toFixed(1)}%` : null;
                  const ramText = alert.memory_mb != null ? `RAM ${alert.memory_mb.toFixed(0)}MB` : null;
                  const parts = [];
                  if (cpuText) parts.push(cpuText);
                  if (ramText) parts.push(ramText);
                  const details = `Last Round - ${parts.length > 0 ? parts.join(' | ') : alert.message || 'Alerte ressource détectée'}`;

                  return (
                    <div key={idx} style={{
                      padding: '10px',
                      background: 'rgba(239, 83, 80, 0.1)',
                      border: '1px solid #ef5350',
                      borderRadius: '4px',
                      fontSize: '12px'
                    }}>
                      <strong>{clientName}</strong>: {details}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Health Metrics Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '15px', marginBottom: '20px' }}>
            {/* Network Health */}
            <div style={{ padding: '15px', background: 'rgba(79, 195, 247, 0.1)', borderRadius: '8px', border: '1px solid rgba(79, 195, 247, 0.3)' }}>
              <h4 style={{ color: '#4fc3f7', margin: '0 0 10px 0', fontSize: '14px' }}>🌐 Network Health</h4>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ fontSize: '12px', color: '#ccc' }}>Avg Latency:</span>
                <span style={{ fontSize: '12px', color: '#fff' }}>
                  {latencyData.latency && latencyData.latency.length > 0
                    ? `${(latencyData.latency.reduce((a, b) => a + b, 0) / latencyData.latency.length).toFixed(1)}ms`
                    : 'N/A'
                  }
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ fontSize: '12px', color: '#ccc' }}>Avg Throughput:</span>
                <span style={{ fontSize: '12px', color: '#fff' }}>
                  {throughputData.throughput_mbps && throughputData.throughput_mbps.length > 0
                    ? `${(throughputData.throughput_mbps.reduce((a, b) => a + b, 0) / throughputData.throughput_mbps.length).toFixed(2)} Mbps`
                    : 'N/A'
                  }
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '12px', color: '#ccc' }}>Packet Loss:</span>
                <span style={{ fontSize: '12px', color: networkMetrics.packet_loss && networkMetrics.packet_loss.some(p => p > 0.01) ? '#ef5350' : '#66bb6a' }}>
                  {networkMetrics.packet_loss && networkMetrics.packet_loss.length > 0
                    ? `${((networkMetrics.packet_loss.reduce((a, b) => a + b, 0) / networkMetrics.packet_loss.length) * 100).toFixed(2)}%`
                    : '0.00%'
                  }
                </span>
              </div>
            </div>

            {/* System Performance */}
            <div style={{ padding: '15px', background: 'rgba(255, 167, 38, 0.1)', borderRadius: '8px', border: '1px solid rgba(255, 167, 38, 0.3)' }}>
              <h4 style={{ color: '#ffa726', margin: '0 0 10px 0', fontSize: '14px' }}>⚡ System Performance</h4>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ fontSize: '12px', color: '#ccc' }}>Avg CPU:</span>
                <span style={{ fontSize: '12px', color: cpuData.cpu && cpuData.cpu.some(c => c > 80) ? '#ef5350' : '#66bb6a' }}>
                  {cpuData.cpu && cpuData.cpu.length > 0
                    ? `${(cpuData.cpu.reduce((a, b) => a + b, 0) / cpuData.cpu.length).toFixed(1)}%`
                    : 'N/A'
                  }
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ fontSize: '12px', color: '#ccc' }}>Active Clients:</span>
                <span style={{ fontSize: '12px', color: '#fff' }}>
                  {(() => {
                    console.log('[DEBUG] Active Clients calc:', {
                      round_in_progress: status.round_in_progress,
                      selected_clients: status.selected_clients,
                      history_length: history.length,
                      last_history_clients: history.length > 0 ? history[history.length - 1]?.clients : 'no history'
                    });
                    
                    if (status.round_in_progress) {
                      return status.selected_clients ? status.selected_clients.length : 0;
                    } else {
                      // Quand pas de round en cours, afficher clients du dernier round terminé
                      const lastRound = history.length > 0 ? history[history.length - 1] : null;
                      return lastRound && lastRound.clients ? lastRound.clients : 0;
                    }
                  })()}/{status.total_clients || 0}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '12px', color: '#ccc' }}>Round Status:</span>
                <span style={{ fontSize: '12px', color: (() => {
                  const isInProgress = status.round_in_progress || (status.selected_clients && status.selected_clients.length > 0);
                  const hasHistory = history.length > 0;
                  const color = isInProgress ? '#4fc3f7' : hasHistory ? '#66bb6a' : '#888';
                  
                  console.log('[DEBUG] Round Status calc:', {
                    round_in_progress: status.round_in_progress,
                    selected_clients: status.selected_clients,
                    history_length: history.length,
                    isInProgress,
                    hasHistory,
                    color
                  });
                  
                  return color;
                })() }}>
                  {(() => {
                    const isInProgress = status.round_in_progress || (status.selected_clients && status.selected_clients.length > 0);
                    const hasHistory = history.length > 0;
                    const statusText = isInProgress ? 'In Progress' : hasHistory ? 'Completed' : 'Idle';
                    
                    console.log('[DEBUG] Round Status text:', statusText);
                    return statusText;
                  })()}
                </span>
              </div>
            </div>

            {/* Training Efficiency */}
            <div style={{ padding: '15px', background: 'rgba(102, 187, 106, 0.1)', borderRadius: '8px', border: '1px solid rgba(102, 187, 106, 0.3)' }}>
              <h4 style={{ color: '#66bb6a', margin: '0 0 10px 0', fontSize: '14px' }}>📈 Training Efficiency</h4>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ fontSize: '12px', color: '#ccc' }}>Convergence:</span>
                <span style={{ fontSize: '12px', color: '#fff' }}>
                  {convergenceValue}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ fontSize: '12px', color: '#ccc' }}>Success Rate:</span>
                <span style={{ fontSize: '12px', color: '#fff' }}>
                  {successRateValue}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '12px', color: '#ccc' }}>Avg Round Time:</span>
                <span style={{ fontSize: '12px', color: '#fff' }}>
                  {avgRoundTimeValue}
                </span>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            <button
              className="btn btn-secondary"
              onClick={() => navigate('/config')}
              style={{ fontSize: '12px', padding: '8px 12px' }}
            >
              ⚙️ Adjust Config
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => navigate('/history')}
              style={{ fontSize: '12px', padding: '8px 12px' }}
            >
              📊 View Full History
            </button>
            <button
              className="btn"
              onClick={handleResetData}
              disabled={resetting}
              style={{ fontSize: '12px', padding: '8px 12px', background: '#ef5350', border: '1px solid #c62828' }}
            >
              {resetting ? '🔄 Resetting...' : '🗑️ Reset Metrics'}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default Dashboard;

