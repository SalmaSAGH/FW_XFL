import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { 
  getStatus, 
  getAccuracyData, 
  getLossData, 
  getClientsData, 
  getBandwidthData,
  getLatencyData,
  getEnergyData,
  getNetworkMetricsData,
  getRoundsHistory,
  startRound,
  setXflStrategy,
  exportData
} from '../services/api';

function Dashboard() {
  const navigate = useNavigate();
  const [status, setStatus] = useState({});
  const [accuracyData, setAccuracyData] = useState({ rounds: [], accuracy: [] });
  const [lossData, setLossData] = useState({ rounds: [], loss: [] });
  const [clients, setClients] = useState([]);
  const [bandwidthData, setBandwidthData] = useState({ labels: [], values: [] });
  const [latencyData, setLatencyData] = useState({ rounds: [], latency: [] });
  const [energyData, setEnergyData] = useState({ rounds: [], energy: [] });
  const [networkMetrics, setNetworkMetrics] = useState({ rounds: [], packet_loss: [], jitter: [] });
  const [history, setHistory] = useState([]);
  const [strategy, setStrategy] = useState('all_layers');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  const [dataError, setDataError] = useState(false);
  const [serverAvailable, setServerAvailable] = useState(true);
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
    
    return () => clearInterval(interval);
  }, []);

  const fetchAllData = async () => {
    // If server was unavailable, add a small delay before retrying
    if (!serverAvailable) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    try {
      setDataError(false);
      
      // Fetch all data with individual error handling to preserve data on partial failures
      let statusRes, accuracyRes, lossRes, clientsRes, bandwidthRes, latencyRes, energyRes, networkRes, historyRes;
      
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

      // Server is available if at least one API call succeeded
      const anySuccess = [statusRes, accuracyRes, lossRes, clientsRes, bandwidthRes, latencyRes, energyRes, networkRes, historyRes]
        .some(res => res && res.data);
      
      setServerAvailable(anySuccess);
      if (anySuccess) {
        setConsecutiveErrors(0);
        setLastSuccessfulFetch(new Date());
        setInitialLoading(false);
      } else {
        setConsecutiveErrors(prev => prev + 1);
      }

      // Update status only if we have valid status data
      if (statusRes && statusRes.data && Object.keys(statusRes.data).length > 0) {
        setStatus(statusRes.data);
        // Sync strategy from server status
        if (statusRes.data.xfl_strategy) {
          setStrategy(statusRes.data.xfl_strategy);
        }
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
      
      // Clients data - check for non-empty clients array
      if (clientsRes && clientsRes.data && clientsRes.data.clients && clientsRes.data.clients.length > 0) {
        setClients(clientsRes.data.clients);
      }
      
      // Bandwidth data - check for non-empty values array
      if (bandwidthRes && bandwidthRes.data && bandwidthRes.data.values && bandwidthRes.data.values.length > 0) {
        setBandwidthData(bandwidthRes.data);
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

  const handleApplyStrategy = async () => {
    setLoading(true);
    setMessage({ type: '', text: '' });
    
    try {
      await setXflStrategy(strategy, 3);
      setMessage({ type: 'success', text: 'Strategy updated successfully!' });
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

  // Prepare chart data - filter out null/undefined values
  const accuracyChartData = accuracyData.rounds
    .map((round, idx) => ({
      round,
      accuracy: accuracyData.accuracy[idx]
    }))
    .filter(item => item.accuracy != null && item.accuracy !== undefined);

  const lossChartData = lossData.rounds
    .map((round, idx) => ({
      round,
      loss: lossData.loss[idx]
    }))
    .filter(item => item.loss != null && item.loss !== undefined);

  const bandwidthChartData = bandwidthData.labels.map((label, idx) => ({
    label,
    value: bandwidthData.values[idx]
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

  // Get Pi card state
  const getPiState = (client) => {
    if (status.round_in_progress) {
      if (client.state === 'active') return 'active';
      if (client.state === 'training') return 'training';
      return 'selected';
    }
    if (client.state === 'active') return 'active';
    return 'idle';
  };

  // Show loading indicator during initial load
  if (initialLoading) {
    return (
      <div className="page-container">
        <header className="header">
          <h1>🔷 XFL-RPiLab - Dashboard</h1>
          <div className="header-nav">
            <a href="#" onClick={() => navigate('/config')}>Config</a>
            <a href="#" onClick={() => navigate('/dashboard')} className="active">Dashboard</a>
            <a href="#" onClick={() => navigate('/history')}>History</a>
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
            <p style={{ color: '#888' }}>Loading dashboard data...</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="page-container">
      {/* Header */}
      <header className="header">
        <h1>🔷 XFL-RPiLab - Dashboard</h1>
        <div className="header-nav">
          <a href="#" onClick={() => navigate('/config')}>Config</a>
          <a href="#" onClick={() => navigate('/dashboard')} className="active">Dashboard</a>
          <a href="#" onClick={() => navigate('/history')}>History</a>
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
            <div className="info-value" style={{ fontSize: '16px' }}>{status.xfl_strategy || 'all_layers'}</div>
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
                style={{ width: `${Math.round((status.submissions_received / status.clients_expected) * 100)}%` }}
              ></div>
              <div className="progress-text">
                {Math.round((status.submissions_received / status.clients_expected) * 100)}%
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
                      CPU: {client.avg_cpu || 0}%
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Charts Row 1 */}
        <div className="grid-2" style={{ marginBottom: '20px' }}>
          <div className="card">
            <h2 className="panel-title">▲ Accuracy Evolution</h2>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={accuracyChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                <XAxis dataKey="round" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip 
                  contentStyle={{ background: '#252942', border: '1px solid #2d3348' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="accuracy" 
                  stroke="#4fc3f7" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h2 className="panel-title">📉 Loss Evolution</h2>
            <ResponsiveContainer width="100%" height={200}>
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
        </div>

        {/* Charts Row 2 */}
        <div className="grid-3" style={{ marginBottom: '20px' }}>
          <div className="card">
            <h2 className="panel-title">📊 Bandwidth Usage (MB)</h2>
            <ResponsiveContainer width="100%" height={150}>
              <BarChart data={bandwidthChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                <XAxis dataKey="label" stroke="#888" fontSize={8} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip 
                  contentStyle={{ background: '#252942', border: '1px solid #2d3348' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Bar dataKey="value" fill="#4fc3f7" />
              </BarChart>
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
                  stroke="#66bb6a" 
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

        {/* Rounds History Table */}
        <div className="card">
          <h2 className="panel-title">📜 Rounds History</h2>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Round</th>
                  <th>Accuracy</th>
                  <th>Loss</th>
                  <th>Agg Time (s)</th>
                  <th>Clients</th>
                </tr>
              </thead>
              <tbody>
                {history.length > 0 ? (
                  history.map((round, idx) => (
                    <tr key={idx}>
                      <td>{round.round}</td>
                      <td>{round.accuracy ? round.accuracy.toFixed(2) + '%' : '-'}</td>
                      <td>{round.loss ? round.loss.toFixed(4) : '-'}</td>
                      <td>{round.agg_time ? round.agg_time.toFixed(2) : '-'}</td>
                      <td>{round.clients || '-'}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan="5" style={{ textAlign: 'center', color: '#888' }}>
                      No rounds yet
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </div>
  );
}

export default Dashboard;
