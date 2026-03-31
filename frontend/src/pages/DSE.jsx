import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  ScatterChart, Scatter, ZAxis, Cell 
} from 'recharts';
import { runDseSweep, getDseStatus, getDseResults, getDseSessions } from '../services/api';

function DSE() {
  const navigate = useNavigate();
  const [params, setParams] = useState({
    // Key params for sweeping
    learningRate: { min: 0.001, max: 0.1, step: 0.001, value: 0.01 },
    localEpochs: { min: 1, max: 10, step: 1, value: 2 },
    batchSize: { min: 32, max: 512, step: 64, value: 128 },
    clientsPerRound: { min: 2, max: 20, step: 2, value: 5 },
    xflParam: { min: 1, max: 10, step: 1, value: 3 },
    networkLatency: { min: 0, max: 200, step: 50, value: 50 },
  });
  const [sweepResults, setSweepResults] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  const [selectedSession, setSelectedSession] = useState(null);
  const [progress, setProgress] = useState(0);
  const [currentSweepSessionId, setCurrentSweepSessionId] = useState(localStorage.getItem('dseSessionId') || null);
  const [currentSweepStatus, setCurrentSweepStatus] = useState(null);

  useEffect(() => {
    loadSessions();
    const storedSessionId = localStorage.getItem('dseSessionId');
    if (storedSessionId) {
      setCurrentSweepSessionId(storedSessionId);
      pollDseStatus(storedSessionId);
    }
  }, []);

  const saveSweepSessionId = (sessionId) => {
    localStorage.setItem('dseSessionId', sessionId);
    setCurrentSweepSessionId(sessionId);
  };

  const clearSweepSessionId = () => {
    localStorage.removeItem('dseSessionId');
    setCurrentSweepSessionId(null);
    setCurrentSweepStatus(null);
  };

  const pollDseStatus = async (sessionId) => {
    try {
      const response = await getDseStatus(sessionId);
      const status = response.data.status || 'unknown';
      setCurrentSweepStatus(status);

      if (status === 'running' || status === 'queued') {
        setTimeout(() => pollDseStatus(sessionId), 5000);
      } else if (status === 'completed') {
        loadSessionResults(sessionId);
      }
    } catch (error) {
      console.log('DSE status poll failed', error);
    }
  };

  const loadSessions = async () => {
    try {
      const response = await getDseSessions();
      setSessions(response.data.sessions || []);
    } catch (error) {
      console.log('No DSE sessions found, backend API may need implementation');
    }
  };

  const handleParamChange = (param, value) => {
    setParams(prev => ({ ...prev, [param]: { ...prev[param], value } }));
  };

  const handleSweep = async () => {
    setLoading(true);
    setMessage({ type: '', text: '' });
    setProgress(0);

    try {
      // Prepare sweep config using slider-selected values
      const sweepConfig = {
        params: Object.fromEntries(
          Object.entries(params).map(([key, range]) => [
            key,
            [range.value]
          ])
        ),
        numShortRounds: 5, // Short runs for quick feedback
        dataset: 'MNIST', // Fixed for demo
      };

      const response = await runDseSweep(sweepConfig);
      if (response.data?.session_id) {
        saveSweepSessionId(response.data.session_id);
        setCurrentSweepStatus(response.data.status || 'started');
        setSweepResults([]);
        setMessage({ type: 'success', text: `DSE sweep started: ${response.data.session_id}` });
        setTimeout(() => pollDseStatus(response.data.session_id), 3000);
        setTimeout(loadSessions, 5000);
      } else {
        setSweepResults(response.data.results || []);
        setMessage({ type: 'success', text: 'DSE sweep completed!' });
        loadSessions();
      }
    } catch (error) {
      setMessage({ type: 'error', text: `Sweep failed: ${error.message}. Backend DSE API may need implementation.` });
    } finally {
      setLoading(false);
    }
  };

  const loadSessionResults = async (sessionId) => {
    try {
      const response = await getDseResults(sessionId);
      setSweepResults(response.data.results || []);
      setSelectedSession(sessionId);
      setMessage({ type: 'success', text: `Loaded results from ${sessionId}` });
      if (currentSweepSessionId === sessionId) {
        setCurrentSweepStatus('completed');
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to load session results' });
    }
  };

  // Transform results for charts (accuracy vs lr example)
  const accuracyVsLrData = sweepResults.map(r => ({
    lr: r.config.learningRate,
    accuracy: r.metrics?.final_accuracy || 0,
    loss: r.metrics?.final_loss || 0,
    time: r.metrics?.total_time || 0,
  })).sort((a, b) => a.lr - b.lr);

  const clientsTimeData = sweepResults.map(r => ({
    clientsPerRound: r.config.clientsPerRound || 0,
    time: r.metrics?.total_time || 0,
    accuracy: r.metrics?.final_accuracy || 0,
  }));

  const handleLogout = () => {
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('username');
    navigate('/login');
  };

  return (
    <div className="page-container">
      <header className="header">
        <h1>🔷 Design Space Exploration</h1>
        <div className="header-nav">
          <a href="#" onClick={() => navigate('/config')}>Config</a>
          <a href="#" onClick={() => navigate('/dashboard')}>Dashboard</a>
          <a href="#" onClick={() => navigate('/dse')} className="active">DSE</a>
          <a href="#" onClick={() => navigate('/history')}>History</a>
          <button className="logout-btn" onClick={handleLogout}>Logout</button>
        </div>
      </header>

      <main className="main-content">
        <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
          {message.text && (
            <div style={{ 
              padding: '15px', marginBottom: '20px', borderRadius: '4px',
              background: message.type === 'success' ? '#66bb6a' : '#ef5350',
              color: 'white' 
            }}>
              {message.text}
            </div>
          )}

          {/* Param Sliders */}
          <div className="card" style={{ marginBottom: '20px', borderLeft: '4px solid #ff5722' }}>
            <h2 className="panel-title">🎛️ Parameter Ranges (for Sweep)</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px', marginTop: '15px' }}>
              {Object.entries(params).map(([key, range]) => (
                <div key={key} className="form-group">
                  <label>{key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}</label>
                  <input
                    type="range"
                    min={range.min}
                    max={range.max}
                    step={range.step}
                    value={range.value}
                    onChange={(e) => handleParamChange(key, parseFloat(e.target.value))}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: '12px', color: '#888' }}>{range.value.toFixed(3)}</div>
                </div>
              ))}
            </div>
            <button 
              className="btn btn-primary" 
              onClick={handleSweep}
              disabled={loading}
              style={{ marginTop: '15px' }}
            >
              {loading ? 'Running Sweep...' : '🚀 Run Design Space Exploration'}
            </button>
            {currentSweepSessionId && (
              <div style={{ marginTop: '15px', color: '#fff' }}>
                <strong>Session en cours:</strong> {currentSweepSessionId}
                <br />
                <strong>Statut:</strong> {currentSweepStatus || 'unknown'}
                {currentSweepStatus === 'completed' && (
                  <button className="btn" style={{ marginLeft: '10px' }} onClick={() => loadSessionResults(currentSweepSessionId)}>
                    Charger les résultats
                  </button>
                )}
              </div>
            )}
            {loading && (
              <div style={{ marginTop: '10px' }}>
                <div className="progress-bar-bg" style={{ width: '300px', height: '20px' }}>
                  <div className="progress-bar-fill" style={{ width: `${progress}%` }}></div>
                </div>
                <span>{progress}%</span>
              </div>
            )}
          </div>

          {/* Existing Sessions */}
          {sessions.length > 0 && (
            <div className="card" style={{ marginBottom: '20px' }}>
              <h2 className="panel-title">📁 Previous DSE Sessions</h2>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                {sessions.map(session => (
                  <button 
                    key={session.id}
                    className="btn"
                    onClick={() => loadSessionResults(session.id)}
                    style={{ background: selectedSession === session.id ? '#4fc3f7' : '#252942' }}
                  >
                    {session.id} ({session.numConfigs} configs)
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Results Visualization */}
          {sweepResults.length > 0 && (
            <>
              <div className="grid-2" style={{ marginBottom: '20px' }}>
                {/* Accuracy vs Learning Rate */}
                <div className="card">
                  <h2 className="panel-title">📈 Accuracy vs Learning Rate</h2>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={accuracyVsLrData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                      <XAxis dataKey="lr" stroke="#888" />
                      <YAxis stroke="#888" />
                      <Tooltip />
                      <Line type="monotone" dataKey="accuracy" stroke="#66bb6a" strokeWidth={3} />
                      <Line type="monotone" dataKey="loss" stroke="#ef5350" strokeWidth={3} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Time vs ClientsPerRound */}
                <div className="card">
                  <h2 className="panel-title">⏱️ Time vs Clients/Round</h2>
                  <ResponsiveContainer width="100%" height={250}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                      <XAxis dataKey="clientsPerRound" name="Clients/Round" stroke="#888" />
                      <YAxis dataKey="time" name="Time (s)" stroke="#888" />
                      <ZAxis dataKey="accuracy" name="Accuracy" range={[64, 256]} />
                      <Tooltip />
                      <Scatter data={clientsTimeData} fill="#4fc3f7">
                        {clientsTimeData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={`hsl(${entry.accuracy * 360 / 100}, 70%, 50%)`} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Summary Table */}
              <div className="card">
                <h2 className="panel-title">📊 Sweep Summary ({sweepResults.length} configs)</h2>
                <div style={{ overflowX: 'auto' }}>
                  <table>
                    <thead>
                      <tr>
                        <th>LR</th>
                        <th>Epochs</th>
                        <th>Batch</th>
                        <th>Acc (%)</th>
                        <th>Loss</th>
                        <th>Time (s)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sweepResults.slice(-10).map((r, i) => (
                        <tr key={i}>
                          <td>{r.config.learningRate?.toFixed(3)}</td>
                          <td>{r.config.localEpochs}</td>
                          <td>{r.config.batchSize}</td>
                          <td>{r.metrics?.final_accuracy?.toFixed(1) || '-'}</td>
                          <td>{r.metrics?.final_loss?.toFixed(4) || '-'}</td>
                          <td>{r.metrics?.total_time?.toFixed(1) || '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

export default DSE;

