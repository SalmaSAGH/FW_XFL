import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { getHistoryByStrategy } from '../services/api';

// Strategy colors
const strategyColors = {
  'all_layers': '#4fc3f7',
  'xfl_cyclic': '#66bb6a',
  'xfl_sparsification': '#ffa726',
  'xfl_quantization': '#ef5350',
  'importance_based': '#ab47bc',
  'layerwise': '#26c6da',
  'adaptive': '#ff7043',
  'split': '#7e57c2'
};

const getStrategyColor = (strategy) => strategyColors[strategy] || '#888888';

const getStrategyLabel = (strategy) => {
  const labels = {
    'all_layers': 'FedAvg (All Layers)',
    'xfl_cyclic': 'XFL - Cyclic',
    'xfl_sparsification': 'XFL - Sparsification',
    'xfl_quantization': 'XFL - Quantization',
    'importance_based': 'Importance Based',
    'layerwise': 'Layerwise',
    'adaptive': 'Adaptive',
    'split': 'Split Learning'
  };
  return labels[strategy] || strategy;
};

// Format a Unix timestamp as a readable date/time string
const formatTimestamp = (ts) => {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit'
  });
};

function History() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [strategies, setStrategies] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [initialLoading, setInitialLoading] = useState(true);

  useEffect(() => {
    fetchHistoryData();
  }, []);

  const fetchHistoryData = async () => {
    try {
      setLoading(true);
      const response = await getHistoryByStrategy();
      
      if (response.data && response.data.strategies) {
        setStrategies(response.data.strategies);
        
        if (response.data.strategies.length > 0 && !selectedStrategy) {
          setSelectedStrategy(response.data.strategies[0]);
          if (response.data.strategies[0].experiments.length > 0) {
            setSelectedExperiment(response.data.strategies[0].experiments[0]);
          }
        }
      }
      
      setInitialLoading(false);
    } catch (err) {
      console.error('Error fetching history:', err);
      setError('Failed to load history data');
      setInitialLoading(false);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('username');
    navigate('/login');
  };

  const handleStrategyClick = (strategy) => {
    setSelectedStrategy(strategy);
    if (strategy.experiments.length > 0) {
      setSelectedExperiment(strategy.experiments[0]);
    } else {
      setSelectedExperiment(null);
    }
  };

  const handleExperimentClick = (experiment) => {
    setSelectedExperiment(experiment);
  };

  // ── Session badge label ────────────────────────────────────────────────────
  // Shows "Session N" for real sessions, "Legacy" for pre-session_id data.
  const getSessionLabel = (exp) => {
    if (exp.is_legacy) return 'Legacy data';
    return `Session ${exp.experiment_id}`;
  };

  const getSessionSublabel = (exp) => {
    if (exp.started_at) return formatTimestamp(exp.started_at);
    if (exp.is_legacy) return 'Recorded before session tracking';
    return 'Started at unknown time';
  };

  if (initialLoading) {
    return (
      <div className="page-container">
        <header className="header">
          <h1>🔷 XFL-RPiLab - History</h1>
          <div className="header-nav">
            <a href="#" onClick={() => navigate('/config')}>Config</a>
            <a href="#" onClick={() => navigate('/dashboard')}>Dashboard</a>
            <a href="#" onClick={() => navigate('/dse')}>DSE</a>
            <a href="#" onClick={() => navigate('/history')} className="active">History</a>
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
            <p style={{ color: '#888' }}>Loading history data...</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="page-container">
      {/* Header */}
      <header className="header">
        <h1>🔷 XFL-RPiLab - History</h1>
        <div className="header-nav">
          <a href="#" onClick={() => navigate('/config')}>Config</a>
          <a href="#" onClick={() => navigate('/dashboard')}>Dashboard</a>
          <a href="#" onClick={() => navigate('/history')} className="active">History</a>
          <button className="logout-btn" onClick={handleLogout}>Logout</button>
        </div>
      </header>

      <main className="main-content">
        {error && (
          <div style={{ 
            padding: '15px', 
            marginBottom: '20px', 
            borderRadius: '4px',
            background: '#ef5350',
            color: 'white'
          }}>
            {error}
          </div>
        )}

        {/* No data message */}
        {strategies.length === 0 && !loading && (
          <div className="card">
            <h2 className="panel-title">📊 History</h2>
            <div style={{ textAlign: 'center', padding: '40px', color: '#888' }}>
              <p style={{ fontSize: '18px', marginBottom: '10px' }}>No history data available</p>
              <p>Start some experiments to see the history here.</p>
              <button 
                className="btn btn-primary" 
                onClick={() => navigate('/config')}
                style={{ marginTop: '20px' }}
              >
                Go to Config
              </button>
            </div>
          </div>
        )}

        {/* Main Layout: Sidebar + Content */}
        {strategies.length > 0 && (
          <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
            
            {/* Sidebar - Strategy List */}
            <div style={{ width: '280px', flexShrink: 0, marginBottom: '20px' }}>
              <div className="card" style={{ padding: '0' }}>
                <div style={{ 
                  padding: '15px', 
                  borderBottom: '1px solid #2d3348',
                  background: '#1a1d2e'
                }}>
                  <h3 style={{ margin: 0, color: '#4fc3f7' }}>📁 Strategies</h3>
                </div>
                
                <div style={{ maxHeight: '70vh', overflowY: 'auto' }}>
                  {strategies.map((strategy) => (
                    <div 
                      key={strategy.strategy}
                      onClick={() => handleStrategyClick(strategy)}
                      style={{
                        padding: '15px',
                        borderBottom: '1px solid #2d3348',
                        cursor: 'pointer',
                        background: selectedStrategy?.strategy === strategy.strategy ? '#252942' : 'transparent',
                        borderLeft: selectedStrategy?.strategy === strategy.strategy 
                          ? `3px solid ${getStrategyColor(strategy.strategy)}` 
                          : '3px solid transparent',
                        transition: 'all 0.2s'
                      }}
                    >
                      <div style={{ 
                        color: getStrategyColor(strategy.strategy), 
                        fontWeight: '600',
                        marginBottom: '5px'
                      }}>
                        {getStrategyLabel(strategy.strategy)}
                      </div>
                      <div style={{ fontSize: '12px', color: '#888' }}>
                        {strategy.total_experiments} session(s) • {strategy.total_rounds} rounds
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Main Content */}
            <div style={{ flex: 1, minWidth: '300px' }}>
              
              {/* Strategy Overview + Session Selector */}
              {selectedStrategy && (
                <div className="card" style={{ marginBottom: '20px' }}>
                  <h2 className="panel-title">
                    <span style={{ color: getStrategyColor(selectedStrategy.strategy) }}>
                      {getStrategyLabel(selectedStrategy.strategy)}
                    </span>
                  </h2>
                  
                  {/* Sessions list */}
                  <div style={{ marginBottom: '20px' }}>
                    <h4 style={{ color: '#888', marginBottom: '12px' }}>
                      Sessions
                      <span style={{ 
                        marginLeft: '8px', 
                        fontSize: '11px', 
                        color: '#555',
                        fontWeight: 'normal'
                      }}>
                        — one per docker-compose up
                      </span>
                    </h4>

                    <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                      {selectedStrategy.experiments.map((exp) => {
                        const isSelected = selectedExperiment?.experiment_id === exp.experiment_id
                          && selectedExperiment?.session_id === exp.session_id;
                        const color = getStrategyColor(selectedStrategy.strategy);

                        return (
                          <button
                            key={`${exp.session_id ?? 'legacy'}-${exp.experiment_id}`}
                            onClick={() => handleExperimentClick(exp)}
                            style={{
                              padding: '10px 15px',
                              borderRadius: '6px',
                              border: `1px solid ${isSelected ? color : '#2d3348'}`,
                              background: isSelected ? color : '#252942',
                              color: isSelected ? '#1a1d2e' : '#e0e0e0',
                              cursor: 'pointer',
                              fontWeight: '500',
                              transition: 'all 0.2s',
                              textAlign: 'left',
                              minWidth: '150px'
                            }}
                          >
                            {/* Session label */}
                            <div style={{ fontWeight: '600', fontSize: '13px' }}>
                              {exp.is_legacy ? '⚠️ Legacy' : `🖥️ ${getSessionLabel(exp)}`}
                            </div>
                            {/* Started at */}
                            <div style={{ 
                              fontSize: '11px', 
                              marginTop: '3px',
                              opacity: isSelected ? 0.75 : 0.6
                            }}>
                              {getSessionSublabel(exp)}
                            </div>
                            {/* Rounds range */}
                            <div style={{ 
                              fontSize: '11px', 
                              marginTop: '2px',
                              opacity: isSelected ? 0.75 : 0.6
                            }}>
                              Rounds {exp.stats?.first_round ?? 0}–{exp.stats?.last_round ?? 0}
                              &nbsp;·&nbsp;
                              {exp.stats?.total_rounds ?? 0} total
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}

              {/* Experiment / Session Details */}
              {selectedExperiment && selectedStrategy && (
                <>
                  {/* Session info banner */}
                  <div className="card" style={{ 
                    marginBottom: '20px',
                    borderLeft: `3px solid ${getStrategyColor(selectedStrategy.strategy)}`
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '20px', flexWrap: 'wrap' }}>
                      <div>
                        <div style={{ color: '#888', fontSize: '12px', marginBottom: '2px' }}>Session</div>
                        <div style={{ color: '#e0e0e0', fontWeight: '600' }}>
                          {selectedExperiment.is_legacy ? 'Legacy data (no session tracking)' : getSessionLabel(selectedExperiment)}
                        </div>
                      </div>
                      {selectedExperiment.started_at && (
                        <div>
                          <div style={{ color: '#888', fontSize: '12px', marginBottom: '2px' }}>Started at</div>
                          <div style={{ color: '#e0e0e0' }}>{formatTimestamp(selectedExperiment.started_at)}</div>
                        </div>
                      )}
                      {selectedExperiment.hostname && (
                        <div>
                          <div style={{ color: '#888', fontSize: '12px', marginBottom: '2px' }}>Host</div>
                          <div style={{ color: '#e0e0e0' }}>{selectedExperiment.hostname}</div>
                        </div>
                      )}
                      {selectedExperiment.session_id && (
                        <div style={{ marginLeft: 'auto' }}>
                          <div style={{ color: '#888', fontSize: '12px', marginBottom: '2px' }}>Session ID</div>
                          <div style={{ 
                            color: '#555', 
                            fontSize: '11px', 
                            fontFamily: 'monospace',
                            wordBreak: 'break-all'
                          }}>
                            {selectedExperiment.session_id}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Experiment Stats */}
                  <div className="card" style={{ marginBottom: '20px' }}>
                    <h3 className="panel-title">📈 Session Statistics</h3>
                    <div className="grid-4">
                      <div className="info-card">
                        <div className="info-label">Total Rounds</div>
                        <div className="info-value">{selectedExperiment.stats?.total_rounds || 0}</div>
                      </div>
                      <div className="info-card">
                        <div className="info-label">Best Accuracy</div>
                        <div className="info-value" style={{ color: '#66bb6a' }}>
                          {selectedExperiment.stats?.best_accuracy 
                            ? `${selectedExperiment.stats.best_accuracy}%` 
                            : '-'}
                        </div>
                      </div>
                      <div className="info-card">
                        <div className="info-label">Final Accuracy</div>
                        <div className="info-value" style={{ color: '#4fc3f7' }}>
                          {selectedExperiment.stats?.final_accuracy 
                            ? `${selectedExperiment.stats.final_accuracy}%` 
                            : '-'}
                        </div>
                      </div>
                      <div className="info-card">
                        <div className="info-label">Avg Loss</div>
                        <div className="info-value" style={{ color: '#ef5350' }}>
                          {selectedExperiment.stats?.avg_loss 
                            ? selectedExperiment.stats.avg_loss.toFixed(4) 
                            : '-'}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Evolution Charts */}
                  <div className="grid-2" style={{ marginBottom: '20px' }}>
                    {/* Accuracy Evolution */}
                    <div className="card">
                      <h3 className="panel-title">📈 Accuracy Evolution</h3>
                      {selectedExperiment.accuracy_evolution?.length > 0 ? (
                        <ResponsiveContainer width="100%" height={250}>
                          <LineChart data={selectedExperiment.accuracy_evolution}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                            <XAxis dataKey="round" stroke="#888" fontSize={10} />
                            <YAxis stroke="#888" fontSize={10} domain={[0, 100]} />
                            <Tooltip 
                              contentStyle={{ background: '#252942', border: '1px solid #2d3348' }}
                              labelStyle={{ color: '#fff' }}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="accuracy" 
                              stroke={getStrategyColor(selectedStrategy.strategy)} 
                              strokeWidth={2}
                              dot={{ fill: getStrategyColor(selectedStrategy.strategy), strokeWidth: 2 }}
                              activeDot={{ r: 6 }}
                              name="Accuracy (%)"
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <p style={{ color: '#888', textAlign: 'center', padding: '20px' }}>
                          No accuracy data available
                        </p>
                      )}
                    </div>

                    {/* Loss Evolution */}
                    <div className="card">
                      <h3 className="panel-title">📉 Loss Evolution</h3>
                      {selectedExperiment.loss_evolution?.length > 0 ? (
                        <ResponsiveContainer width="100%" height={250}>
                          <LineChart data={selectedExperiment.loss_evolution}>
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
                              dot={{ fill: '#ef5350', strokeWidth: 2 }}
                              activeDot={{ r: 6 }}
                              name="Loss"
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <p style={{ color: '#888', textAlign: 'center', padding: '20px' }}>
                          No loss data available
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Configuration Used */}
                  <div className="card" style={{ marginBottom: '20px' }}>
                    <h3 className="panel-title">⚙️ Configuration Used</h3>
                    {selectedExperiment.config && Object.keys(selectedExperiment.config).length > 0 ? (
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '15px' }}>
                        {Object.entries(selectedExperiment.config).map(([key, value]) => (
                          <div key={key} style={{ 
                            padding: '10px', 
                            background: '#1a1d2e', 
                            borderRadius: '4px',
                            border: '1px solid #2d3348'
                          }}>
                            <div style={{ color: '#888', fontSize: '12px', marginBottom: '5px' }}>{key}</div>
                            <div style={{ color: '#4fc3f7', fontWeight: '500', fontSize: '14px' }}>
                              {value !== null && value !== undefined ? String(value) : '-'}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p style={{ color: '#888' }}>No configuration data available for this session.</p>
                    )}
                  </div>

                  {/* Detailed Rounds Table */}
                  <div className="card">
                    <h3 className="panel-title">📜 Detailed Rounds</h3>
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
                          {selectedExperiment.rounds?.length > 0 ? (
                            selectedExperiment.rounds.map((round, idx) => (
                              <tr key={idx}>
                                <td>{round.round}</td>
                                <td>
                                  {round.accuracy !== null ? (
                                    <span style={{ color: '#66bb6a', fontWeight: '600' }}>
                                      {round.accuracy}%
                                    </span>
                                  ) : '-'}
                                </td>
                                <td>
                                  {round.loss !== null ? (
                                    <span style={{ color: '#ef5350' }}>
                                      {round.loss.toFixed(4)}
                                    </span>
                                  ) : '-'}
                                </td>
                                <td>
                                  {round.agg_time !== null ? (
                                    <span style={{ color: '#ffa726' }}>{round.agg_time}s</span>
                                  ) : '-'}
                                </td>
                                <td>{round.clients || '-'}</td>
                              </tr>
                            ))
                          ) : (
                            <tr>
                              <td colSpan={5} style={{ textAlign: 'center', color: '#888' }}>
                                No round data available
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default History;