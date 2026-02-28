import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { getRoundsHistory, getAccuracyData, getLossData, exportData } from '../services/api';

function History() {
  const navigate = useNavigate();
  const [history, setHistory] = useState([]);
  const [accuracyData, setAccuracyData] = useState({ rounds: [], accuracy: [] });
  const [lossData, setLossData] = useState({ rounds: [], loss: [] });
  const [filter, setFilter] = useState('all');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  const [lastUpdated, setLastUpdated] = useState(null);
  const [dataError, setDataError] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);

  useEffect(() => {
    // Initial fetch
    fetchData();
    
    // Set up polling interval - fetch every 5 seconds
    const interval = setInterval(fetchData, 5000);
    
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

  const fetchData = async () => {
    try {
      setDataError(false);
      
      // Fetch data individually with error handling to preserve existing data on partial failures
      let historyRes, accuracyRes, lossRes;
      
      try {
        historyRes = await getRoundsHistory();
      } catch (e) {
        console.warn('History fetch failed:', e.message);
        historyRes = { data: null };
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

      // FIXED: Always exit initial loading state after first successful fetch attempt
      // This prevents getting stuck on "Loading..." even if no data exists yet
      setInitialLoading(false);
      
      // FIXED: Only update state if we have valid data with actual content - preserve existing data
      // History data - check for non-empty rounds array
      if (historyRes && historyRes.data && historyRes.data.rounds && historyRes.data.rounds.length > 0) {
        setHistory(historyRes.data.rounds);
      }
      
      // Accuracy data - check for non-empty rounds array
      if (accuracyRes && accuracyRes.data && accuracyRes.data.rounds && accuracyRes.data.rounds.length > 0) {
        setAccuracyData(accuracyRes.data);
      }
      
      // Loss data - check for non-empty rounds array
      if (lossRes && lossRes.data && lossRes.data.rounds && lossRes.data.rounds.length > 0) {
        setLossData(lossRes.data);
      }
      
      setLastUpdated(new Date());
    } catch (error) {
      console.error('Error fetching data:', error);
      setDataError(true);
      setInitialLoading(false);
      // Don't clear existing data on error - preserve it
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('username');
    navigate('/login');
  };

  const handleExport = async () => {
    setLoading(true);
    try {
      await exportData();
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

  // Filter history based on selection
  const filteredHistory = filter === 'all' 
    ? history 
    : history.filter(r => r.round <= parseInt(filter));

  // Group rounds by strategy and calculate statistics
  const strategyStats = React.useMemo(() => {
    const stats = {};
    
    history.forEach(round => {
      if (!round.strategy) return;
      
      if (!stats[round.strategy]) {
        stats[round.strategy] = {
          strategy: round.strategy,
          rounds: [],
          accuracies: [],
          losses: [],
          aggTimes: []
        };
      }
      
      if (round.accuracy !== null && round.accuracy !== undefined) {
        stats[round.strategy].rounds.push(round.round);
        stats[round.strategy].accuracies.push(round.accuracy);
      }
      if (round.loss !== null && round.loss !== undefined) {
        stats[round.strategy].losses.push(round.loss);
      }
      if (round.agg_time !== null && round.agg_time !== undefined) {
        stats[round.strategy].aggTimes.push(round.agg_time);
      }
    });

    // Calculate aggregated statistics
    return Object.values(stats).map(s => {
      const validAccuracies = s.accuracies.filter(a => a !== null);
      const validLosses = s.losses.filter(l => l !== null);
      const validAggTimes = s.aggTimes.filter(t => t !== null);
      
      return {
        strategy: s.strategy,
        numRounds: s.rounds.length,
        bestAccuracy: validAccuracies.length > 0 ? Math.max(...validAccuracies) : 0,
        worstLoss: validLosses.length > 0 ? Math.max(...validLosses) : 0,
        avgAccuracy: validAccuracies.length > 0 ? validAccuracies.reduce((a, b) => a + b, 0) / validAccuracies.length : 0,
        avgLoss: validLosses.length > 0 ? validLosses.reduce((a, b) => a + b, 0) / validLosses.length : 0,
        avgAggTime: validAggTimes.length > 0 ? validAggTimes.reduce((a, b) => a + b, 0) / validAggTimes.length : 0,
        firstRound: s.rounds.length > 0 ? Math.min(...s.rounds) : 0,
        lastRound: s.rounds.length > 0 ? Math.max(...s.rounds) : 0
      };
    }).sort((a, b) => b.numRounds - a.numRounds);
  }, [history]);

  // Prepare bar chart data for strategy comparison
  const strategyComparisonData = strategyStats.map(s => ({
    strategy: s.strategy,
    avgAccuracy: parseFloat(s.avgAccuracy.toFixed(2)),
    bestAccuracy: parseFloat(s.bestAccuracy.toFixed(2)),
    numRounds: s.numRounds
  }));

  // Prepare line chart data for accuracy evolution by strategy
  const strategyAccuracyEvolution = React.useMemo(() => {
    const evolutionByStrategy = {};
    
    history.forEach(round => {
      if (!round.strategy || round.accuracy === null || round.accuracy === undefined) return;
      
      if (!evolutionByStrategy[round.strategy]) {
        evolutionByStrategy[round.strategy] = [];
      }
      
      evolutionByStrategy[round.strategy].push({
        round: round.round,
        accuracy: round.accuracy
      });
    });

    // Sort each strategy's data by round
    Object.keys(evolutionByStrategy).forEach(strategy => {
      evolutionByStrategy[strategy].sort((a, b) => a.round - b.round);
    });

    return evolutionByStrategy;
  }, [history]);

  // Calculate statistics
  const stats = {
    totalRounds: history.length,
    avgAccuracy: history.length > 0 
      ? (history.reduce((sum, r) => sum + (r.accuracy || 0), 0) / history.length).toFixed(2)
      : 0,
    avgLoss: history.length > 0
      ? (history.reduce((sum, r) => sum + (r.loss || 0), 0) / history.length).toFixed(4)
      : 0,
    bestAccuracy: history.length > 0
      ? Math.max(...history.map(r => r.accuracy || 0)).toFixed(2)
      : 0,
  };

  // Strategy colors for charts
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

  const getStrategyColor = (strategy) => {
    return strategyColors[strategy] || '#888888';
  };

  // Show loading indicator during initial load
  if (initialLoading) {
    return (
      <div className="page-container">
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

      {/* Main Content */}
      <main className="main-content">
        {/* Message Display */}
        {message.text && (
          <div style={{ 
            padding: '15px', 
            marginBottom: '20px', 
            borderRadius: '4px',
            background: message.type === 'success' ? '#66bb6a' : '#ef5350',
            color: 'white'
          }}>
            {message.text}
          </div>
        )}

        {/* Statistics Cards */}
        <div className="grid-4" style={{ marginBottom: '20px' }}>
          <div className="info-card">
            <div className="info-label">Total Rounds</div>
            <div className="info-value">{stats.totalRounds}</div>
          </div>
          <div className="info-card">
            <div className="info-label">Average Accuracy</div>
            <div className="info-value">{stats.avgAccuracy}%</div>
          </div>
          <div className="info-card">
            <div className="info-label">Average Loss</div>
            <div className="info-value">{stats.avgLoss}</div>
          </div>
          <div className="info-card">
            <div className="info-label">Best Accuracy</div>
            <div className="info-value">{stats.bestAccuracy}%</div>
          </div>
        </div>

        {/* Strategy Analysis Section */}
        <div className="card" style={{ marginBottom: '20px' }}>
          <h2 className="panel-title">📊 Strategy Analysis</h2>
          <p style={{ color: '#888', marginBottom: '20px' }}>
            This section shows the performance metrics grouped by XFL strategy used during training rounds.
          </p>
          
          {/* Strategy Comparison Bar Chart */}
          <div style={{ marginBottom: '30px' }}>
            <h3 style={{ color: '#4fc3f7', marginBottom: '15px' }}>Average Accuracy by Strategy</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={strategyComparisonData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                <XAxis dataKey="strategy" stroke="#888" fontSize={10} angle={-45} textAnchor="end" height={80} />
                <YAxis stroke="#888" fontSize={10} />
                <Tooltip 
                  contentStyle={{ background: '#252942', border: '1px solid #2d3348' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Bar dataKey="avgAccuracy" name="Avg Accuracy (%)" fill="#4fc3f7" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Strategy Statistics Table */}
          <h3 style={{ color: '#4fc3f7', marginBottom: '15px' }}>Strategy Performance Summary</h3>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Rounds</th>
                  <th>Best Accuracy (%)</th>
                  <th>Worst Loss</th>
                  <th>Avg Accuracy (%)</th>
                  <th>Avg Loss</th>
                  <th>Avg Agg Time (s)</th>
                  <th>First Round</th>
                  <th>Last Round</th>
                </tr>
              </thead>
              <tbody>
                {strategyStats.length > 0 ? (
                  strategyStats.map((stat, idx) => (
                    <tr key={idx}>
                      <td>
                        <span style={{ 
                          color: getStrategyColor(stat.strategy), 
                          fontWeight: '600' 
                        }}>
                          {stat.strategy}
                        </span>
                      </td>
                      <td>{stat.numRounds}</td>
                      <td>
                        <span style={{ color: '#66bb6a', fontWeight: '600' }}>
                          {stat.bestAccuracy.toFixed(2)}%
                        </span>
                      </td>
                      <td>
                        <span style={{ color: '#ef5350' }}>
                          {stat.worstLoss.toFixed(4)}
                        </span>
                      </td>
                      <td>{stat.avgAccuracy.toFixed(2)}%</td>
                      <td>{stat.avgLoss.toFixed(4)}</td>
                      <td>{stat.avgAggTime.toFixed(2)}s</td>
                      <td>{stat.firstRound}</td>
                      <td>{stat.lastRound}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={9} style={{ textAlign: 'center', color: '#888' }}>
                      No strategy data available
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Accuracy Evolution by Strategy */}
        {Object.keys(strategyAccuracyEvolution).length > 0 && (
          <div className="card" style={{ marginBottom: '20px' }}>
            <h2 className="panel-title">📈 Accuracy Evolution by Strategy</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                <XAxis dataKey="round" stroke="#888" fontSize={10} />
                <YAxis stroke="#888" fontSize={10} domain={[0, 100]} />
                <Tooltip 
                  contentStyle={{ background: '#252942', border: '1px solid #2d3348' }}
                  labelStyle={{ color: '#fff' }}
                />
                <Legend />
                {Object.entries(strategyAccuracyEvolution).map(([strategy, data]) => (
                  <Line 
                    key={strategy}
                    type="monotone" 
                    data={data}
                    dataKey="accuracy" 
                    name={strategy}
                    stroke={getStrategyColor(strategy)}
                    strokeWidth={2}
                    dot={{ fill: getStrategyColor(strategy), strokeWidth: 2 }}
                    activeDot={{ r: 6 }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Filter and Export */}
        <div className="card" style={{ marginBottom: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '15px' }}>
            <h2 className="panel-title" style={{ marginBottom: 0, borderBottom: 'none', paddingBottom: 0 }}>
              Metrics History
            </h2>
            
            <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <label style={{ marginBottom: 0 }}>Filter by Rounds:</label>
                <select 
                  value={filter}
                  onChange={(e) => setFilter(e.target.value)}
                  style={{ width: 'auto' }}
                >
                  <option value="all">All Rounds</option>
                  <option value="10">Last 10 Rounds</option>
                  <option value="20">Last 20 Rounds</option>
                  <option value="30">Last 30 Rounds</option>
                </select>
              </div>

              <button 
                className="btn btn-primary"
                onClick={handleExport}
                disabled={loading}
              >
                💾 Export to CSV
              </button>
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="grid-2" style={{ marginBottom: '20px' }}>
          <div className="card">
            <h2 className="panel-title">▲ Accuracy Over Time</h2>
            <ResponsiveContainer width="100%" height={250}>
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
                  dot={{ fill: '#4fc3f7', strokeWidth: 2 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h2 className="panel-title">📉 Loss Over Time</h2>
            <ResponsiveContainer width="100%" height={250}>
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
                  dot={{ fill: '#ef5350', strokeWidth: 2 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Detailed History Table */}
        <div className="card">
          <h2 className="panel-title">📜 Detailed Rounds History</h2>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Round</th>
                  <th>Strategy</th>
                  <th>Global Test Accuracy</th>
                  <th>Global Test Loss</th>
                  <th>Aggregation Time (s)</th>
                  <th>Number of Clients</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredHistory.length > 0 ? (
                  filteredHistory.map((round, idx) => (
                    <tr key={idx}>
                      <td>{round.round}</td>
                      <td>
                        <span style={{ 
                          color: getStrategyColor(round.strategy), 
                          fontWeight: '600' 
                        }}>
                          {round.strategy || 'all_layers'}
                        </span>
                      </td>
                      <td>
                        {round.accuracy ? (
                          <span style={{ color: '#66bb6a', fontWeight: '600' }}>
                            {round.accuracy.toFixed(2)}%
                          </span>
                        ) : '-'}
                      </td>
                      <td>
                        {round.loss ? (
                          <span style={{ color: '#ef5350' }}>
                            {round.loss.toFixed(4)}
                          </span>
                        ) : '-'}
                      </td>
                      <td>
                        {round.agg_time ? (
                          <span style={{ color: '#ffa726' }}>
                            {round.agg_time.toFixed(2)}s
                          </span>
                        ) : '-'}
                      </td>
                      <td>{round.clients || '-'}</td>
                      <td>
                        {round.accuracy ? (
                          <span className="status-badge success">Completed</span>
                        ) : (
                          <span className="status-badge warning">In Progress</span>
                        )}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={7} style={{ textAlign: 'center', color: '#888' }}>
                      No history data available
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Strategy Comparison Info */}
        <div className="card" style={{ marginTop: '20px' }}>
          <h2 className="panel-title">📊 Strategy Comparison Notes</h2>
          <div style={{ color: '#888', fontSize: '14px', lineHeight: '1.8' }}>
            <p style={{ marginBottom: '10px' }}>
              <strong style={{ color: '#4fc3f7' }}>FedAvg (All Layers):</strong> Standard federated averaging that aggregates all model layers from clients.
            </p>
            <p style={{ marginBottom: '10px' }}>
              <strong style={{ color: '#4fc3f7' }}>XFL - Cyclic:</strong> Clients send only one layer per round, selected cyclically. Reduces communication overhead.
            </p>
            <p style={{ marginBottom: '10px' }}>
              <strong style={{ color: '#4fc3f7' }}>XFL - Sparsification:</strong> Only top-K important parameters are sent, reducing bandwidth.
            </p>
            <p style={{ marginBottom: '10px' }}>
              <strong style={{ color: '#4fc3f7' }}>XFL - Quantization:</strong> Model weights are quantized before transmission to reduce data size.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default History;
