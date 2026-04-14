import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  ScatterChart, Scatter, ZAxis, Cell 
} from 'recharts';
import { runDseSweep, getDseStatus, getDseProgress, getDseResults, getDseAllResults, getDseResultsByDataset, getDseSessions, resetDse } from '../services/api';

const AVAILABLE_DATASETS = ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'EMNIST'];

function DSE() {
  const navigate = useNavigate();
  const [selectedDataset, setSelectedDataset] = useState('');
  const [params, setParams] = useState({
    // Key params for sweeping
    learningRate: { min: 0.001, max: 0.1, step: 0.001, value: 0.01 },
    localEpochs: { min: 1, max: 10, step: 1, value: 2 },
    numShortRounds: { min: 1, max: 20, step: 1, value: 5 },
    batchSize: { min: 32, max: 512, step: 64, value: 128 },
    clientsPerRound: { min: 2, max: 20, step: 2, value: 5 },
    xflParam: { min: 1, max: 10, step: 1, value: 3 },
    networkLatency: { min: 0, max: 200, step: 50, value: 50 },
  });
  const [sweepResults, setSweepResults] = useState([]);
  const [resultsByDataset, setResultsByDataset] = useState({});
  const [activeDatasetTab, setActiveDatasetTab] = useState('all');
  const [viewMode, setViewMode] = useState('all'); // 'all' ou 'filtered'
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  const [selectedSession, setSelectedSession] = useState(null);
  const selectedSessionRef = useRef(selectedSession);
  const [progress, setProgress] = useState(0);
  const [searchStrategy, setSearchStrategy] = useState('grid');
  const [maxConfigs, setMaxConfigs] = useState(1);
  const [aggregationStrategy, setAggregationStrategy] = useState('fedavg');
  const [xflStrategyValue, setXflStrategyValue] = useState('xfl_cyclic');
  const [currentSweepSessionId, setCurrentSweepSessionId] = useState(localStorage.getItem('dseSessionId') || null);
  const [currentSweepStatus, setCurrentSweepStatus] = useState(null);
  const [currentSweepProgress, setCurrentSweepProgress] = useState({ completed_configs: 0, total_configs: 0, best_accuracy: 0.0 });
  const [resetting, setResetting] = useState(false);

  useEffect(() => {
    loadSessions();
    const storedSessionId = localStorage.getItem('dseSessionId');
    if (storedSessionId) {
      setCurrentSweepSessionId(storedSessionId);
      pollDseStatus(storedSessionId);
    }
  }, []);

  useEffect(() => {
    selectedSessionRef.current = selectedSession;
  }, [selectedSession]);

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
      await loadDseProgress(sessionId);

      if (status === 'running' || status === 'queued') {
        setTimeout(() => pollDseStatus(sessionId), 5000);
      } else if (status === 'completed') {
        if (selectedSessionRef.current === sessionId || selectedSessionRef.current === null) {
          loadSessionResults(sessionId);
        }
      }
    } catch (error) {
      console.log('DSE status poll failed', error);
    }
  };

  const loadDseProgress = async (sessionId) => {
    try {
      const response = await getDseProgress(sessionId);
      setCurrentSweepProgress(response.data || { completed_configs: 0, total_configs: 0, best_accuracy: 0.0 });
    } catch (error) {
      console.log('DSE progress failed', error);
    }
  };

  const loadSessions = async () => {
    try {
      const response = await getDseSessions();
      const sessionList = (response.data.sessions || []).map((session, index) => ({
        ...session,
        label: `exp ${index + 1}`,
      }));
      setSessions(sessionList);
      if (!selectedSession && sessionList.length > 0) {
        loadAllSessionResults(sessionList);
      }
    } catch (error) {
      console.log('No DSE sessions found, backend API may need implementation');
    }
  };

  const handleParamChange = (param, value) => {
    setParams(prev => ({ ...prev, [param]: { ...prev[param], value } }));
  };

  const handleSweep = async () => {
    if (!selectedDataset) {
      setMessage({ type: 'error', text: 'Veuillez sélectionner un dataset avant de lancer le sweep.' });
      return;
    }

    setLoading(true);
    setMessage({ type: '', text: '' });
    setProgress(0);

    try {
      // Prepare sweep config using slider-selected values
      const sweepParams = Object.fromEntries(
        Object.entries(params).map(([key, range]) => [
          key,
          [range.value]
        ])
      );

      const sweepConfig = {
        params: {
          ...sweepParams,
          strategy: [aggregationStrategy],
          xflStrategy: [xflStrategyValue]
        },
        numShortRounds: Number(params.numShortRounds.value),
        dataset: selectedDataset,
        searchStrategy,
        maxConfigs,
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
      const results = response.data.results || [];
      setSweepResults(results);
      setSelectedSession(sessionId);
      
      // Group results by dataset for filtering
      const grouped = {};
      results.forEach(r => {
        const dataset = r.config?.dataset || 'Unknown';
        if (!grouped[dataset]) {
          grouped[dataset] = [];
        }
        grouped[dataset].push(r);
      });
      setResultsByDataset(grouped);
      setActiveDatasetTab('all');
      setViewMode('all');
      
      setMessage({ type: 'success', text: `Loaded results from ${sessionId}` });
      if (currentSweepSessionId === sessionId) {
        setCurrentSweepStatus('completed');
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to load session results' });
    }
  };

  const loadAllSessionResults = async (sessionList = null) => {
    let targetSessions = sessionList || sessions;

    if (!targetSessions.length) {
      try {
        const response = await getDseSessions();
        targetSessions = response.data.sessions || [];
        setSessions(targetSessions);
      } catch (error) {
        setMessage({ type: 'error', text: 'Aucune session DSE disponible pour charger.' });
        return;
      }
    }

    if (!targetSessions.length) {
      setMessage({ type: 'error', text: 'Aucune session DSE disponible pour charger.' });
      return;
    }

    try {
      // Try to load results grouped by dataset
      const response = await getDseResultsByDataset();
      const resultsByDatasetData = response.data.results_by_dataset || {};
      setResultsByDataset(resultsByDatasetData);
      setActiveDatasetTab('all');
      setViewMode('all');
      
      // Flatten results for backwards compatibility
      const allResults = [];
      for (const datasetResults of Object.values(resultsByDatasetData)) {
        allResults.push(...datasetResults);
      }
      setSweepResults(allResults);
      setSelectedSession('all');
      
      const numDatasets = response.data.total_datasets || 0;
      const totalResults = response.data.total_results || allResults.length;
      setMessage({ type: 'success', text: `Chargé résultats DSE groupés par dataset (${numDatasets} datasets, ${totalResults} configs)` });
      return;
    } catch (error) {
      console.warn('getDseResultsByDataset failed, falling back to all results fetch', error);
    }

    try {
      const response = await getDseAllResults();
      const allResults = response.data.results || [];
      
      // Group results by dataset for filtering
      const grouped = {};
      allResults.forEach(r => {
        const dataset = r.config?.dataset || 'Unknown';
        if (!grouped[dataset]) {
          grouped[dataset] = [];
        }
        grouped[dataset].push(r);
      });
      setResultsByDataset(grouped);
      setActiveDatasetTab('all');
      setViewMode('all');
      
      setSweepResults(allResults);
      setSelectedSession('all');
      setMessage({ type: 'success', text: `Chargé toutes les sessions DSE (${response.data.session_count || targetSessions.length} sessions, ${allResults.length} configs)` });
      return;
    } catch (error) {
      console.warn('getDseAllResults failed, falling back to per-session fetch', error);
    }

    try {
      const responses = await Promise.allSettled(
        targetSessions.map((session) => getDseResults(session.id))
      );
      const allResults = responses.reduce((acc, result) => {
        if (result.status === 'fulfilled') {
          acc.push(...(result.value.data.results || []));
        }
        return acc;
      }, []);

      // Group results by dataset for filtering
      const grouped = {};
      allResults.forEach(r => {
        const dataset = r.config?.dataset || 'Unknown';
        if (!grouped[dataset]) {
          grouped[dataset] = [];
        }
        grouped[dataset].push(r);
      });
      setResultsByDataset(grouped);
      setActiveDatasetTab('all');
      setViewMode('all');

      setSweepResults(allResults);
      setSelectedSession('all');
      setMessage({ type: 'success', text: `Chargé toutes les sessions DSE (${targetSessions.length} sessions, ${allResults.length} configs)` });

      const failed = responses.filter((r) => r.status === 'rejected');
      if (failed.length > 0) {
        console.warn(`Failed to load ${failed.length} session(s)`);
      }
    } catch (error) {
      let detail = '';
      if (error?.message) {
        detail = `: ${error.message}`;
      }
      setMessage({ type: 'error', text: `Échec du chargement de toutes les sessions DSE${detail}` });
    }
  };

  const handleResetData = async () => {
    setResetting(true);
    setMessage({ type: '', text: '' });

    try {
      const response = await resetDse();
      if (response.data && response.data.status === 'ok') {
        setMessage({ type: 'success', text: 'Données DSE réinitialisées.' });
        setSweepResults([]);
        setSessions([]);
        setSelectedSession(null);
        clearSweepSessionId();
        loadSessions();
      } else {
        setMessage({ type: 'error', text: 'Impossible de réinitialiser les données DSE.' });
      }
    } catch (err) {
      console.error('Reset failed:', err);
      setMessage({ type: 'error', text: 'Échec de la réinitialisation : ' + (err.response?.data?.error || err.message) });
    } finally {
      setResetting(false);
    }
  };

  // Transform results for charts (accuracy vs lr example)
  // Filter results based on view mode
  let filteredResults;
  
  if (viewMode === 'filtered' && activeDatasetTab !== 'all') {
    // Filtered mode: show only results from selected dataset
    filteredResults = sweepResults.filter(r => {
      const resultDataset = r.config?.dataset;
      return resultDataset === activeDatasetTab;
    });
  } else {
    // All mode: show all results
    filteredResults = sweepResults;
  }

  const accuracyVsLrData = filteredResults.map((r, index) => ({
    experience: `exp ${index + 1}`,
    session_id: r.session_id || r.sessionId || r.config?.session_id || 'unknown',
    lr: r.config.learningRate,
    accuracy: r.metrics?.final_accuracy || 0,
    loss: r.metrics?.final_loss || 0,
    time: r.metrics?.total_time || 0,
    info: `lr=${r.config.learningRate}, epochs=${r.config.localEpochs}, batch=${r.config.batchSize}`,
  }));

  const clientsTimeData = filteredResults.map(r => ({
    clientsPerRound: r.config.clientsPerRound || 0,
    time: r.metrics?.total_time || 0,
    accuracy: r.metrics?.final_accuracy || 0,
  }));

  const accuracyVsTimeData = filteredResults.map(r => ({
    time: r.metrics?.total_time || 0,
    accuracy: r.metrics?.final_accuracy || 0,
    label: `lr=${r.config.learningRate}, epochs=${r.config.localEpochs}`,
  }));

  const bestConfig = filteredResults.reduce((best, r) => {
    if (!best) return r;
    return (parseFloat(r.metrics?.final_accuracy) || 0) > (parseFloat(best.metrics?.final_accuracy) || 0) ? r : best;
  }, null);

  const topConfigs = [...filteredResults]
    .sort((a, b) => (parseFloat(b.metrics?.final_accuracy) || 0) - (parseFloat(a.metrics?.final_accuracy) || 0))
    .slice(0, 3);

  const bestAccuracyFromResults = filteredResults.reduce((best, r) => {
    const accuracy = parseFloat(r.metrics?.final_accuracy) || 0;
    return accuracy > best ? accuracy : best;
  }, 0);

  const displayBestAccuracy = Math.max(
    bestAccuracyFromResults,
    currentSweepProgress.best_accuracy || 0
  );

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
          <button className="btn btn-secondary" onClick={handleResetData} disabled={resetting} style={{ marginRight: '10px' }}>
            {resetting ? 'Réinitialisation...' : 'Réinitialiser'}
          </button>
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

          {/* Dataset Selection - REQUIRED */}
          <div className="card" style={{ marginBottom: '20px', borderLeft: '4px solid #db1515', backgroundColor: selectedDataset ? '#2d3348' : '#ffffff' }}>
            <h2 className="panel-title" style={{color: selectedDataset ? '#ffffff' : '#000000', fontWeight: 'bold'}}>📊 Select Dataset (Required)</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '15px', marginTop: '15px' }}>
              {AVAILABLE_DATASETS.map((dataset) => (
                <label key={dataset} style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', padding: '10px', borderRadius: '4px', backgroundColor: selectedDataset === dataset ? '#bbdefb' : 'transparent', transition: 'background-color 0.2s' }}>
                  <input
                    type="radio"
                    name="dataset"
                    value={dataset}
                    checked={selectedDataset === dataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    style={{ marginRight: '10px', cursor: 'pointer' }}
                  />
                  <span style={{ fontWeight: selectedDataset === dataset ? 'bold' : 'normal' }}>{dataset}</span>
                </label>
              ))}
            </div>
            {!selectedDataset && (
              <div style={{ color: '#d32f2f', marginTop: '10px', fontWeight: 'bold' }}>⚠️ Un dataset doit être sélectionné pour procéder</div>
            )}
          </div>

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
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '15px', marginTop: '15px' }}>
            <div className="form-group">
              <label>Search strategy</label>
              <select value={searchStrategy} onChange={(e) => setSearchStrategy(e.target.value)}>
                <option value="grid">Grid search</option>
                <option value="random">Random search</option>
              </select>
            </div>
            <div className="form-group">
              <label>Aggregation strategy</label>
              <select value={aggregationStrategy} onChange={(e) => setAggregationStrategy(e.target.value)}>
                <option value="fedavg">FedAvg</option>
                <option value="xfl">XFL</option>
              </select>
            </div>
            {aggregationStrategy === 'xfl' && (
              <div className="form-group">
                <label>XFL variant</label>
                <select value={xflStrategyValue} onChange={(e) => setXflStrategyValue(e.target.value)}>
                  <option value="xfl_cyclic">XFL cyclic</option>
                  <option value="xfl_sparsification">XFL sparsification</option>
                  <option value="xfl_quantization">XFL quantization</option>
                </select>
              </div>
            )}
            <div className="form-group">
              <label>Max configs</label>
              <input
                type="number"
                min={1}
                max={10}
                value={maxConfigs}
                onChange={(e) => setMaxConfigs(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          </div>
          <button 
              className="btn btn-primary" 
              onClick={handleSweep}
              disabled={loading || !selectedDataset}
              style={{ marginTop: '15px', opacity: !selectedDataset ? 0.5 : 1 }}
              title={!selectedDataset ? 'Veuillez sélectionner un dataset' : ''}
            >
              {loading ? 'Running Sweep...' : 'Run Design Space Exploration'}
            </button>
            {currentSweepSessionId && (
              <div style={{ marginTop: '15px', color: '#fff' }}>
                <strong>Current session:</strong> {currentSweepSessionId}
                <br />
                <strong>Statut:</strong> {currentSweepStatus || 'unknown'}
                <br />
                <strong>Progression:</strong> {currentSweepProgress.completed_configs}/{currentSweepProgress.total_configs} configs
                <br />
                <strong>Best accuracy:</strong> {displayBestAccuracy.toFixed(2)}%
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
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginBottom: '10px' }}>
                <button
                  className="btn"
                  onClick={loadAllSessionResults}
                  disabled={sessions.length === 0}
                  style={{ background: selectedSession === 'all' ? '#4fc3f7' : '#252942', opacity: sessions.length === 0 ? 0.6 : 1 }}
                >
                  Toutes les sessions
                </button>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px' }}>
                {sessions.map(session => (
                  <button 
                    key={session.id}
                    className="btn"
                    onClick={() => loadSessionResults(session.id)}
                    style={{ background: selectedSession === session.id ? '#4fc3f7' : '#252942' }}
                  >
                    {session.label || session.id} ({session.id}) — {session.numConfigs} configs
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Results Visualization */}
          {sweepResults.length > 0 && (
            <>
              {/* View Mode Toggle */}
              <div className="card" style={{ marginBottom: '20px', borderLeft: '4px solid #ff9800' }}>
                <h2 className="panel-title">👁️ Affichage des Résultats</h2>
                <div style={{ display: 'flex', gap: '15px', marginTop: '15px', flexWrap: 'wrap' }}>
                  <button
                    onClick={() => setViewMode('all')}
                    style={{
                      background: viewMode === 'all' ? '#4fc3f7' : '#252942',
                      color: '#fff',
                      border: viewMode === 'all' ? '2px solid #4fc3f7' : '1px solid #555',
                      padding: '12px 20px',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontWeight: viewMode === 'all' ? 'bold' : 'normal',
                      fontSize: '14px',
                      transition: 'all 0.3s'
                    }}
                  >
                    📊 Tous les Résultats
                  </button>
                  <button
                    onClick={() => setViewMode('filtered')}
                    style={{
                      background: viewMode === 'filtered' ? '#4fc3f7' : '#252942',
                      color: '#fff',
                      border: viewMode === 'filtered' ? '2px solid #4fc3f7' : '1px solid #555',
                      padding: '12px 20px',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontWeight: viewMode === 'filtered' ? 'bold' : 'normal',
                      fontSize: '14px',
                      transition: 'all 0.3s'
                    }}
                  >
                    🔍 Filtrer par Dataset
                  </button>
                </div>

                {/* Dataset Selection in Filtered Mode */}
                {viewMode === 'filtered' && Object.keys(resultsByDataset).length > 0 && (
                  <div style={{ marginTop: '20px' }}>
                    <p style={{ color: '#4fc3f7', fontWeight: 'bold', marginBottom: '10px' }}>
                      📌 Sélectionnez un dataset pour voir ses résultats :
                    </p>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
                      {Object.keys(resultsByDataset).map(dataset => (
                        <button
                          key={dataset}
                          onClick={() => setActiveDatasetTab(dataset)}
                          style={{
                            background: activeDatasetTab === dataset ? '#1a237e' : '#252942',
                            color: activeDatasetTab === dataset ? '#4fc3f7' : '#fff',
                            border: activeDatasetTab === dataset ? '3px solid #4fc3f7' : '1px solid #555',
                            padding: '15px',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            fontWeight: activeDatasetTab === dataset ? 'bold' : 'normal',
                            fontSize: '15px',
                            transition: 'all 0.3s',
                            boxShadow: activeDatasetTab === dataset ? '0 0 10px rgba(79, 195, 247, 0.3)' : 'none'
                          }}
                        >
                          <div style={{ fontSize: '20px', marginBottom: '5px' }}>📁</div>
                          {dataset}
                          <div style={{ fontSize: '12px', marginTop: '5px', opacity: 0.8 }}>
                            {resultsByDataset[dataset].length} configs
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Info Message */}
                {viewMode === 'all' && (
                  <div style={{ marginTop: '15px', padding: '12px', background: '#1d2436', borderLeft: '3px solid #4fc3f7', borderRadius: '4px', color: '#4fc3f7' }}>
                    ℹ️ Mode "Tous les Résultats": Affichage de toutes les configurations testées (tous les datasets mélangés)
                  </div>
                )}
                {viewMode === 'filtered' && (
                  <div style={{ marginTop: '15px', padding: '12px', background: '#1d2436', borderLeft: '3px solid #ff9800', borderRadius: '4px', color: '#ff9800' }}>
                    ℹ️ Mode "Filtrer par Dataset": Affichage des configurations du dataset sélectionné uniquement
                  </div>
                )}
              </div>

              <div className="grid-2" style={{ marginBottom: '20px' }}>
                {/* Accuracy vs Learning Rate */}
                <div className="card">
                  <h2 className="panel-title">📈 Accuracy vs Learning Rate {viewMode === 'filtered' && activeDatasetTab !== 'all' && `(${activeDatasetTab})`}</h2>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={accuracyVsLrData} margin={{ top: 20, right: 20, left: 0, bottom: 70 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                      <XAxis dataKey="experience" stroke="#888" interval={0} tick={{ fontSize: 10, angle: -35, textAnchor: 'end' }} height={70} />
                      <YAxis stroke="#888" />
                      <Tooltip formatter={(value, name) => [value, name === 'accuracy' ? 'Accuracy (%)' : 'Loss']} labelFormatter={(label) => `Experience: ${label}`} />
                      <Bar dataKey="accuracy" fill="#66bb6a" barSize={20} />
                      <Bar dataKey="loss" fill="#ef5350" barSize={20} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Accuracy vs Total Time */}
                <div className="card">
                  <h2 className="panel-title">⏱️ Accuracy vs Total Training Time {viewMode === 'filtered' && activeDatasetTab !== 'all' && `(${activeDatasetTab})`}</h2>
                  <ResponsiveContainer width="100%" height={250}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" />
                      <XAxis dataKey="time" name="Time (s)" stroke="#888" />
                      <YAxis dataKey="accuracy" name="Accuracy (%)" stroke="#888" />
                      <ZAxis dataKey="accuracy" range={[64, 256]} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(value) => [value, '']} />
                      <Scatter data={accuracyVsTimeData} fill="#4fc3f7">
                        {accuracyVsTimeData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={`hsl(${entry.accuracy * 240 / 100}, 70%, 55%)`} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Best Configuration Summary */}
              {bestConfig && (
                <div className="card" style={{ marginBottom: '20px' }}>
                  <h2 className="panel-title">🏆 Best Configuration {viewMode === 'filtered' && activeDatasetTab !== 'all' ? `(${activeDatasetTab})` : ''}</h2>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
                    {viewMode === 'filtered' && activeDatasetTab !== 'all' && (
                      <div style={{ gridColumn: '1 / -1', color: '#ff9800', fontSize: '12px', fontStyle: 'italic', backgroundColor: '#1d2436', padding: '8px', borderRadius: '4px' }}>
                        📌 Affichage filtré pour le dataset: <strong>{activeDatasetTab}</strong>
                      </div>
                    )}
                    {viewMode === 'all' && (
                      <div style={{ gridColumn: '1 / -1', color: '#4fc3f7', fontSize: '12px', fontStyle: 'italic', backgroundColor: '#1d2436', padding: '8px', borderRadius: '4px' }}>
                        📊 Affichage pour tous les résultats (tous les datasets confondus)
                      </div>
                    )}
                    <div><strong>Learning Rate</strong><br />{bestConfig.config.learningRate}</div>
                    <div><strong>Epochs</strong><br />{bestConfig.config.localEpochs}</div>
                    <div><strong>Batch Size</strong><br />{bestConfig.config.batchSize}</div>
                    <div><strong>Strategy</strong><br />{bestConfig.config.strategy || '-'}</div>
                    <div><strong>XFL variant</strong><br />{bestConfig.config.xflStrategy || '-'}</div>
                    <div><strong>Clients/Round</strong><br />{bestConfig.config.clientsPerRound || '-'}</div>
                    <div><strong>Accuracy</strong><br />{bestConfig.metrics?.final_accuracy?.toFixed(2) || '-'}%</div>
                    <div><strong>Time</strong><br />{bestConfig.metrics?.total_time?.toFixed(1) || '-'} s</div>
                  </div>
                </div>
              )}

              {/* Top 3 Configurations */}
              {topConfigs.length > 0 && (
                <div className="card" style={{ marginBottom: '20px' }}>
                  <h2 className="panel-title">🔝 Top 3 Configurations {viewMode === 'filtered' && activeDatasetTab !== 'all' ? `(${activeDatasetTab})` : ''}</h2>
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>LR</th>
                        <th>Epochs</th>
                        <th>Batch</th>
                        <th>Strategy</th>
                        <th>Acc (%)</th>
                        <th>Time (s)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {topConfigs.map((r, idx) => (
                        <tr key={idx}>
                          <td>{idx + 1}</td>
                          <td>{r.config.learningRate?.toFixed(4)}</td>
                          <td>{r.config.localEpochs}</td>
                          <td>{r.config.batchSize}</td>
                          <td>{r.config.strategy || '-'}</td>
                          <td>{r.metrics?.final_accuracy?.toFixed(2) || '-'}%</td>
                          <td>{r.metrics?.total_time?.toFixed(1) || '-'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {/* Summary Table */}
              <div className="card">
                <h2 className="panel-title">📊 Sweep Summary ({filteredResults.length} configs{viewMode === 'filtered' && activeDatasetTab !== 'all' && `, Dataset: ${activeDatasetTab}`})</h2>
                {filteredResults.length === 0 ? (
                  <div style={{ padding: '20px', textAlign: 'center', color: '#ff9800' }}>
                    <p>⚠️ Aucun résultat trouvé pour {viewMode === 'filtered' && activeDatasetTab !== 'all' ? `le dataset ${activeDatasetTab}` : 'les datasets sélectionnés'}</p>
                    <p style={{ fontSize: '12px', color: '#888' }}>Assurez-vous d'avoir exécuté le sweep avec ce dataset</p>
                  </div>
                ) : (
                  <div style={{ overflowX: 'auto' }}>
                    <table>
                      <thead>
                        <tr>
                          <th>LR</th>
                          <th>Epochs</th>
                          <th>Batch</th>
                          <th>Strategy</th>
                          <th>Acc (%)</th>
                          <th>Loss</th>
                          <th>Time (s)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredResults.slice(-10).map((r, i) => (
                          <tr key={i}>
                            <td>{r.config.learningRate?.toFixed(3)}</td>
                            <td>{r.config.localEpochs}</td>
                            <td>{r.config.batchSize}</td>
                            <td>{r.config.strategy || '-'}</td>
                            <td>{r.metrics?.final_accuracy?.toFixed(1) || '-'}%</td>
                            <td>{r.metrics?.final_loss?.toFixed(4) || '-'}</td>
                            <td>{r.metrics?.total_time?.toFixed(1) || '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

export default DSE;

