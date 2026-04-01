import axios from 'axios';

const API_BASE_URL = '/api';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Ensure axios throws on HTTP error status codes (4xx, 5xx)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // If the server responded with an error status, still throw the error
    // so it can be caught by the calling code
    if (error.response) {
      // Server responded with error status
      return Promise.reject(error);
    } else if (error.request) {
      // Request made but no response (server not running)
      return Promise.reject(error);
    } else {
      // Something else went wrong
      return Promise.reject(error);
    }
  }
);

// Auth APIs
export const register = (username, password) => 
  api.post('/register', { username, password });

export const login = (username, password) => 
  api.post('/login', { username, password });

export const logout = (token) => 
  api.post('/logout', { token });

export const verifyToken = (token) => 
  api.post('/verify_token', { token });

// Status API
export const getStatus = () => api.get('/status');
export const startRound = () => api.post('/start_round');
export const setXflStrategy = (strategy, param) => 
  api.post('/xfl/set_strategy', { strategy, param });
export const saveConfig = (config) => 
  api.post('/config/save', config);
export const getConfig = () => api.get('/config');
export const exportData = () => api.get('/export');

// Metrics APIs
export const getAccuracyData = () => api.get('/accuracy');
export const getLossData = () => api.get('/loss');
export const getClientsData = () => api.get('/clients');
export const getBandwidthData = (params) => api.get('/bandwidth', { params });
export const getLatencyData = (params) => api.get('/latency', { params });
export const getEnergyData = (params) => api.get('/energy', { params });
export const getNetworkMetricsData = (params) => api.get('/network_metrics', { params });
export const getRoundsHistory = () => api.get('/rounds_history');
export const getHistoryByStrategy = () => api.get('/history_by_strategy');

// DSE APIs
export const runDseSweep = (sweepConfig) => api.post('/dse/sweep', sweepConfig, { timeout: 600000 });
export const getDseStatus = (sessionId) => api.get(`/dse/status/${sessionId}`);
export const getDseProgress = (sessionId) => api.get(`/dse/progress/${sessionId}`);
export const getDseResults = (sessionId) => api.get(`/dse/results/${sessionId}`);
export const getDseAllResults = () => api.get('/dse/all_results');
export const getDseSessions = () => api.get('/dse/sessions');

export default api;

