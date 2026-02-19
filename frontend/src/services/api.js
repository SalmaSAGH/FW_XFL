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
export const getBandwidthData = () => api.get('/bandwidth');
export const getLatencyData = () => api.get('/latency');
export const getEnergyData = () => api.get('/energy');
export const getNetworkMetricsData = () => api.get('/network_metrics');
export const getRoundsHistory = () => api.get('/rounds_history');

export default api;
