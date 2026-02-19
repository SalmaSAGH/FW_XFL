import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { 
  getStatus, 
  getAccuracyData, 
  getLossData, 
  getClientsData, 
  getBandwidthData,
  getLatencyData,
  getEnergyData,
  getNetworkMetricsData,
  getRoundsHistory
} from '../services/api';

const DataContext = createContext(null);

// Cache duration in milliseconds (e.g., 30 seconds)
const CACHE_DURATION = 30000;

export function DataProvider({ children }) {
  const [cachedData, setCachedData] = useState({
    status: null,
    accuracy: { rounds: [], accuracy: [] },
    loss: { rounds: [], loss: [] },
    clients: [],
    bandwidth: { labels: [], values: [] },
    latency: { rounds: [], latency: [] },
    energy: { rounds: [], energy: [] },
    networkMetrics: { rounds: [], packet_loss: [], jitter: [] },
    history: []
  });
  
  const [loading, setLoading] = useState(true);
  const [lastFetchTime, setLastFetchTime] = useState(null);
  const [error, setError] = useState(null);

  const fetchAllData = useCallback(async (forceRefresh = false) => {
    // Check if we have valid cached data that's still fresh
    const now = Date.now();
    if (!forceRefresh && lastFetchTime && (now - lastFetchTime < CACHE_DURATION)) {
      return; // Cache is still valid, don't fetch
    }

    try {
      setError(null);
      
      // Fetch all data in parallel
      const [statusRes, accuracyRes, lossRes, clientsRes, bandwidthRes, latencyRes, energyRes, networkRes, historyRes] = await Promise.allSettled([
        getStatus().catch(() => ({ data: null })),
        getAccuracyData().catch(() => ({ data: null })),
        getLossData().catch(() => ({ data: null })),
        getClientsData().catch(() => ({ data: null })),
        getBandwidthData().catch(() => ({ data: null })),
        getLatencyData().catch(() => ({ data: null })),
        getEnergyData().catch(() => ({ data: null })),
        getNetworkMetricsData().catch(() => ({ data: null })),
        getRoundsHistory().catch(() => ({ data: null }))
      ]);

      // Update cache with new data
      setCachedData({
        status: statusRes.value?.data || null,
        accuracy: accuracyRes.value?.data || { rounds: [], accuracy: [] },
        loss: lossRes.value?.data || { rounds: [], loss: [] },
        clients: clientsRes.value?.data?.clients || [],
        bandwidth: bandwidthRes.value?.data || { labels: [], values: [] },
        latency: latencyRes.value?.data || { rounds: [], latency: [] },
        energy: energyRes.value?.data || { rounds: [], energy: [] },
        networkMetrics: networkRes.value?.data || { rounds: [], packet_loss: [], jitter: [] },
        history: historyRes.value?.data?.rounds || []
      });
      
      setLastFetchTime(now);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching data:', err);
      setError(err.message);
      setLoading(false);
    }
  }, [lastFetchTime]);

  // Initial fetch
  useEffect(() => {
    fetchAllData(true);
    
    // Set up polling interval to refresh data
    const interval = setInterval(() => fetchAllData(false), 2000);
    
    return () => clearInterval(interval);
  }, [fetchAllData]);

  // Provide data and refresh function to consumers
  const value = {
    data: cachedData,
    loading,
    error,
    refresh: () => fetchAllData(true)
  };

  return (
    <DataContext.Provider value={value}>
      {children}
    </DataContext.Provider>
  );
}

export function useData() {
  const context = useContext(DataContext);
  if (!context) {
    throw new Error('useData must be used within a DataProvider');
  }
  return context;
}

export default DataContext;
