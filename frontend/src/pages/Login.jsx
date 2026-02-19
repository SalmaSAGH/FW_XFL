import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { login } from '../services/api';

function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');

    if (!username || !password) {
      setError('Please enter username and password');
      return;
    }

    setLoading(true);

    try {
      const response = await login(username, password);
      console.log('Login response:', response.data);
      
      if (response.data.status === 'success') {
        localStorage.setItem('isAuthenticated', 'true');
        localStorage.setItem('username', response.data.username);
        localStorage.setItem('token', response.data.token);
        navigate('/config');
      }
    } catch (err) {
      console.error('Login error:', err);
      
      if (err.response) {
        // Server responded with an error
        console.log('Error response:', err.response.data);
        if (err.response.data && err.response.data.error) {
          setError(err.response.data.error);
        } else if (err.response.data && err.response.data.message) {
          setError(err.response.data.message);
        } else {
          setError(`Login failed. Please try again. (Status: ${err.response.status})`);
        }
      } else if (err.request) {
        // Request was made but no response received
        setError('Unable to connect to server. Please check if the server is running.');
      } else {
        // Something else went wrong
        setError('Login failed. Please try again.');
      }
    }

    setLoading(false);
  };

  return (
    <div className="login-container">
      <div className="login-box">
        <div className="login-logo">
          <h1>🔷 XFL-RPiLab</h1>
          <p>Federated Learning Dashboard</p>
        </div>

        <form onSubmit={handleLogin}>
          <div className="form-group">
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username"
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password"
            />
          </div>

          {error && (
            <div style={{ color: '#ef5350', marginBottom: '15px', fontSize: '14px' }}>
              {error}
            </div>
          )}

          <button 
            type="submit" 
            className="btn btn-primary" 
            style={{ width: '100%' }}
            disabled={loading}
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>

        <div style={{ marginTop: '20px', textAlign: 'center', color: '#888', fontSize: '14px' }}>
          <p>Don't have an account? <a href="#" onClick={() => navigate('/register')} style={{ color: '#4dabf7' }}>Register here</a></p>
        </div>
      </div>
    </div>
  );
}

export default Login;
