import React, { createContext, useState, useContext, useEffect } from 'react';
import axios from 'axios';

// Set base URL for all requests
axios.defaults.baseURL = 'http://localhost:8000/api/v1';

const AuthContext = createContext();

// Helper to apply auth headers
const applyAuthHeader = () => {
  const token = localStorage.getItem('access_token');
  const type = (localStorage.getItem('token_type') || 'Bearer').replace(/^bearer$/i, 'Bearer');
  if (token) axios.defaults.headers.common.Authorization = `${type} ${token}`;
  else delete axios.defaults.headers.common.Authorization;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    applyAuthHeader();
    (async () => {
      try {
        const res = await axios.get('/auth/me'); // with your baseURL configured
        setUser(res.data);
      } catch (e) {
        setUser(null);
        if (e?.response?.status === 401) {
          localStorage.removeItem('access_token');
          localStorage.removeItem('token_type');
          applyAuthHeader();
        }
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const login = async (email, password) => {
    const response = await axios.post(
      '/auth/login',
      new URLSearchParams({ username: email, password, grant_type: 'password' }),
      { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
    );
    const data = response.data;
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('token_type', (data.token_type || 'Bearer').replace(/^bearer$/i, 'Bearer'));
    applyAuthHeader();
    setUser(data.user ?? { email });
    return data;
  };

  const register = async (userData) => {
    const response = await axios.post('/auth/register', userData);
    return response.data;
  };

  const logout = async () => {
    try { 
      await axios.post('/auth/logout'); 
    } catch (e) {
      console.error('Logout error:', e);
    }
    setUser(null);
    localStorage.removeItem('access_token');
    localStorage.removeItem('token_type');
    applyAuthHeader();
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {!loading && children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  return useContext(AuthContext);
};

export default AuthContext;
