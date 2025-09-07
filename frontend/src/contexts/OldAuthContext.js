import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import api from '../services/api';

const AuthContext = createContext();

// Helper to apply auth headers
const applyAuthHeader = () => {
  const token = localStorage.getItem('access_token');
  const type = (localStorage.getItem('token_type') || 'Bearer').replace(/^bearer$/i, 'Bearer');
  
  if (token) {
    // Ensure the token is in the format 'Bearer <token>'
    const authHeader = token.startsWith('Bearer ') ? token : `${type} ${token}`;
    api.defaults.headers.common['Authorization'] = authHeader;
    console.log('Auth header set:', { hasToken: !!token, header: authHeader.substring(0, 20) + '...' });
  } else {
    delete api.defaults.headers.common['Authorization'];
    console.log('Auth header cleared');
  }
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  const setUserData = useCallback((userData) => {
    const isAuth = !!userData;
    setUser(userData);
    setIsAuthenticated(isAuth);
    console.log('Auth state updated:', { isAuthenticated: isAuth, user: userData });
  }, []);

  const fetchUser = useCallback(async () => {
    const token = localStorage.getItem('access_token');
    console.log('Fetching user data...', { hasToken: !!token });
    
    if (!token) {
      console.log('No access token found, user is not authenticated');
      setUserData(null);
      return null;
    }
    
    try {
      // Explicitly set the auth header for this request
      const response = await api.get('/auth/me', {
        headers: {
          'Authorization': `${localStorage.getItem('token_type') || 'Bearer'} ${token}`.trim()
        },
        // Skip the request interceptor to avoid any potential issues
        skipAuth: true
      });
      
      if (!response.data) {
        throw new Error('No user data received');
      }
      
      console.log('User data received:', response.data);
      setUserData(response.data);
      return response.data;
      
    } catch (error) {
      console.error('Error fetching user data:', {
        status: error.response?.status,
        statusText: error.response?.statusText,
        message: error.message,
        data: error.response?.data
      });
      
      // Clear auth state on 401 or if the token is invalid
      if (error.response?.status === 401 || error.message?.includes('401')) {
        console.log('Authentication failed, clearing tokens');
        localStorage.removeItem('access_token');
        localStorage.removeItem('token_type');
      }
      
      setUserData(null);
      throw error;
    }
  }, []);

  useEffect(() => {
    let isMounted = true;
    
    const loadUser = async () => {
      try {
        setLoading(true);
        await fetchUser();
      } catch (error) {
        console.log('Initial user load failed, user is not authenticated');
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };
    
    applyAuthHeader();
    loadUser();
    
    return () => {
      isMounted = false;
    };
  }, [fetchUser]);

  const login = async (email, password) => {
    // Clear any existing tokens first
    localStorage.removeItem('access_token');
    localStorage.removeItem('token_type');
    
    console.log('Attempting to log in with email:', email);
    
    try {
      // Use URLSearchParams for form data
      const formData = new URLSearchParams();
      formData.append('username', email);
      formData.append('password', password);
      formData.append('grant_type', 'password');
      
      // Make login request using the API service
      console.log('Sending login request...');
      const response = await api.post('/auth/login', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'Accept': 'application/json'
        },
        skipAuth: true // Skip auth header for login request
      });
      
      console.log('Login response received:', { 
        hasToken: !!response.data?.access_token,
        tokenType: response.data?.token_type 
      });
      
      const { access_token, token_type } = response.data;
      
      if (!access_token) {
        throw new Error('No access token received from server');
      }
      
      // Store tokens
      const cleanTokenType = (token_type || 'Bearer').replace(/^bearer$/i, 'Bearer');
      localStorage.setItem('access_token', access_token);
      localStorage.setItem('token_type', cleanTokenType);
      
      // Update axios headers with the new token
      applyAuthHeader();
      
      // Verify the token was set correctly
      const tokenAfterSet = localStorage.getItem('access_token');
      console.log('Token after set:', tokenAfterSet ? 'present' : 'missing');
      
      if (!tokenAfterSet) {
        throw new Error('Failed to store access token');
      }
      
      // Fetch user data with the new token
      console.log('Fetching user profile...');
      const userData = await fetchUser();
      
      if (!userData) {
        throw new Error('No user data received');
      }
      
      console.log('Login successful, user data:', userData);
      return { ...response.data, user: userData };
      
    } catch (error) {
      console.error('Login error:', {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
        stack: error.stack
      });
      
      // Clear any partial auth state on error
      localStorage.removeItem('access_token');
      localStorage.removeItem('token_type');
      
      // Provide a more user-friendly error message based on the error type
      let errorMessage = 'Login failed. Please try again.';
      
      if (error.response) {
        // Handle HTTP error responses
        const { status, data } = error.response;
        
        if (status === 401) {
          errorMessage = data?.detail || 'Invalid email or password. Please try again.';
        } else if (status === 400) {
          errorMessage = data?.detail || 'Invalid request. Please check your input.';
        } else if (status === 429) {
          errorMessage = 'Too many login attempts. Please try again later.';
        } else if (status >= 500) {
          errorMessage = 'Server error. Please try again later.';
        } else if (data?.detail) {
          errorMessage = data.detail;
        } else if (data?.error_description) {
          errorMessage = data.error_description;
        }

  return (
    <AuthContext.Provider value={{ isAuthenticated, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export default AuthContext;
