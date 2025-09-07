import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import authService from '../services/authService';

const AuthContext = createContext(null);
export const useAuth = () => useContext(AuthContext);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const isAuthenticated = !!user;

  const login = async (credentials) => {
    try {
      setLoading(true);
      setError(null);
      
      // Attempt login - authService will handle the cookie
      const userData = await authService.login(credentials);
      setUser(userData);
      
      toast.success('Successfully logged in');
      return userData;
    } catch (err) {
      console.error('Login error:', err);
      const errorMessage = err.response?.data?.detail || 'Login failed. Please check your credentials.';
      setError(errorMessage);
      toast.error(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const logout = async (options = {}) => {
    const { redirect = true, silent = false } = options;
    
    try {
      await authService.logout();
      if (!silent) {
        toast.success('You have been logged out successfully');
      }
    } catch (err) {
      console.error('Logout error:', err);
    } finally {
      // Clear user data
      setUser(null);
      setLoading(false);
      if (redirect && !window.location.pathname.startsWith('/login')) {
        navigate('/login', { replace: true });
      }
    }
  };

  // Check auth status on mount
  useEffect(() => {
    let isMounted = true;

    const checkAuth = async () => {
      try {
        const userData = await authService.getCurrentUser();
        if (isMounted) {
          setUser(userData);
          setError(null);
        }
      } catch (err) {
        if (isMounted) {
          console.error('Auth check failed:', err);
          await logout({ silent: true });
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    checkAuth();

    return () => {
      isMounted = false;
    };
  }, []);

  const value = useMemo(
    () => ({
      user,
      isAuthenticated,
      loading,
      error,
      login,
      logout,
      refreshUser: () => authService.getCurrentUser().then(setUser),
    }),
    [user, isAuthenticated, loading, error]
  );

  return (
    <AuthContext.Provider value={value}>
      {!loading ? children : (
        <div className="flex items-center justify-center min-h-screen">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
        </div>
      )}
    </AuthContext.Provider>
  );
}
