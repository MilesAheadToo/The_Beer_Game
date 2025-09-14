// /frontend/src/contexts/AuthContext.js
import { createContext, useContext, useEffect, useMemo, useState, useCallback, useRef } from 'react';
import { mixedGameApi } from '../services/api';
import { toast } from 'react-toastify';

// Session timeout in milliseconds (30 minutes)
const SESSION_TIMEOUT = 30 * 60 * 1000;
// Warning time before logout (5 minutes before timeout)
const WARNING_TIME = 5 * 60 * 1000;

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showTimeoutWarning, setShowTimeoutWarning] = useState(false);
  const [timeLeft, setTimeLeft] = useState(0);
  
  const logoutTimer = useRef(null);
  const warningTimer = useRef(null);
  const activityEvents = useMemo(() => ['mousedown', 'keydown', 'scroll', 'touchstart'], []);

  const logout = useCallback(async () => {
    try {
      setLoading(true);
      await mixedGameApi.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      if (logoutTimer.current) clearTimeout(logoutTimer.current);
      if (warningTimer.current) clearTimeout(warningTimer.current);
      setUser(null);
      setIsAuthenticated(false);
      setShowTimeoutWarning(false);
      setLoading(false);
      localStorage.removeItem('authState');
    }
  }, []);

  // Handle user activity - reset timers
  const resetTimers = useCallback(() => {
    if (logoutTimer.current) clearTimeout(logoutTimer.current);
    if (warningTimer.current) clearTimeout(warningTimer.current);
    setShowTimeoutWarning(false);

    if (isAuthenticated) {
      // Set warning timer (5 minutes before logout)
      warningTimer.current = setTimeout(() => {
        setShowTimeoutWarning(true);
        const warningDuration = WARNING_TIME / 1000 / 60; // Convert to minutes
        toast.warning(`Your session will expire in ${warningDuration} minutes due to inactivity.`, {
          autoClose: 10000,
          closeOnClick: false,
          pauseOnHover: true,
          draggable: true,
        });
      }, SESSION_TIMEOUT - WARNING_TIME);

      // Set logout timer
      logoutTimer.current = setTimeout(() => {
        logout();
        toast.info('Your session has expired. Please log in again.');
      }, SESSION_TIMEOUT);

      // Update time left counter
      const updateTimeLeft = () => {
        if (logoutTimer.current) {
          const timeLeft = Math.ceil((logoutTimer.current._idleStart + SESSION_TIMEOUT - Date.now()) / 1000 / 60);
          setTimeLeft(timeLeft > 0 ? timeLeft : 0);
        }
      };
      
      const interval = setInterval(updateTimeLeft, 60000); // Update every minute
      updateTimeLeft(); // Initial update
      
      return () => clearInterval(interval);
    }
  }, [isAuthenticated, logout]);

  // Set up activity listeners
  useEffect(() => {
    if (isAuthenticated) {
      // Add event listeners for user activity
      const handleActivity = () => {
        resetTimers();
      };
      
      activityEvents.forEach(event => {
        window.addEventListener(event, handleActivity);
      });
      
      // Initialize timers
      resetTimers();
      
      // Clean up
      return () => {
        activityEvents.forEach(event => {
          window.removeEventListener(event, handleActivity);
        });
        if (logoutTimer.current) clearTimeout(logoutTimer.current);
        if (warningTimer.current) clearTimeout(warningTimer.current);
      };
    }
  }, [isAuthenticated, resetTimers, activityEvents]);

  // Check if user is authenticated on initial load and handle token refresh
  useEffect(() => {
    const checkAuth = async () => {
      try {
        // Avoid running auth check on the login route to prevent noisy 401/refresh loops
        if (typeof window !== 'undefined' && window.location.pathname.startsWith('/login')) {
          setLoading(false);
          return;
        }

        setLoading(true);
        const userData = await mixedGameApi.getCurrentUser();
        setUser(userData);
        setIsAuthenticated(true);

        // Refresh token periodically (every 15 minutes)
        const refreshInterval = setInterval(async () => {
          try {
            await mixedGameApi.refreshToken();
          } catch (error) {
            console.error('Token refresh failed:', error);
            // If refresh fails, log the user out
            logout();
          }
        }, 15 * 60 * 1000); // 15 minutes

        return () => clearInterval(refreshInterval);
      } catch (err) {
        setIsAuthenticated(false);
        setUser(null);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, [logout]);

  const login = useCallback(async (credentials) => {
    try {
      setLoading(true);
      setError(null);
      
      // This will automatically handle CSRF token and cookies
      const result = await mixedGameApi.login(credentials);
      
      // API returns shape { success, user } on success
      if (result?.success) {
        const nextUser = result.user;
        setUser(nextUser);
        setIsAuthenticated(true);
        
        // Reset timers after successful login
        resetTimers();
        return { success: true, user: nextUser };
      }
      
      const message = result?.error || 'Login failed. Please check your credentials.';
      setError(message);
      return { success: false, error: message };
    } catch (error) {
      const message = error.response?.data?.detail || 'Login failed. Please check your credentials.';
      setError(message);
      return { success: false, error: message };
    } finally {
      setLoading(false);
    }
  }, [resetTimers]);

  

  const refreshUser = useCallback(async () => {
    try {
      const userData = await mixedGameApi.getCurrentUser();
      setUser(userData);
      return userData;
    } catch (error) {
      setIsAuthenticated(false);
      setUser(null);
      throw error;
    }
  }, []);

  // ----- Role helpers -----
  const hasRole = useCallback((role) => {
    if (!user) return false;
    const roles = Array.isArray(user.roles) ? user.roles : [];
    return Boolean(user.is_superuser) || roles.includes(role);
  }, [user]);

  const hasAnyRole = useCallback((roles = []) => {
    if (!roles || roles.length === 0) return true;
    return roles.some((r) => hasRole(r));
  }, [hasRole]);

  const hasAllRoles = useCallback((roles = []) => {
    if (!roles || roles.length === 0) return true;
    return roles.every((r) => hasRole(r));
  }, [hasRole]);

  const isAdmin = useMemo(() => {
    if (!user) return false;
    const roles = Array.isArray(user.roles) ? user.roles : [];
    return Boolean(user.is_superuser) || roles.includes('admin');
  }, [user]);

  const value = useMemo(() => ({
    isAuthenticated,
    user,
    loading,
    error,
    login,
    logout,
    refreshUser,
    // role helpers
    hasRole,
    hasAnyRole,
    hasAllRoles,
    isAdmin,
    showTimeoutWarning,
    timeLeft,
    resetTimers, // Export resetTimers to allow manual reset from components
  }), [isAuthenticated, user, loading, error, login, logout, refreshUser, hasRole, hasAnyRole, hasAllRoles, isAdmin, showTimeoutWarning, timeLeft, resetTimers]);

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthContext;
