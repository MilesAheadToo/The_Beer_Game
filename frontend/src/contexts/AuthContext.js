import { createContext, useContext, useEffect, useState, useCallback, useRef } from "react";
import api from '../services/api';

const REFRESH_THRESHOLD = 5 * 60 * 1000; // 5 minutes before token expires

const AuthContext = createContext(null);
export const useAuth = () => useContext(AuthContext);

export function AuthProvider({ children }) {
  const [isAuthenticated, setIsAuthenticated] = useState(() => 
    !!localStorage.getItem("access_token")
  );
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState(null);
  const refreshTimeout = useRef();

  const login = useCallback(({ access_token, token_type = "Bearer", refresh_token, expires_in }) => {
    localStorage.setItem("access_token", access_token);
    localStorage.setItem("token_type", token_type.replace(/^bearer$/i, "Bearer"));
    if (refresh_token) localStorage.setItem("refresh_token", refresh_token);
    
    // Set token expiration time (default to 1 hour if not provided)
    const expiresAt = Date.now() + (expires_in || 3600) * 1000;
    localStorage.setItem("expires_at", expiresAt.toString());
    
    scheduleTokenRefresh(expiresAt);
    setIsAuthenticated(true);
    
    // Fetch user profile
    fetchUserProfile();
  }, []);

  const logout = useCallback(() => {
    clearTimeout(refreshTimeout.current);
    localStorage.removeItem("access_token");
    localStorage.removeItem("token_type");
    localStorage.removeItem("refresh_token");
    localStorage.removeItem("expires_at");
    setUser(null);
    setIsAuthenticated(false);
  }, []);

  const fetchUserProfile = async () => {
    try {
      const userData = await api.get("/users/me");
      setUser(userData);
    } catch (error) {
      console.error('Failed to fetch user profile:', error);
      // Don't log out on profile fetch failure
    }
  };

  const refreshToken = useCallback(async () => {
    const refreshToken = localStorage.getItem("refresh_token");
    if (!refreshToken) {
      logout();
      return;
    }

    try {
      const tokens = await api.request("/auth/refresh", {
        method: 'POST',
        body: {
          refresh_token: refreshToken,
          grant_type: "refresh_token"
        }
      });

      login({
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token || refreshToken,
        expires_in: tokens.expires_in
      });
    } catch (error) {
      console.error('Failed to refresh token:', error);
      logout();
    }
  }, [login, logout]);

  const scheduleTokenRefresh = (expiresAt) => {
    const now = Date.now();
    const expiresIn = expiresAt - now;
    
    // Clear any existing timeout
    clearTimeout(refreshTimeout.current);
    
    if (expiresIn <= 0) {
      logout();
      return;
    }
    
    // Schedule refresh 5 minutes before expiration
    const refreshIn = Math.max(expiresIn - REFRESH_THRESHOLD, 1000);
    refreshTimeout.current = setTimeout(refreshToken, refreshIn);
  };

  // Check auth state on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem("access_token");
      const expiresAt = localStorage.getItem("expires_at");
      
      if (!token || !expiresAt) {
        setLoading(false);
        return;
      }
      
      const expiresIn = parseInt(expiresAt) - Date.now();
      
      if (expiresIn <= 0) {
        // Token expired, try to refresh
        try {
          await refreshToken();
        } catch (error) {
          logout();
        }
      } else {
        // Schedule refresh
        scheduleTokenRefresh(parseInt(expiresAt));
        // Fetch user profile
        await fetchUserProfile();
      }
      
      setLoading(false);
    };
    
    checkAuth();
    
    // Cleanup on unmount
    return () => clearTimeout(refreshTimeout.current);
  }, [refreshToken, logout]);

  // Keep state in sync if another tab logs in/out
  useEffect(() => {
    const onStorage = (e) => {
      if (e.key === "access_token") {
        const isAuthenticated = !!localStorage.getItem("access_token");
        setIsAuthenticated(isAuthenticated);
        if (isAuthenticated) {
          fetchUserProfile();
        } else {
          setUser(null);
        }
      }
    };
    
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  return (
    <AuthContext.Provider value={{
      isAuthenticated,
      loading,
      user,
      login,
      logout,
      refreshToken
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export default AuthContext;
