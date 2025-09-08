// /frontend/src/services/api.js
import axios from "axios";
import { API_BASE_URL } from "../config/api";

// Create a single axios instance for the app
const http = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true, // Required for cookies
  timeout: 20000,
  headers: {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
  },
});

// Backward-compatible axios export for modules that import `{ api }`
export const api = http;

// Request interceptor: handle CSRF token and auth headers
http.interceptors.request.use(async (config) => {
  // Skip for token refresh and CSRF endpoints to avoid infinite loops
  const isAuthRequest = ['/auth/login', '/auth/refresh-token', '/auth/csrf-token'].some(path => 
    config.url?.includes(path)
  );
  
  if (!isAuthRequest) {
    // Get CSRF token from cookie or fetch a new one
    const csrfToken = getCookie('csrftoken') || await fetchCsrfToken();
    if (csrfToken) {
      config.headers['X-CSRF-Token'] = csrfToken;
    }
  }
  
  return config;
});

// Response interceptor: handle token refresh and auth errors
http.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    
    // Handle 401 Unauthorized
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        // Try to refresh the token
        await http.post('/auth/refresh-token');
        // Retry the original request with new token
        return http(originalRequest);
      } catch (refreshError) {
        // If refresh fails, clear auth state and redirect to login
        if (window.location.pathname !== '/login') {
          const returnTo = encodeURIComponent(window.location.pathname + window.location.search);
          window.location.href = `/login?redirect=${returnTo}`;
        }
        return Promise.reject(refreshError);
      }
    }
    
    // Handle CSRF token errors
    if (error.response?.status === 403 && error.response.data?.code === 'csrf_token_mismatch') {
      // Clear the invalid CSRF token
      document.cookie = 'csrftoken=; Path=/; Expires=Thu, 01 Jan 1970 00:00:01 GMT;';
      // Retry the request with a new CSRF token
      return http(originalRequest);
    }
    
    return Promise.reject(error);
  }
);

// Helper function to get cookie by name
function getCookie(name) {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
  return null;
}

// Fetch a new CSRF token
async function fetchCsrfToken() {
  try {
    const response = await http.get('/auth/csrf-token');
    return response.data.csrf_token;
  } catch (error) {
    console.error('Failed to fetch CSRF token:', error);
    return null;
  }
}

// ----- High-level API wrappers -----
export const mixedGameApi = {
  async health() {
    const { data } = await http.get("/health");
    return data;
  },

  // Mixed Games management
  async createGame(gameData) {
    const { data } = await http.post('/mixed-games/', gameData);
    return data;
  },
  async getGames() {
    const { data } = await http.get('/mixed-games/');
    return data;
  },

  async startGame(gameId) {
    const { data } = await http.post(`/mixed-games/${gameId}/start`);
    return data;
  },

  async stopGame(gameId) {
    const { data } = await http.post(`/mixed-games/${gameId}/stop`);
    return data;
  },

  async nextRound(gameId) {
    const { data } = await http.post(`/mixed-games/${gameId}/next-round`);
    return data;
  },

  async getGameState(gameId) {
    const { data } = await http.get(`/mixed-games/${gameId}/state`);
    return data;
  },

  // Classic game endpoints (state, details, orders)
  async getGame(gameId) {
    const { data } = await http.get(`/games/${gameId}`);
    return data;
  },

  async submitOrder(gameId, playerId, quantity) {
    const { data } = await http.post(`/games/${gameId}/players/${playerId}/orders`, { quantity });
    return data;
  },

  async getRoundStatus(gameId) {
    const { data } = await http.get(`/games/${gameId}/rounds/current/status`);
    return data;
  },

  // Authentication endpoints
  async login(credentials) {
    const form = new URLSearchParams();
    form.set('username', credentials.username);
    form.set('password', credentials.password);
    form.set('grant_type', 'password');

    const { data } = await http.post('/auth/login', form, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });

    // Handle successful login
    if (data?.access_token) {
      // Store tokens in httpOnly cookies (handled by the browser)
      // The backend should set the appropriate cookies
      return { success: true, user: data.user };
    }
    
    return { success: false, error: data?.detail || 'Login failed' };
  },

  async logout() {
    try {
      await http.post('/auth/logout');
      return { success: true };
    } catch (error) {
      console.error('Logout error:', error);
      return { success: false, error: 'Failed to log out' };
    }
  },

  async getCurrentUser() {
    try {
      const { data } = await http.get('/auth/me');
      return data;
    } catch (error) {
      console.error('Failed to fetch current user:', error);
      throw error;
    }
  },

  async refreshToken() {
    try {
      const { data } = await http.post('/auth/refresh-token');
      return data;
    } catch (error) {
      console.error('Failed to refresh token:', error);
      throw error;
    }
  },

  async requestPasswordReset(email) {
    try {
      const { data } = await http.post('/auth/forgot-password', { email });
      return { success: true, data };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to request password reset',
      };
    }
  },

  async resetPassword(token, newPassword) {
    try {
      const { data } = await http.post('/auth/reset-password', {
        token,
        new_password: newPassword,
      });
      return { success: true, data };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to reset password',
      };
    }
  },

  async changePassword(currentPassword, newPassword) {
    try {
      const { data } = await http.post('/auth/change-password', {
        current_password: currentPassword,
        new_password: newPassword,
      });
      return { success: true, data };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to change password',
      };
    }
  },

  // MFA endpoints
  async setupMFA() {
    const { data } = await http.post('/auth/mfa/setup');
    return data;
  },

  async verifyMFA({ code, secret }) {
    const { data } = await http.post('/auth/mfa/verify', { code, secret });
    return data;
  },

  async disableMFA() {
    const { data } = await http.post('/auth/mfa/disable');
    return data;
  },

  // User management endpoints
  async register(userData) {
    try {
      const { data } = await http.post('/auth/register', userData);
      return { success: true, data };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Registration failed',
      };
    }
  },

  async updateProfile(userData) {
    try {
      const { data } = await http.patch('/auth/me', userData);
      return { success: true, data };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data?.detail || 'Failed to update profile',
      };
    }
  },

  // ...add other application-specific endpoints below
};

export default mixedGameApi;
