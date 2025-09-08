import axios from "axios";
import { toast } from "react-toastify";
import { API_BASE_URL } from "../config";

// Create axios instance with default config
const http = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true, // Required for cookies
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
  timeout: 30000, // 30 seconds
});

/**
 * Get CSRF token from cookies
 */
function getCSRFToken() {
  // Try to get from meta tag first (Django's default)
  const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
  if (csrfToken) return csrfToken;
  
  // Fallback to cookie
  const value = `; ${document.cookie}`;
  const parts = value.split('; csrftoken=');
  if (parts.length === 2) return parts.pop().split(';').shift();
  
  return null;
}

// Request interceptor for CSRF and auth headers
http.interceptors.request.use(
  (config) => {
    // Skip for external URLs
    if (!config.url.startsWith(API_BASE_URL)) return config;

    // Add CSRF token for state-changing requests
    if (['post', 'put', 'patch', 'delete'].includes(config.method?.toLowerCase())) {
      const csrfToken = getCSRFToken();
      if (csrfToken) {
        config.headers['X-CSRFToken'] = csrfToken;
      }
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
http.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    
    // Handle 401 Unauthorized
    if (error.response?.status === 401 && !originalRequest._retry) {
      // If we already tried to refresh, redirect to login
      if (originalRequest.url.includes('/auth/refresh')) {
        // Clear auth state
        document.cookie = 'access_token=; Path=/; Expires=Thu, 01 Jan 1970 00:00:01 GMT;';
        localStorage.removeItem('access_token');
        
        // Redirect to login if not already there
        if (!window.location.pathname.startsWith('/login')) {
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
      
      // Try to refresh the token
      try {
        originalRequest._retry = true;
        await http.post('/auth/refresh');
        // Retry the original request
        return http(originalRequest);
      } catch (refreshError) {
        // Refresh failed, clear auth and redirect
        document.cookie = 'access_token=; Path=/; Expires=Thu, 01 Jan 1970 00:00:01 GMT;';
        localStorage.removeItem('access_token');
        
        if (!window.location.pathname.startsWith('/login')) {
          window.location.href = '/login';
        }
        return Promise.reject(refreshError);
      }
    }
    
    // Handle CSRF token errors
    if (error.response?.status === 403 && 
        (error.response.data?.detail === 'CSRF token validation failed' || 
         error.response.data?.detail?.includes('CSRF'))) {
      // Try to get a new CSRF token
      try {
        await http.get('/auth/csrf-token');
        // Retry the original request
        return http(originalRequest);
      } catch (csrfError) {
        console.error('CSRF token refresh failed:', csrfError);
        return Promise.reject(csrfError);
      }
    }
    
    // Show error toast for server errors
    if (error.response?.status >= 500) {
      toast.error('A server error occurred. Please try again later.');
    } else if (error.response?.data?.detail) {
      // Show validation or other API errors
      toast.error(error.response.data.detail);
    }
    
    return Promise.reject(error);
  }
);

/**
 * Auth API service
 */
export const authApi = {
  /**
   * Login with username/email and password
   * @param {Object} credentials - Login credentials
   * @param {string} credentials.username - Username or email
   * @param {string} credentials.password - Password
   * @param {boolean} [rememberMe=false] - Whether to remember the user
   * @returns {Promise<Object>} User data
   */
  login: async ({ username, password, rememberMe = false }) => {
    const response = await http.post('/auth/login', {
      username,
      password,
      remember_me: rememberMe,
    });
    return response.data;
  },

  /**
   * Get current user data
   * @returns {Promise<Object>} User data
   */
  me: async () => {
    const response = await http.get('/auth/me');
    return response.data;
  },

  /**
   * Logout the current user
   * @returns {Promise<void>}
   */
  logout: async () => {
    try {
      await http.post('/auth/logout');
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear all auth tokens
      document.cookie = 'access_token=; Path=/; Expires=Thu, 01 Jan 1970 00:00:01 GMT;';
      document.cookie = 'token_type=; Path=/; Expires=Thu, 01 Jan 1970 00:00:01 GMT;';
      localStorage.removeItem('access_token');
      localStorage.removeItem('token_type');
    }
  },

  /**
   * Request a password reset email
   * @param {string} email - User's email address
   * @returns {Promise<void>}
   */
  requestPasswordReset: async (email) => {
    await http.post('/auth/forgot-password', { email });
  },

  /**
   * Reset password with a token
   * @param {string} token - Password reset token
   * @param {string} newPassword - New password
   * @returns {Promise<void>}
   */
  resetPassword: async (token, newPassword) => {
    await http.post('/auth/reset-password', {
      token,
      new_password: newPassword,
    });
  },

  /**
   * Refresh the access token
   * @returns {Promise<void>}
   */
  refreshToken: async () => {
    const response = await http.post('/auth/refresh');
    return response.data;
  },
};

/**
 * Game API service
 */
export const gameApi = {
  // Game management
  createGame: (payload) => http.post("/games", payload),
  listGames: (params) => http.get("/games", { params }),
  getGame: (id) => http.get(`/games/${id}`),
  updateGame: (id, data) => http.patch(`/games/${id}`, data),
  deleteGame: (id) => http.delete(`/games/${id}`),

  // Game actions
  startGame: (id) => http.post(`/games/${id}/start`),
  joinGame: (id, role) => http.post(`/games/${id}/join`, { role }),
  leaveGame: (id) => http.post(`/games/${id}/leave`),

  // Gameplay actions
  submitOrder: (gameId, order) => http.post(`/games/${gameId}/orders`, order),
  getOrders: (gameId) => http.get(`/games/${gameId}/orders`),
  getHistory: (gameId) => http.get(`/games/${gameId}/history`),

  // Chat
  getChatMessages: (gameId) => http.get(`/games/${gameId}/chat`),
  sendChatMessage: (gameId, message) =>
    http.post(`/games/${gameId}/chat`, { message }),
};

/**
 * User API service
 */
export const userApi = {
  getProfile: () => http.get("/users/me"),
  updateProfile: (data) => http.patch("/users/me", data),
  changePassword: (currentPassword, newPassword) =>
    http.post("/users/me/change-password", { currentPassword, newPassword }),

  // MFA setup
  setupMFA: () => http.post("/users/me/mfa/setup"),
  verifyMFA: (code) => http.post("/users/me/mfa/verify", { code }),
  disableMFA: () => http.post("/users/me/mfa/disable"),
  getRecoveryCodes: () => http.get("/users/me/mfa/recovery-codes"),
  generateRecoveryCodes: () => http.post("/users/me/mfa/recovery-codes"),
};

// For backward compatibility
export const mixedGameApi = {
  ...gameApi,
  health: async () => (await http.get("/health")).data,
  
  // Add login method that matches the expected signature
  login: async ({ username, password, grant_type = "password" }) => {
    const body = new URLSearchParams({ username, password, grant_type });
    const { data } = await http.post("/auth/login", body, {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });
    return data;
  },
  
  // Add me endpoint
  me: async () => (await http.get("/auth/me")).data,
  
  // Add logout endpoint
  logout: async () => { 
    try {
      await http.post("/auth/logout");
    } catch (error) {
      // Even if logout fails, clear local storage
      console.error("Logout error:", error);
    }
    localStorage.removeItem("access_token");
    localStorage.removeItem("token_type");
  },
};

export default {
  auth: authApi,
  game: gameApi,
  user: userApi,
  http, // Export the axios instance for direct use if needed
};
