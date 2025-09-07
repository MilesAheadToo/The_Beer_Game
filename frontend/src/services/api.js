const BASE_URL = "http://localhost:8000/api/v1";

/**
 * Main API service for making authenticated requests to the backend
 */
const api = {
  /**
   * Authenticate a user and get access token
   * @param {Object} credentials - Login credentials
   * @param {string} credentials.username - User's email/username
   * @param {string} credentials.password - User's password
   * @param {string} [credentials.grant_type=password] - OAuth2 grant type
   * @returns {Promise<{access_token: string, token_type: string, refresh_token: string}>} Auth tokens
   */
  login: async ({ username, password, grant_type = "password" }) => {
    const body = new URLSearchParams({ username, password, grant_type });

    const res = await fetch(`${BASE_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `Login failed with ${res.status}`);
    }

    const data = await res.json();
    return {
      access_token: data.access_token,
      token_type: data.token_type,
      refresh_token: data.refresh_token,
    };
  },
  
  /**
   * Helper method for making authenticated requests
   * @private
   */
  request: async (endpoint, options = {}) => {
    try {
      const res = await fetch(`${BASE_URL}${endpoint}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        body: options.body && typeof options.body === 'object' 
          ? JSON.stringify(options.body) 
          : options.body
      });
      
      if (!res.ok) {
        const error = new Error(`Request failed with status ${res.status}`);
        error.status = res.status;
        try {
          error.data = await res.json();
        } catch {
          error.data = await res.text().catch(() => null);
        }
        
        // Handle 401 Unauthorized
        if (res.status === 401) {
          console.log('Authentication required');
          
          // Only redirect if we're not already on the login page
          if (!window.location.pathname.includes('/login')) {
            const redirect = encodeURIComponent(window.location.pathname + window.location.search);
            window.location.href = `/login?redirect=${redirect}`;
          }
        }
        
        throw error;
      }
      
      // For 204 No Content responses, return null
      if (res.status === 204) return null;
      
      return res.json();
    } catch (error) {
      // Handle network errors
      if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        console.error('Network Error:', error.message);
        throw new Error('Unable to connect to the server. Please check your internet connection.');
      }
      throw error;
    }
  },
  
  // Game API methods
  getGames: async (status) => {
    const query = status ? `?status=${status}` : '';
    return api.request(`/games${query}`);
  },
  
  getGame: async (gameId) => {
    return api.request(`/games/${gameId}`);
  },
  
  createGame: async (gameData) => {
    return api.request('/games', {
      method: 'POST',
      body: gameData
    });
  },
  
  updateGame: async (gameId, gameData) => {
    return api.request(`/games/${gameId}`, {
      method: 'PUT',
      body: gameData
    });
  },
  
  deleteGame: async (gameId) => {
    return api.request(`/games/${gameId}`, {
      method: 'DELETE'
    });
  },
  
  startGame: async (gameId) => {
    return api.request(`/games/${gameId}/start`, {
      method: 'POST'
    });
  },
  
  stopGame: async (gameId) => {
    return api.request(`/games/${gameId}/stop`, {
      method: 'POST'
    });
  },
  
  nextRound: async (gameId) => {
    return api.request(`/games/${gameId}/next-round`, {
      method: 'POST'
    });
  },
  
  getGameState(gameId) {
    return this.request(`/games/${gameId}/state`);
  },
  
  getGameResults(gameId) {
    return this.request(`/games/${gameId}/results`);
  },
  
  // Get the current round status including time remaining
  getRoundStatus(gameId) {
    return this.request(`/games/${gameId}/rounds/current/status`);
  },
  
  // Submit or update an order for the current round
  submitOrder(gameId, playerId, quantity) {
    return this.request(`/games/${gameId}/players/${playerId}/orders`, {
      method: 'POST',
      body: { quantity }
    });
  },
  
  // Get the current player's order for the round
  getPlayerOrder(gameId, playerId, roundNumber) {
    return this.request(`/games/${gameId}/players/${playerId}/rounds/${roundNumber}/order`);
  }
};

export default api;
