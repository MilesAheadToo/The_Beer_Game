import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

// Set up axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Mixed Game API
export const mixedGameApi = {
  // Create a new mixed game
  createGame: async (gameData) => {
    try {
      const response = await api.post('/mixed-games/', gameData);
      return response.data;
    } catch (error) {
      console.error('Error creating mixed game:', error);
      throw error;
    }
  },

  // Get all mixed games
  getGames: async (status) => {
    try {
      const params = status ? { status } : {};
      const response = await api.get('/mixed-games/', { params });
      return response.data;
    } catch (error) {
      console.error('Error fetching mixed games:', error);
      throw error;
    }
  },

  // Get a single mixed game by ID
  getGame: async (gameId) => {
    try {
      const response = await api.get(`/mixed-games/${gameId}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching mixed game ${gameId}:`, error);
      throw error;
    }
  },

  // Update a mixed game
  updateGame: async (gameId, gameData) => {
    try {
      const response = await api.put(`/mixed-games/${gameId}`, gameData);
      return response.data;
    } catch (error) {
      console.error(`Error updating mixed game ${gameId}:`, error);
      throw error;
    }
  },

  // Start a mixed game
  startGame: async (gameId) => {
    try {
      const response = await api.post(`/mixed-games/${gameId}/start`);
      return response.data;
    } catch (error) {
      console.error(`Error starting mixed game ${gameId}:`, error);
      throw error;
    }
  },

  // Stop a mixed game
  stopGame: async (gameId) => {
    try {
      const response = await api.post(`/mixed-games/${gameId}/stop`);
      return response.data;
    } catch (error) {
      console.error(`Error stopping mixed game ${gameId}:`, error);
      throw error;
    }
  },

  // Advance to next round
  nextRound: async (gameId) => {
    try {
      const response = await api.post(`/mixed-games/${gameId}/next-round`);
      return response.data;
    } catch (error) {
      console.error(`Error advancing to next round in game ${gameId}:`, error);
      throw error;
    }
  },

  // Get game state
  getGameState: async (gameId) => {
    try {
      const response = await api.get(`/mixed-games/${gameId}/state`);
      return response.data;
    } catch (error) {
      console.error(`Error getting state for game ${gameId}:`, error);
      throw error;
    }
  }
};

// Original Game API
export const gameApi = {
  // Get all games
  getGames: async () => {
    try {
      const response = await api.get('/mixed-games/');
      return response.data;
    } catch (error) {
      console.error('Error fetching games:', error);
      throw error;
    }
  },

  // Get a single game by ID
  getGame: async (gameId) => {
    try {
      const response = await api.get(`/games/${gameId}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching game ${gameId}:`, error);
      throw error;
    }
  },

  // Create a new game
  createGame: async (gameData) => {
    try {
      const response = await api.post('/games/', gameData);
      return response.data;
    } catch (error) {
      console.error('Error creating game:', error);
      throw error;
    }
  },

  // Update a game
  updateGame: async (gameId, gameData) => {
    try {
      const response = await api.put(`/games/${gameId}`, gameData);
      return response.data;
    } catch (error) {
      console.error(`Error updating game ${gameId}:`, error);
      throw error;
    }
  },

  // Delete a game
  deleteGame: async (gameId) => {
    try {
      await api.delete(`/games/${gameId}`);
    } catch (error) {
      console.error(`Error deleting game ${gameId}:`, error);
      throw error;
    }
  },

  // Start a game
  startGame: async (gameId) => {
    try {
      const response = await api.post(`/games/${gameId}/start`);
      return response.data;
    } catch (error) {
      console.error(`Error starting game ${gameId}:`, error);
      throw error;
    }
  },
};

export default api;
