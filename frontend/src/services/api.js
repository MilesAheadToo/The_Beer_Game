import axios from 'axios';

const API_BASE_URL = '/api/v1';

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
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Game API
export const gameApi = {
  // Get all games
  getGames: async () => {
    try {
      const response = await api.get('/games/');
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
