import { api, mixedGameApi } from './api';

const shouldTryMixedGames = (error) => {
  const status = error?.response?.status;
  if (status && [401, 403, 404, 405, 409, 422, 500].includes(status)) {
    return true;
  }
  const message = error?.message || '';
  return message.includes('AsyncSession') || message.includes('query');
};

const gameApi = {
  // Get all available games
  async getGames() {
    try {
      const response = await api.get('/games/');
      return response.data;
    } catch (error) {
      if (shouldTryMixedGames(error)) {
        return mixedGameApi.getGames();
      }
      throw error;
    }
  },
  
  // Get a specific game by ID
  async getGame(gameId) {
    const response = await api.get(`/games/${gameId}/`);
    return response.data;
  },
  
  // Create a new game
  async createGame(gameData) {
    const response = await api.post('/games/', gameData);
    return response.data;
  },

  // Update an existing game
  async updateGame(gameId, gameData) {
    // Prefer modern mixed games endpoint for updates
    try {
      return await mixedGameApi.updateGame(gameId, gameData);
    } catch (error) {
      const response = await api.put(`/games/${gameId}`, gameData);
      return response.data;
    }
  },

  // Delete a game
  async deleteGame(gameId) {
    try {
      const response = await api.delete(`/games/${gameId}`);
      return response.data;
    } catch (error) {
      if (shouldTryMixedGames(error)) {
        const { data } = await api.delete(`/mixed-games/${gameId}`);
        return data;
      }
      throw error;
    }
  },
  
  // Join an existing game
  async joinGame(gameId) {
    const response = await api.post(`/games/${gameId}/join/`);
    return response.data;
  },
  
  // Leave a game
  async leaveGame(gameId) {
    const response = await api.post(`/games/${gameId}/leave/`);
    return response.data;
  },
  
  // Start a game (game master only)
  async startGame(gameId) {
    try {
      const response = await api.post(`/games/${gameId}/start/`);
      return response.data;
    } catch (error) {
      if (shouldTryMixedGames(error)) {
        return mixedGameApi.startGame(gameId);
      }
      throw error;
    }
  },
  
  // End a game (game master only)
  async endGame(gameId) {
    const response = await api.post(`/games/${gameId}/end/`);
    return response.data;
  },
  
  // Submit an order in a game
  async submitOrder(gameId, orderData) {
    const response = await api.post(`/games/${gameId}/orders/`, orderData);
    return response.data;
  },
  
  // Set player ready status
  async setPlayerReady(gameId, { is_ready }) {
    const response = await api.patch(`/games/${gameId}/player/`, { is_ready });
    return response.data;
  },
  
  // Get game history
  async getGameHistory() {
    const response = await api.get('/games/history/');
    return response.data;
  },
  
  // Get player statistics
  async getPlayerStats() {
    const response = await api.get('/games/player/stats/');
    return response.data;
  },
  
  // Send a chat message
  async sendChatMessage(gameId, message) {
    const response = await api.post(`/games/${gameId}/chat/`, { message });
    return response.data;
  },
  
  // Get game chat history
  async getChatHistory(gameId) {
    const response = await api.get(`/games/${gameId}/chat/`);
    return response.data;
  },
  
  // Update game settings (game master only)
  async updateGameSettings(gameId, settings) {
    const response = await api.patch(`/games/${gameId}/settings/`, settings);
    return response.data;
  },
  
  // Kick a player from the game (game master only)
  async kickPlayer(gameId, playerId) {
    const response = await api.post(`/games/${gameId}/kick/`, { player_id: playerId });
    return response.data;
  },
  
  // Transfer game ownership (game master only)
  async transferOwnership(gameId, newOwnerId) {
    const response = await api.post(`/games/${gameId}/transfer/`, { new_owner_id: newOwnerId });
    return response.data;
  },
  
  // Get game statistics
  async getGameStats(gameId) {
    const response = await api.get(`/games/${gameId}/stats/`);
    return response.data;
  },
  
  // Get player's current game (if any)
  async getCurrentGame() {
    const response = await api.get('/games/current/');
    return response.data;
  },
  
  // Reconnect to a game (after disconnection)
  async reconnectToGame(gameId) {
    const response = await api.post(`/games/${gameId}/reconnect/`);
    return response.data;
  }
};

export default gameApi;
