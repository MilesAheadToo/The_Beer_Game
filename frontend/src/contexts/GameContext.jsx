import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { useAuth } from './AuthContext';
import { useWebSocket } from './WebSocketContext';
import gameApi from '../services/gameApi';

const GameContext = createContext({
  // Game state
  game: null,
  currentPlayer: null,
  isGameMaster: false,
  isGameActive: false,
  isLoading: true,
  
  // Game actions
  submitOrder: async () => {},
  startGame: async () => {},
  endGame: async () => {},
  leaveGame: async () => {},
  setPlayerReady: async () => {},
  sendChatMessage: async () => {},
  
  // UI state
  activeTab: 'game',
  setActiveTab: () => {},
  
  // Chat
  chatMessages: [],
  
  // Error handling
  error: null,
});

export const GameProvider = ({ children }) => {
  const { gameId } = useParams();
  const { user } = useAuth();
  const { isConnected, sendMessage, subscribe, connect } = useWebSocket();
  
  // Game state
  const [game, setGame] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('game');
  const [chatMessages, setChatMessages] = useState([]);
  
  // Derived state
  const currentPlayer = game?.players?.find(p => p.user_id === user?.id);
  const isGameMaster = game?.created_by === user?.id;
  const isGameActive = game?.status === 'in_progress';

  // Connect to WebSocket when game and player are loaded
  useEffect(() => {
    if (gameId && currentPlayer?.id && isConnected === false) {
      // Connect to WebSocket with game ID and player ID
      const connected = connect(gameId, currentPlayer.id);
      if (!connected) {
        console.error('Failed to connect to WebSocket');
      }
    }
  }, [gameId, currentPlayer?.id, isConnected, connect]);

  // Fetch game data
  const fetchGame = useCallback(async () => {
    if (!gameId) return;
    
    try {
      setIsLoading(true);
      const gameData = await gameApi.getGame(gameId);
      setGame(gameData);
      setError(null);
      
      // If we have chat messages, update the chat
      if (gameData.chat_messages) {
        setChatMessages(gameData.chat_messages);
      }
    } catch (err) {
      console.error('Failed to fetch game:', err);
      setError('Failed to load game. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, [gameId]);

  // Connect to WebSocket when component mounts or game changes
  useEffect(() => {
    if (gameId && user?.access_token) {
      // Connect to WebSocket for this game
      sendMessage('connect', { gameId });
      
      // Initial game data fetch
      fetchGame();
      
      // Set up polling for game updates (fallback)
      const pollInterval = setInterval(fetchGame, 30000); // Every 30 seconds
      
      return () => {
        clearInterval(pollInterval);
        // Don't disconnect WebSocket here as it might be used by other components
      };
    }
  }, [gameId, user?.access_token, fetchGame, sendMessage]);

  // Handle WebSocket messages
  useEffect(() => {
    if (!isConnected) return;
    
    const handleMessage = (event, data) => {
      switch (event) {
        case 'game_update':
          setGame(prevGame => ({
            ...prevGame,
            ...data.game,
            // Preserve players array if not provided in update
            players: data.game.players || prevGame?.players || []
          }));
          break;
          
        case 'chat_message':
          setChatMessages(prev => [...prev, data]);
          break;
          
        case 'player_joined':
        case 'player_left':
        case 'player_ready':
          fetchGame(); // Refresh game data
          break;
          
        case 'game_started':
          setGame(prev => ({ ...prev, status: 'in_progress' }));
          break;
          
        case 'game_ended':
          setGame(prev => ({ ...prev, status: 'completed' }));
          break;
          
        case 'error':
          console.error('Game error:', data);
          setError(data.message || 'An error occurred');
          break;
          
        default:
          console.warn('Unhandled WebSocket message:', event, data);
      }
    };
    
    // Subscribe to WebSocket events
    const unsubscribe = subscribe(handleMessage);
    return () => unsubscribe();
  }, [isConnected, fetchGame, subscribe]);

  // Game actions
  const submitOrder = async (amount) => {
    if (!gameId || !amount) return false;
    
    try {
      await gameApi.submitOrder(gameId, { amount: parseInt(amount, 10) });
      return true;
    } catch (error) {
      console.error('Failed to submit order:', error);
      setError(error.response?.data?.detail || 'Failed to submit order');
      return false;
    }
  };
  
  const startGame = async () => {
    if (!gameId) return false;
    
    try {
      await gameApi.startGame(gameId);
      return true;
    } catch (error) {
      console.error('Failed to start game:', error);
      setError(error.response?.data?.detail || 'Failed to start game');
      return false;
    }
  };
  
  const endGame = async () => {
    if (!gameId) return false;
    
    try {
      await gameApi.endGame(gameId);
      return true;
    } catch (error) {
      console.error('Failed to end game:', error);
      setError(error.response?.data?.detail || 'Failed to end game');
      return false;
    }
  };
  
  const leaveGame = async () => {
    if (!gameId) return false;
    
    try {
      await gameApi.leaveGame(gameId);
      return true;
    } catch (error) {
      console.error('Failed to leave game:', error);
      setError(error.response?.data?.detail || 'Failed to leave game');
      return false;
    }
  };
  
  const setPlayerReady = async (isReady) => {
    if (!gameId) return false;
    
    try {
      await gameApi.setPlayerReady(gameId, { is_ready: isReady });
      return true;
    } catch (error) {
      console.error('Failed to set player ready status:', error);
      setError('Failed to update ready status');
      return false;
    }
  };
  
  const sendChatMessage = async (message) => {
    if (!gameId || !message?.trim()) return false;
    
    try {
      if (isConnected) {
        sendMessage('chat_message', { 
          message: message.trim(),
          sender: user.username,
          timestamp: new Date().toISOString()
        });
      } else {
        // Fallback to HTTP if WebSocket is not available
        await gameApi.sendChatMessage(gameId, message.trim());
      }
      return true;
    } catch (error) {
      console.error('Failed to send chat message:', error);
      return false;
    }
  };

  // Context value
  const value = {
    // Game state
    game,
    currentPlayer,
    isGameMaster,
    isGameActive,
    isLoading,
    
    // Game actions
    submitOrder,
    startGame,
    endGame,
    leaveGame,
    setPlayerReady,
    sendChatMessage,
    
    // UI state
    activeTab,
    setActiveTab,
    
    // Chat
    chatMessages,
    
    // Error handling
    error,
  };

  return (
    <GameContext.Provider value={value}>
      {children}
    </GameContext.Provider>
  );
};

export const useGame = () => {
  const context = useContext(GameContext);
  if (context === undefined) {
    throw new Error('useGame must be used within a GameProvider');
  }
  return context;
};
