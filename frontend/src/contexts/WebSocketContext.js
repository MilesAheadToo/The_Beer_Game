import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { webSocketService } from '../services/websocket';
import { useToast } from '@chakra-ui/react';
import { useAuth } from './AuthContext';

const WebSocketContext = createContext(null);

export const WebSocketProvider = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [gameState, setGameState] = useState(null);
  const [players, setPlayers] = useState([]);
  const [currentRound, setCurrentRound] = useState(0);
  const [gameStatus, setGameStatus] = useState('idle');
  const toast = useToast();
  const callbacks = useRef(new Map());
  const params = useParams();
  const { accessToken } = useAuth();

  // Register a callback for specific message types
  const on = (eventType, callback) => {
    callbacks.current.set(eventType, callback);
    
    // Return cleanup function
    return () => {
      callbacks.current.delete(eventType);
    };
  };

  // Send a message through the WebSocket
  const send = (type, data = {}) => {
    return webSocketService.send({ type, ...data });
  };

  // Initialize WebSocket connection
  useEffect(() => {
    const gameId = params.gameId;
    if (!gameId || !accessToken) return;

    const handleWebSocketMessage = (event, data) => {
      switch (event) {
        case 'connected':
          setIsConnected(true);
          // Request the current game state when connected
          send('get_state');
          break;
          
        case 'disconnected':
          setIsConnected(false);
          break;
          
        case 'message':
          handleIncomingMessage(data);
          break;
          
        case 'error':
          console.error('WebSocket error:', data);
          toast({
            title: 'Connection Error',
            description: 'There was a problem with the game connection.',
            status: 'error',
            duration: 5000,
            isClosable: true,
          });
          break;
          
        case 'reconnect_failed':
          toast({
            title: 'Connection Lost',
            description: 'Unable to reconnect to the game server.',
            status: 'error',
            duration: null, // Don't auto-close
            isClosable: true,
          });
          break;
          
        default:
          console.log('Unhandled WebSocket event:', event, data);
          break;
      }
    };

    const connectWebSocket = async () => {
      try {
        // Connect to WebSocket with the current access token
        webSocketService.connect(gameId, accessToken);
        const unsubscribe = webSocketService.subscribe(handleWebSocketMessage);

        // Cleanup on unmount
        return () => {
          unsubscribe();
          webSocketService.disconnect();
        };
      } catch (error) {
        console.error('Failed to connect to WebSocket:', error);
        toast({
          title: 'Connection Error',
          description: 'Failed to connect to the game server. Please try again.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    };

    connectWebSocket();
    
    // Cleanup on unmount
    return () => {
      webSocketService.disconnect();
    };
  }, [params.gameId, accessToken, toast, handleIncomingMessage, send]);

  // Handle incoming WebSocket messages
  const handleIncomingMessage = useCallback((message) => {
    console.log('WebSocket message received:', message);
    
    // Update local state based on message type
    switch (message.type) {
      case 'game_state':
        setGameState(message.state);
        setPlayers(message.players || []);
        setCurrentRound(message.current_round || 0);
        setGameStatus(message.status || 'unknown');
        break;
        
      case 'round_started':
        setCurrentRound(message.round_number);
        toast({
          title: `Round ${message.round_number} Started`,
          description: message.message || 'A new round has begun!',
          status: 'info',
          duration: 3000,
          isClosable: true,
        });
        break;
        
      case 'round_ended':
        toast({
          title: `Round ${message.round_number} Ended`,
          description: message.message || 'The round has ended.',
          status: 'info',
          duration: 3000,
          isClosable: true,
        });
        break;
        
      case 'player_joined':
        toast({
          title: 'Player Joined',
          description: `${message.player_name} has joined the game.`,
          status: 'info',
          duration: 3000,
          isClosable: true,
        });
        break;
        
      case 'player_left':
        toast({
          title: 'Player Left',
          description: `${message.player_name} has left the game.`,
          status: 'warning',
          duration: 3000,
          isClosable: true,
        });
        break;
        
      case 'game_ended':
        setGameStatus('completed');
        toast({
          title: 'Game Over',
          description: message.message || 'The game has ended.',
          status: 'success',
          duration: null, // Don't auto-close
          isClosable: true,
        });
        break;
        
      default:
        // Check if there's a registered callback for this message type
        const callback = callbacks.current.get(message.type);
        if (callback) {
          callback(message);
        }
    }
  }, [toast]);

  // Expose the WebSocket API to child components
  const api = {
    isConnected,
    gameState,
    players,
    currentRound,
    gameStatus,
    send,
    on,
  };

  return (
    <WebSocketContext.Provider value={api}>
      {children}
    </WebSocketContext.Provider>
  );
};

// Custom hook to use the WebSocket context
export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

export default WebSocketContext;
