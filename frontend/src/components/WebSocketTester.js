import React, { useEffect, useState } from 'react';
import { Button, Box, Text, VStack, useToast } from '@chakra-ui/react';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useAuth } from '../contexts/AuthContext';

export const WebSocketTester = ({ gameId }) => {
  const { isConnected, send, on } = useWebSocket();
  const { accessToken } = useAuth();
  const [messages, setMessages] = useState([]);
  const toast = useToast();

  useEffect(() => {
    // Subscribe to WebSocket messages
    const unsubscribe = on('message', (message) => {
      setMessages(prev => [...prev.slice(-9), message]); // Keep last 10 messages
      console.log('WebSocket message:', message);
    });

    return () => {
      if (unsubscribe) unsubscribe();
    };
  }, [on]);

  const handlePing = () => {
    try {
      send('ping', { timestamp: Date.now() });
      toast({
        title: 'Ping sent',
        status: 'success',
        duration: 2000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Failed to send ping:', error);
      toast({
        title: 'Failed to send ping',
        description: error.message,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  return (
    <Box p={4} borderWidth={1} borderRadius="md" bg="white" boxShadow="sm">
      <VStack spacing={4} align="stretch">
        <Text fontSize="lg" fontWeight="bold">WebSocket Connection Test</Text>
        
        <Box>
          <Text>Status: <Text as="span" color={isConnected ? 'green.500' : 'red.500'} fontWeight="bold">
            {isConnected ? 'Connected' : 'Disconnected'}
          </Text></Text>
          <Text>Game ID: {gameId || 'Not set'}</Text>
          <Text>Access Token: {accessToken ? 'Present' : 'Missing'}</Text>
        </Box>

        <Button 
          colorScheme="blue" 
          onClick={handlePing}
          isDisabled={!isConnected}
        >
          Send Ping
        </Button>

        <Box mt={4}>
          <Text fontWeight="bold">Last Messages:</Text>
          <Box mt={2} p={2} bg="gray.50" borderRadius="md" minH="100px" maxH="200px" overflowY="auto">
            {messages.length === 0 ? (
              <Text color="gray.500">No messages received yet</Text>
            ) : (
              messages.map((msg, i) => (
                <Text key={i} fontSize="sm" fontFamily="mono" whiteSpace="pre-wrap">
                  {JSON.stringify(msg, null, 2)}
                </Text>
              ))
            )}
          </Box>
        </Box>
      </VStack>
    </Box>
  );
};

export default WebSocketTester;
