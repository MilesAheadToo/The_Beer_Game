import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import PageLayout from '../components/PageLayout';
import RoundTimer from '../components/RoundTimer';
import { 
  Box, 
  Button, 
  VStack, 
  HStack, 
  Text, 
  Badge, 
  useToast,
  useColorModeValue,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Alert,
  AlertIcon,
  Spinner,
  Center,
  FormControl,
  FormLabel,
  Input,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
  Select as ChakraSelect
} from '@chakra-ui/react';
import { useWebSocket } from '../contexts/WebSocketContext';
import { useAuth } from '../contexts/AuthContext';
import mixedGameApi from '../services/api';

const GameBoard = () => {
  // Theme values
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  const { gameId } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const { user } = useAuth();
  const [gameState, setGameState] = useState(null);
  const [gameDetails, setGameDetails] = useState(null);
  const [playerRole, setPlayerRole] = useState('');
  const [playerId, setPlayerId] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isPlayerTurn, setIsPlayerTurn] = useState(false);
  const [myGames, setMyGames] = useState([]);
  
  const { isOpen, onClose } = useDisclosure();
  const { gameStatus } = useWebSocket();
  
  // Update game state when game ID changes
  useEffect(() => {
    const fetchGameState = async () => {
      if (gameId) {
        try {
          const state = await mixedGameApi.getGameState(gameId);
          setGameState(state);
          
          // Update derived state
          if (state.current_round) {
            setIsPlayerTurn(state.current_round.current_player_id === state.player_id);
          }
        } catch (error) {
          console.error('Error fetching game state:', error);
          toast({
            title: 'Error',
            description: 'Failed to load game state',
            status: 'error',
            duration: 5000,
            isClosable: true,
          });
        }
      }
    };
    
    fetchGameState();
  }, [gameId, toast]);

  // Load list of games created by this admin to allow quick switch
  useEffect(() => {
    (async () => {
      try {
        const games = await mixedGameApi.getGames();
        const mine = (games || []).filter(g => g.created_by === user?.id);
        setMyGames(mine);
      } catch (e) {
        // ignore
      }
    })();
  }, [user?.id]);
  
  // Fetch game details on component mount
  useEffect(() => {
    const fetchGameDetails = async () => {
      try {
        setIsLoading(true);
        const game = await mixedGameApi.getGame(gameId);
        setGameDetails(game);
        
        // Get current user ID and player info from auth context
        const currentUserId = user?.id;
        const player = game.players.find(p => p.user_id === currentUserId);
        if (player) {
          setPlayerRole(player.role);
          setPlayerId(player.id);
          
          // Check if it's the player's turn
          if (game.current_player_turn === player.role) {
            setIsPlayerTurn(true);
          }
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching game details:', error);
        toast({
          title: 'Error',
          description: 'Failed to load game details. Please try again.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
        navigate('/games');
      }
    };
    
    fetchGameDetails();
  }, [gameId, navigate, toast, user?.id]);
  
  // Check if it's the player's turn
  useEffect(() => {
    if (gameState && playerRole) {
      const currentPlayerTurn = gameState.current_player_turn === playerRole;
      setIsPlayerTurn(currentPlayerTurn);
      
      if (currentPlayerTurn) {
        toast({
          title: 'Your Turn!',
          description: 'It\'s your turn to place an order.',
          status: 'info',
          duration: 5000,
          isClosable: true,
        });
      }
    }
  }, [gameState, playerRole, toast]);
  
  // Handle order submission
  const handleOrderSubmit = async (quantity) => {
    try {
      await mixedGameApi.submitOrder(gameId, playerId, quantity);
      toast({
        title: 'Order submitted!',
        description: `Order of ${quantity} units has been placed.`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error submitting order:', error);
      toast({
        title: 'Error',
        description: 'Failed to submit order. Please try again.',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
    }
  };

  // Render game board
  return (
    <PageLayout title={`Game: ${gameDetails?.name || 'Loading...'}`}>
      <Box p={4}>
        {isLoading ? (
          <Center h="200px">
            <Spinner size="xl" />
          </Center>
        ) : (
          <VStack spacing={6} align="stretch">
            {/* Game status bar with round timer */}
              <HStack spacing={4} align="stretch">
                {/* Game info card */}
                <Box flex={1} p={4} bg={cardBg} borderRadius="lg" borderWidth="1px" borderColor={borderColor}>
                  <VStack align="start" spacing={4}>
                    <VStack align="start" spacing={0}>
                      <Text fontSize="sm" color="gray.500">Game</Text>
                      <Text fontSize="lg" fontWeight="bold">{gameDetails?.name || 'Untitled Game'}</Text>
                    </VStack>
                    {myGames.length > 0 && (
                      <ChakraSelect
                        placeholder="Switch to another of my games"
                        size="sm"
                        value={String(gameId)}
                        onChange={(e) => navigate(`/games/${e.target.value}`)}
                        maxW="sm"
                      >
                        {myGames.map((g) => (
                          <option key={g.id} value={g.id}>{g.name}</option>
                        ))}
                      </ChakraSelect>
                    )}
                  
                  <HStack spacing={6}>
                    <VStack align="start" spacing={0}>
                      <Text fontSize="sm" color="gray.500">Round</Text>
                      <Text fontSize="xl" fontWeight="bold">{gameState?.current_round || 1}</Text>
                    </VStack>
                    
                    <VStack align="start" spacing={0}>
                      <Text fontSize="sm" color="gray.500">Status</Text>
                      <Badge colorScheme={gameStatus === 'in_progress' ? 'green' : 'yellow'} px={2} py={1}>
                        {gameStatus === 'in_progress' ? 'In Progress' : 'Waiting'}
                      </Badge>
                    </VStack>
                    
                    <VStack align="start" spacing={0}>
                      <Text fontSize="sm" color="gray.500">Your Role</Text>
                      <Badge colorScheme="blue" px={2} py={1} textTransform="capitalize">
                        {playerRole || 'Unknown'}
                      </Badge>
                    </VStack>
                  </HStack>
                </VStack>
              </Box>
              
              {/* Round timer component */}
              {gameStatus === 'in_progress' && playerId && (
                <Box width="400px">
                  <RoundTimer 
                    gameId={gameId}
                    playerId={playerId}
                    roundNumber={gameState?.current_round || 1}
                    onOrderSubmit={handleOrderSubmit}
                    isPlayerTurn={isPlayerTurn}
                  />
                </Box>
              )}
            </HStack>
            
            {/* Rest of the game board content */}
            <Box p={4} bg={cardBg} borderRadius="lg" borderWidth="1px" borderColor={borderColor}>
              <Text>Game content goes here...</Text>
            </Box>
            
            {/* Game settings modal */}
            <Modal isOpen={isOpen} onClose={onClose} size="xl">
              <ModalOverlay />
              <ModalContent>
                <ModalHeader>Game Settings</ModalHeader>
                <ModalCloseButton />
                <ModalBody>
                  <Tabs>
                    <TabList>
                      <Tab>General</Tab>
                      <Tab>Advanced</Tab>
                    </TabList>
                    <TabPanels>
                      <TabPanel>
                        <VStack align="stretch" spacing={4}>
                          <FormControl>
                            <FormLabel>Game Name</FormLabel>
                            <Input value={gameDetails?.name || ''} isReadOnly />
                          </FormControl>
                        </VStack>
                      </TabPanel>
                      <TabPanel>
                        <VStack align="stretch" spacing={4}>
                          <Alert status="warning" borderRadius="md">
                            <AlertIcon />
                            <Box>
                              <Text fontWeight="bold">Advanced Settings</Text>
                              <Text fontSize="sm">These settings can affect game balance and performance.</Text>
                            </Box>
                          </Alert>
                        </VStack>
                      </TabPanel>
                    </TabPanels>
                  </Tabs>
                </ModalBody>
                <ModalFooter>
                  <Button colorScheme="blue" mr={3} onClick={onClose}>
                    Close
                  </Button>
                </ModalFooter>
              </ModalContent>
            </Modal>
          </VStack>
        )}
      </Box>
    </PageLayout>
  );
};

export default GameBoard;
