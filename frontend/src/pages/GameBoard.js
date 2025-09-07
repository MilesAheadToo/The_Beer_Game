import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { WebSocketTester } from '../components/WebSocketTester';
import PageLayout from '../components/PageLayout';
import { 
  Box, 
  Button, 
  VStack, 
  HStack, 
  Text, 
  Heading, 
  Divider, 
  Badge, 
  useToast,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  CardFooter,
  Progress,
  Stat,
  Input,
  StatLabel,
  StatNumber,
  StatHelpText,
  StatArrow,
  StatGroup,
  IconButton,
  Tooltip,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalFooter,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  FormControl,
  FormLabel,
  useColorModeValue,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Alert,
  AlertIcon,
  Spinner,
  Center,
  Wrap,
  WrapItem,
  Avatar,
  Container,
} from '@chakra-ui/react';
import { 
  CheckCircleIcon, 
  TimeIcon, 
  RepeatIcon, 
  InfoIcon, 
  ArrowForwardIcon,
  ArrowBackIcon,
  ChatIcon,
  SettingsIcon,
  ViewIcon,
  StarIcon,
  WarningIcon,
} from '@chakra-ui/icons';
import { useWebSocket } from '../contexts/WebSocketContext';
import api from '../services/api';

const GameBoard = () => {
  // Theme values - must be called unconditionally at the top level
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  
  const { gameId } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const [isLoading, setIsLoading] = useState(true);
  const [orderQuantity, setOrderQuantity] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [gameDetails, setGameDetails] = useState(null);
  const [playerRole, setPlayerRole] = useState(null);
  const [isPlayerTurn, setIsPlayerTurn] = useState(false);
  
  const { 
    isConnected, 
    gameState, 
    players, 
    currentRound, 
    gameStatus, 
    send 
  } = useWebSocket();
  
  // Modal for game settings
  const { isOpen, onOpen, onClose } = useDisclosure();
  
  // Fetch game details on component mount
  useEffect(() => {
    const fetchGameDetails = async () => {
      try {
        setIsLoading(true);
        const game = await api.getGame(gameId);
        setGameDetails(game);
        
        // Determine player role (in a real app, this would come from auth context)
        const currentUserId = localStorage.getItem('user_id');
        const player = game.players.find(p => p.user_id === parseInt(currentUserId));
        if (player) {
          setPlayerRole(player.role);
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
  }, [gameId, navigate, toast]);
  
  // Check if it's the player's turn
  useEffect(() => {
    if (gameState && playerRole) {
      // In a real implementation, this would check the game state to see whose turn it is
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
  
  // Handle submitting an order
  const handleSubmitOrder = async () => {
    if (orderQuantity < 0) {
      toast({
        title: 'Invalid Order',
        description: 'Order quantity cannot be negative.',
        status: 'warning',
        duration: 3000,
        isClosable: true,
      });
      return;
    }
    
    try {
      setIsSubmitting(true);
      
      // In a real implementation, this would send the order via WebSocket
      await api.request(`/games/${gameId}/orders`, {
        method: 'POST',
        body: {
          round_number: currentRound,
          quantity: orderQuantity,
          role: playerRole,
        }
      });
      
      toast({
        title: 'Order Submitted',
        description: `You've ordered ${orderQuantity} units.`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      
      // Reset order quantity
      setOrderQuantity(0);
      
    } catch (error) {
      console.error('Error submitting order:', error);
      toast({
        title: 'Error',
        description: error.response?.data?.detail || 'Failed to submit order. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsSubmitting(false);
    }
  };
  
  // Handle starting the next round
  const handleNextRound = async () => {
    try {
      setIsSubmitting(true);
      await api.nextRound(gameId);
      toast({
        title: 'Round Advanced',
        description: 'The game has advanced to the next round.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error('Error advancing round:', error);
      toast({
        title: 'Error',
        description: error.response?.data?.detail || 'Failed to advance round. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsSubmitting(false);
    }
  };
  
  // Render player card
  const renderPlayerCard = (player) => {
    const isCurrentPlayer = player.role === playerRole;
    const isAI = player.is_ai;
    const inventory = player.inventory || 0;
    const backorders = player.backorders || 0;
    const incomingShipments = player.incoming_shipments || [];
    
    return (
      <Card 
        key={player.role} 
        borderWidth="1px"
        borderColor={isCurrentPlayer ? 'blue.200' : 'gray.200'}
        bg={isCurrentPlayer ? 'blue.50' : 'white'}
        boxShadow={isCurrentPlayer ? 'lg' : 'md'}
      >
        <CardHeader pb={2}>
          <HStack justify="space-between" align="center">
            <HStack>
              <Avatar 
                size="sm" 
                name={player.role}
                bg={isAI ? 'teal.500' : 'blue.500'}
                color="white"
              />
              <Box>
                <Text fontWeight="bold" textTransform="capitalize">
                  {player.role}
                  {isCurrentPlayer && (
                    <Badge ml={2} colorScheme="blue">You</Badge>
                  )}
                  {isAI && (
                    <Badge ml={1} colorScheme="teal">AI</Badge>
                  )}
                </Text>
                <Text fontSize="sm" color="gray.500">
                  {player.strategy ? `Strategy: ${player.strategy}` : 'Human Player'}
                </Text>
              </Box>
            </HStack>
            {player.is_turn && (
              <Badge colorScheme="green">Turn</Badge>
            )}
          </HStack>
        </CardHeader>
        <CardBody pt={0} pb={4}>
          <VStack align="stretch" spacing={3}>
            <Stat>
              <StatLabel>Inventory</StatLabel>
              <StatNumber>{inventory}</StatNumber>
              <StatHelpText>
                <StatArrow type={inventory > 10 ? 'increase' : 'decrease'} />
                {inventory > 10 ? 'Good' : 'Low'}
              </StatHelpText>
            </Stat>
            
            {backorders > 0 && (
              <Alert status="warning" size="sm" borderRadius="md">
                <AlertIcon />
                {backorders} backordered units
              </Alert>
            )}
            
            {incomingShipments.length > 0 && (
              <Box>
                <Text fontSize="sm" color="gray.600" mb={1}>
                  Incoming Shipments:
                </Text>
                <HStack spacing={2}>
                  {incomingShipments.map((qty, idx) => (
                    <Badge key={idx} colorScheme="blue">
                      {qty} in {idx + 1} round{idx > 0 ? 's' : ''}
                    </Badge>
                  ))}
                </HStack>
              </Box>
            )}
          </VStack>
        </CardBody>
        {isCurrentPlayer && isPlayerTurn && (
          <CardFooter pt={0} borderTopWidth="1px">
            <HStack width="100%">
              <NumberInput 
                size="sm" 
                min={0} 
                max={100}
                value={orderQuantity}
                onChange={(value) => setOrderQuantity(parseInt(value) || 0)}
                flex={1}
              >
                <NumberInputField placeholder="Qty" />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
              <Button 
                size="sm" 
                colorScheme="blue"
                onClick={handleSubmitOrder}
                isLoading={isSubmitting}
                loadingText="Submitting..."
                isDisabled={!orderQuantity || !isPlayerTurn}
                ml={2}
                textTransform="none"
                fontWeight="500"
                height="32px"
                px={4}
              >
                Place Order
              </Button>
            </HStack>
          </CardFooter>
        )}
      </Card>
    );
  };
  
  // Render game stats
  const renderGameStats = () => (
    <Card>
      <CardHeader pb={2}>
        <HStack justify="space-between" align="center">
          <Text fontWeight="bold">Game Stats</Text>
          <Badge colorScheme={gameStatus === 'in_progress' ? 'green' : 'gray'}>
            {gameStatus.replace('_', ' ').toUpperCase()}
          </Badge>
        </HStack>
      </CardHeader>
      <CardBody>
        <VStack align="stretch" spacing={4}>
          <Stat>
            <StatLabel>Current Round</StatLabel>
            <StatNumber>{currentRound}</StatNumber>
            <StatHelpText>of {gameDetails?.max_rounds || '?'} rounds</StatHelpText>
          </Stat>
          
          <Divider />
          
          <Stat>
            <StatLabel>Customer Demand</StatLabel>
            <StatNumber>{gameState?.current_demand || '?'}</StatNumber>
            <StatHelpText>units this round</StatHelpText>
          </Stat>
          
          <Divider />
          
          <Box>
            <Text fontWeight="medium" mb={2}>Demand Pattern</Text>
            <Text>{gameDetails?.demand_pattern?.type || 'Classic'}</Text>
            <Text fontSize="sm" color="gray.500">
              {gameDetails?.demand_pattern?.description || 'Standard step increase pattern'}
            </Text>
          </Box>
          
          {gameStatus === 'in_progress' && (
            <Button 
              leftIcon={<RepeatIcon />} 
              colorScheme="green"
              onClick={handleNextRound}
              isDisabled={!isPlayerTurn}
              textTransform="none"
              fontWeight="500"
              height="40px"
              px={4}
            >
              Next Round
            </Button>
          )}
        </VStack>
      </CardBody>
    </Card>
  );
  
  // Render game log
  const renderGameLog = () => (
    <Card height="100%">
      <CardHeader pb={2}>
        <Text fontWeight="bold">Game Log</Text>
      </CardHeader>
      <CardBody p={0}>
        <Box p={4} height="300px" overflowY="auto">
          {gameState?.game_log?.length > 0 ? (
            <VStack align="stretch" spacing={3}>
              {gameState.game_log.map((log, idx) => (
                <Box 
                  key={idx} 
                  p={3} 
                  bg="gray.50" 
                  borderRadius="md"
                  borderLeftWidth="3px"
                  borderLeftColor={log.type === 'round_start' ? 'green.500' : 'blue.500'}
                >
                  <Text fontSize="sm">
                    <strong>Round {log.round}:</strong> {log.message}
                  </Text>
                  <Text fontSize="xs" color="gray.500" mt={1}>
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </Text>
                </Box>
              ))}
            </VStack>
          ) : (
            <Center height="200px">
              <VStack>
                <InfoIcon fontSize="2xl" color="gray.400" />
                <Text color="gray.500">No game events yet</Text>
              </VStack>
            </Center>
          )}
        </Box>
      </CardBody>
    </Card>
  );
  
  // Render loading state
  if (isLoading) {
    const showWebSocketTester = process.env.NODE_ENV === 'development';

    return (
      <PageLayout title="Loading Game...">
        <VStack spacing={6} align="stretch" minH="60vh" justify="center">
          {showWebSocketTester && (
            <Box mb={6}>
              <WebSocketTester gameId={gameId} />
            </Box>
          )}
          <Center>
            <VStack spacing={4}>
              <Spinner size="xl" />
              <Text>Loading game data...</Text>
            </VStack>
          </Center>
        </VStack>
      </PageLayout>
    );
  }

  if (!gameDetails) {
    return (
      <PageLayout title="Game Not Found">
        <VStack spacing={6} align="stretch" minH="60vh" justify="center">
          <Alert status="error" borderRadius="md">
            <AlertIcon />
            Game not found or you don't have permission to view it.
          </Alert>
          <Button 
            mt={4} 
            onClick={() => navigate('/games')}
            alignSelf="flex-start"
            leftIcon={<ArrowBackIcon />}
            textTransform="none"
            fontWeight="500"
            variant="outline"
            colorScheme="blue"
          >
            Back to Games
          </Button>
        </VStack>
      </PageLayout>
    );
  }
  
  // Main game board
  return (
    <PageLayout title="Game Board">
      {/* Game Header */}
      <HStack justify="space-between" mb={6}>
        <VStack align="flex-start" spacing={1}>
          <Heading size="lg">{gameDetails.name}</Heading>
          <Text color="gray.600">{gameDetails.description || 'No description'}</Text>
        </VStack>
        <HStack>
          <Button 
            leftIcon={<SettingsIcon />} 
            variant="outline"
            onClick={onOpen}
            textTransform="none"
            fontWeight="500"
            colorScheme="blue"
          >
            Settings
          </Button>
          <Button 
            leftIcon={<ViewIcon />} 
            colorScheme="blue"
            onClick={() => navigate(`/games/${gameId}/analytics`)}
            textTransform="none"
            fontWeight="500"
          >
            View Analytics
          </Button>
        </HStack>
      </HStack>
      
      {/* Connection Status */}
      {!isConnected && (
        <Alert status="warning" mb={6} borderRadius="md">
          <AlertIcon />
          <Box>
            <Text fontWeight="bold">Connecting to game server...</Text>
            <Text fontSize="sm">Attempting to establish a real-time connection.</Text>
          </Box>
        </Alert>
      )}
      
      {/* Main Game Area */}
      <SimpleGrid columns={{ base: 1, lg: 3 }} gap={6} mb={6}>
        {/* Left Column - Game Stats */}
        <Box>
          {renderGameStats()}
        </Box>
        
        {/* Middle Column - Player Cards */}
        <Box>
          <VStack spacing={4}>
            {players
              .sort((a, b) => {
                // Sort players in supply chain order: Factory -> Distributor -> Wholesaler -> Retailer
                const order = { 'factory': 0, 'distributor': 1, 'wholesaler': 2, 'retailer': 3 };
                return order[a.role] - order[b.role];
              })
              .map(renderPlayerCard)}
          </VStack>
        </Box>
        
        {/* Right Column - Game Log */}
        <Box>
          {renderGameLog()}
        </Box>
      </SimpleGrid>
      
      {/* Game Controls */}
      <Card variant="outlined" mb={6}>
        <CardBody p={4}>
          <HStack justify="space-between">
            <Text fontWeight="medium">Game Controls</Text>
            <HStack spacing={2}>
              <Button 
                leftIcon={<RepeatIcon />}
                onClick={handleNextRound}
                isDisabled={!isPlayerTurn || gameStatus !== 'in_progress'}
                colorScheme="blue"
                textTransform="none"
                fontWeight="500"
                height="40px"
                px={4}
              >
                Next Round
              </Button>
              <Button 
                leftIcon={gameStatus === 'paused' ? <ArrowForwardIcon /> : <TimeIcon />}
                colorScheme={gameStatus === 'paused' ? 'green' : 'orange'}
                onClick={() => {
                  if (gameStatus === 'paused') {
                    api.request(`/games/${gameId}/resume`, { method: 'POST' });
                  } else {
                    api.request(`/games/${gameId}/pause`, { method: 'POST' });
                  }
                }}
                isDisabled={!isPlayerTurn}
                textTransform="none"
                fontWeight="500"
                height="40px"
                px={4}
              >
                {gameStatus === 'paused' ? 'Resume Game' : 'Pause Game'}
              </Button>
              <Button 
                leftIcon={<WarningIcon />}
                colorScheme="red"
                variant="outline"
                onClick={() => {
                  if (window.confirm('Are you sure you want to end the game? This cannot be undone.')) {
                    api.request(`/games/${gameId}/end`, { method: 'POST' });
                  }
                }}
                isDisabled={!isPlayerTurn}
                textTransform="none"
                fontWeight="500"
                height="40px"
                px={4}
              >
                End Game
              </Button>
            </HStack>
          </HStack>
        </CardBody>
      </Card>
      
      {/* Game Settings Modal */}
      <Modal isOpen={isOpen} onClose={onClose} size="xl">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Game Settings</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <Tabs isFitted variant="enclosed">
              <TabList mb="1em">
                <Tab>General</Tab>
                <Tab>Players</Tab>
                <Tab>Advanced</Tab>
              </TabList>
              <TabPanels>
                <TabPanel>
                  <VStack align="stretch" spacing={4}>
                    <FormControl>
                      <FormLabel>Game Name</FormLabel>
                      <Input value={gameDetails.name} isReadOnly />
                    </FormControl>
                    <FormControl>
                      <FormLabel>Status</FormLabel>
                      <Input value={gameStatus} isReadOnly />
                    </FormControl>
                    <FormControl>
                      <FormLabel>Current Round</FormLabel>
                      <Input value={`${currentRound} / ${gameDetails.max_rounds}`} isReadOnly />
                    </FormControl>
                  </VStack>
                </TabPanel>
                <TabPanel>
                  <VStack align="stretch" spacing={4}>
                    {players.map(player => (
                      <Box key={player.role} p={3} borderWidth="1px" borderRadius="md">
                        <HStack justify="space-between">
                          <VStack align="flex-start" spacing={0}>
                            <Text fontWeight="medium" textTransform="capitalize">
                              {player.role}
                              {player.role === playerRole && (
                                <Badge ml={2} colorScheme="blue">You</Badge>
                              )}
                            </Text>
                            <Text fontSize="sm" color="gray.500">
                              {player.is_ai ? `AI (${player.strategy})` : 'Human Player'}
                            </Text>
                          </VStack>
                          <Badge colorScheme={player.is_ready ? 'green' : 'gray'}> 
                            {player.is_ready ? 'Ready' : 'Not Ready'}
                          </Badge>
                        </HStack>
                      </Box>
                    ))}
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
                    
                    <FormControl>
                      <FormLabel>Demand Pattern</FormLabel>
                      <Input value={gameDetails.demand_pattern?.type || 'Classic'} isReadOnly />
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel>Lead Time</FormLabel>
                      <Input value={`${gameDetails.lead_time || 2} rounds`} isReadOnly />
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel>Holding Cost</FormLabel>
                      <Input value={`$${gameDetails.holding_cost || '0.50'} per unit per round`} isReadOnly />
                    </FormControl>
                    
                    <FormControl>
                      <FormLabel>Backorder Cost</FormLabel>
                      <Input value={`$${gameDetails.backorder_cost || '2.00'} per unit`} isReadOnly />
                    </FormControl>
                  </VStack>
                </TabPanel>
              </TabPanels>
            </Tabs>
          </ModalBody>
          <ModalFooter>
            <Button 
              colorScheme="blue" 
              onClick={onClose}
              textTransform="none"
              fontWeight="500"
              px={6}
            >
              Close
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </PageLayout>
  );
};

export default GameBoard;
