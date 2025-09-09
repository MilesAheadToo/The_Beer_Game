import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { 
  Box, 
  Button, 
  Table, 
  Thead, 
  Tbody, 
  Tr, 
  Th, 
  Td, 
  Badge, 
  IconButton, 
  Menu, 
  MenuButton, 
  MenuList, 
  MenuItem,
  useDisclosure,
  AlertDialog,
  AlertDialogOverlay,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogBody,
  AlertDialogFooter,
  useToast,
  Text,
  HStack,
  VStack,
  Tooltip,
  useColorModeValue,
  Flex,
  Spinner,
  Heading,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  Icon
} from '@chakra-ui/react';
import PageLayout from '../components/PageLayout';
import { 
  AddIcon, 
  ChevronDownIcon, 
  DeleteIcon, 
  EditIcon, 
  RepeatIcon,
  ArrowForwardIcon,
  TimeIcon,
  CheckIcon,
  ViewIcon,
  WarningTwoIcon
} from '@chakra-ui/icons';
import mixedGameApi from '../services/api';
import { getModelStatus } from '../services/modelService';

const statusColors = {
  CREATED: 'blue',
  IN_PROGRESS: 'green',
  PAUSED: 'orange',
  COMPLETED: 'purple',
  CANCELLED: 'red'
};

const MixedGamesList = () => {
  const [games, setGames] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [modelStatus, setModelStatus] = useState(null);
  const [selectedGame, setSelectedGame] = useState(null);
  const navigate = useNavigate();
  const toast = useToast();
  
  const { 
    isOpen: isDeleteDialogOpen, 
    onOpen: onDeleteDialogOpen, 
    onClose: onDeleteDialogClose 
  } = useDisclosure();
  
  const cancelRef = React.useRef();

  const fetchGames = useCallback(async () => {
    try {
      setIsLoading(true);
      const data = await mixedGameApi.getGames();
      setGames(data);
    } catch (error) {
      console.error('Error fetching games:', error);
      toast({
        title: 'Error loading games',
        description: 'Failed to fetch games. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  }, [toast]);

  const fetchModelStatus = useCallback(async () => {
    try {
      setIsModelLoading(true);
      const status = await getModelStatus();
      setModelStatus(status);
    } catch (error) {
      console.error('Error fetching model status:', error);
      toast({
        title: 'Error',
        description: 'Failed to fetch GNN model status',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsModelLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    fetchGames();
    fetchModelStatus();
  }, [fetchGames, fetchModelStatus]);

  const handleStartGame = async (gameId) => {
    try {
      await mixedGameApi.startGame(gameId);
      toast({
        title: 'Game started',
        description: 'The game has been started successfully.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      fetchGames();
    } catch (error) {
      console.error('Error starting game:', error);
      toast({
        title: 'Error starting game',
        description: error.response?.data?.detail || 'Failed to start game',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleStopGame = async (gameId) => {
    try {
      await mixedGameApi.stopGame(gameId);
      toast({
        title: 'Game stopped',
        description: 'The game has been stopped.',
        status: 'info',
        duration: 3000,
        isClosable: true,
      });
      fetchGames();
    } catch (error) {
      console.error('Error stopping game:', error);
      toast({
        title: 'Error stopping game',
        description: error.response?.data?.detail || 'Failed to stop game',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleNextRound = async (gameId) => {
    try {
      await mixedGameApi.nextRound(gameId);
      toast({
        title: 'Round advanced',
        description: 'The game has advanced to the next round.',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      fetchGames();
    } catch (error) {
      console.error('Error advancing round:', error);
      toast({
        title: 'Error advancing round',
        description: error.response?.data?.detail || 'Failed to advance round',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const confirmDeleteGame = (game) => {
    setSelectedGame(game);
    onDeleteDialogOpen();
  };

  const handleDeleteGame = async () => {
    if (!selectedGame) return;
    
    try {
      // In a real implementation, you would call the delete endpoint
      // await api.deleteGame(selectedGame.id);
      
      toast({
        title: 'Game deleted',
        description: `The game "${selectedGame.name}" has been deleted.`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      
      fetchGames();
      onDeleteDialogClose();
    } catch (error) {
      console.error('Error deleting game:', error);
      toast({
        title: 'Error deleting game',
        description: error.response?.data?.detail || 'Failed to delete game',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const getStatusBadge = (status) => (
    <Badge colorScheme={statusColors[status] || 'gray'}>
      {status.replace('_', ' ')}
    </Badge>
  );

  const getActionButton = (game) => {
    switch (game.status) {
      case 'CREATED':
        return (
          <Button
            size="sm"
            colorScheme="green"
            leftIcon={<ArrowForwardIcon />}
            onClick={() => handleStartGame(game.id)}
          >
            Start Game
          </Button>
        );
      case 'IN_PROGRESS':
        return (
          <HStack spacing={2}>
            <Button
              size="sm"
              colorScheme="blue"
              leftIcon={<RepeatIcon />}
              onClick={() => handleNextRound(game.id)}
            >
              Next Round
            </Button>
            <Button
              size="sm"
              colorScheme="red"
              variant="outline"
              onClick={async ()=> { try { await mixedGameApi.finishGame(game.id); fetchGames(); } catch(e){} }}
            >
              Finish Game
            </Button>
            <Button
              size="sm"
              colorScheme="orange"
              leftIcon={<TimeIcon />}
              onClick={() => handleStopGame(game.id)}
              variant="outline"
            >
              Pause
            </Button>
          </HStack>
        );
      case 'PAUSED':
        return (
          <Button
            size="sm"
            colorScheme="green"
            leftIcon={<ArrowForwardIcon />}
            onClick={() => handleStartGame(game.id)}
          >
            Resume
          </Button>
        );
      case 'COMPLETED':
        return (
          <Button
            size="sm"
            leftIcon={<CheckIcon />}
            isDisabled
            variant="outline"
          >
            Completed
          </Button>
        );
      default:
        return null;
    }
  };

  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  return (
    <PageLayout title="Mixed Games">
      {/* Model Status Alert */}
      {!isModelLoading && modelStatus && (
        <Alert 
          status={modelStatus.is_trained ? 'success' : 'warning'} 
          variant="left-accent"
          mb={6}
          borderRadius="md"
          alignItems="center"
        >
          <Box mr={4} display="flex" alignItems="center">
            <Icon as={WarningTwoIcon} boxSize="2.5rem" color="red.500" />
          </Box>
          <Box flex="1">
            <AlertTitle>
              {modelStatus.is_trained ? 'GNN Model Trained' : 'GNN Model Not Trained'}
            </AlertTitle>
            <AlertDescription fontSize="sm">
              {modelStatus.is_trained 
                ? `Model was last trained on ${new Date(modelStatus.last_modified).toLocaleString()}`
                : 'The GNN model needs to be trained before AI agents can be used.'}
              {modelStatus.is_trained && (
                <Text fontSize="xs" mt={1}>
                  File: {modelStatus.file_size_mb?.toFixed(2)}MB
                  {modelStatus.epoch && ` | Epoch: ${modelStatus.epoch}`}
                  {modelStatus.training_loss && ` | Loss: ${modelStatus.training_loss.toFixed(4)}`}
                </Text>
              )}
            </AlertDescription>
          </Box>
        </Alert>
      )}
        
      <Flex justify="space-between" align="center" mb={6} mt={4}>
        <VStack align="flex-start" spacing={1} pt={2}>
          <Heading size="xl" fontWeight="600">Mixed Games</Heading>
          <Text color="gray.500" fontSize="md">
            Manage and join mixed human-AI games
          </Text>
        </VStack>
        <Button 
          as={Link} 
          to="/games/new" 
          colorScheme="blue"
          leftIcon={<AddIcon />}
          size="md"
          height="44px"
          px={6}
          textTransform="none"
          fontWeight="500"
          _hover={{
            transform: 'translateY(-1px)',
          }}
          _active={{
            transform: 'none'
          }}
        >
          Create New Game
        </Button>
      </Flex>

      <Box 
        bg={cardBg} 
        borderRadius="lg" 
        borderWidth="1px" 
        borderColor={borderColor}
        overflow="hidden"
        boxShadow="sm"
        className="table-surface"
      >
        {/* Slightly lighter, compact table font */}
        <Box fontSize="sm" overflowX="auto">
          <Table variant="simple" size="sm">
            <Thead>
              <Tr>
                <Th>Name</Th>
                <Th>Status</Th>
                <Th>Round</Th>
                <Th>Players</Th>
                <Th>Created</Th>
                <Th>Actions</Th>
              </Tr>
            </Thead>
            <Tbody>
            {isLoading ? (
              <Tr>
                <Td colSpan={6} textAlign="center" py={8}>
                  <Box textAlign="center" py={10}>
                    <Spinner size="xl" />
                    <Text mt={4} color="gray.600">Loading games...</Text>
                  </Box>
                </Td>
              </Tr>
            ) : games.length === 0 ? (
              <Tr>
                <Td colSpan={6} textAlign="center" py={8}>
                  <VStack spacing={4}>
                    <Text>No games found</Text>
                    <Button
                      leftIcon={<AddIcon />}
                      colorScheme="blue"
                      onClick={() => navigate('/games/new')}
                    >
                      Create a new game
                    </Button>
                  </VStack>
                </Td>
              </Tr>
            ) : (
              games.map((game) => (
                <Tr key={game.id} _hover={{ bg: 'gray.50' }}>
                  <Td>
                    <Text
                      fontSize="sm"
                      fontWeight="semibold"
                      isTruncated
                      maxW={{ base: '16rem', sm: '20rem', md: '28rem', lg: '36rem' }}
                      display="inline-block"
                    >
                      <Link to={`/games/new?name=${encodeURIComponent(game.name || '')}` +
                        `&description=${encodeURIComponent(game.description || '')}` +
                        (game.pricing_config ? `&pricing_config=${encodeURIComponent(JSON.stringify(game.pricing_config))}` : '') +
                        (game.system_config ? `&system_config=${encodeURIComponent(JSON.stringify(game.system_config))}` : '') +
                        (game.node_policies ? `&node_policies=${encodeURIComponent(JSON.stringify(game.node_policies))}` : '')
                      } style={{ textDecoration: 'underline' }}>
                        {game.name}
                      </Link>
                      {game.description && (
                        <Text as="span" ml={2} color="gray.500">({game.description})</Text>
                      )}
                    </Text>
                  </Td>
                  <Td>{getStatusBadge(game.status)}</Td>
                  <Td>
                    <HStack>
                      <Text>{game.current_round || 0}</Text>
                      <Text color="gray.500">/ {game.max_rounds}</Text>
                    </HStack>
                  </Td>
                  <Td>
                    <HStack spacing={2}>
                      {game.players && game.players.map((player) => (
                        <Tooltip 
                          key={player.role} 
                          label={`${player.role} (${player.is_ai ? 'AI' : 'Human'})`}
                          placement="top"
                        >
                          <Box
                            w={3}
                            h={3}
                            borderRadius="full"
                            bg={player.is_ai ? 'teal.500' : 'blue.500'}
                          />
                        </Tooltip>
                      ))}
                    </HStack>
                  </Td>
                  <Td>
                    <Text fontSize="sm" noOfLines={1} maxW={{ base: '12rem', sm: '16rem', md: '20rem' }}>
                      {new Date(game.created_at).toLocaleString()}
                    </Text>
                  </Td>
                  <Td>
                    <HStack spacing={2}>
                      {getActionButton(game)}
                      
                      <Menu>
                        <MenuButton
                          as={IconButton}
                          size="sm"
                          variant="ghost"
                          icon={<ChevronDownIcon />}
                        />
                        <MenuList>
                          <MenuItem 
                            icon={<EditIcon />}
                            onClick={() => navigate(`/games/${game.id}/edit`)}
                          >
                            Edit
                          </MenuItem>
                          <MenuItem 
                            icon={<DeleteIcon />} 
                            color="red.500"
                            onClick={() => confirmDeleteGame(game)}
                            isDisabled={game.status === 'IN_PROGRESS'}
                          >
                            Delete
                          </MenuItem>
                          <MenuItem 
                            as={Button}
                            variant="ghost"
                            leftIcon={<ViewIcon />}
                            justifyContent="flex-start"
                            onClick={() => navigate(`/dashboard?gameId=${game.id}`)}
                            w="full"
                            textAlign="left"
                            px={4}
                            py={2}
                            borderRadius={0}
                          >
                            View Results
                          </MenuItem>
                        </MenuList>
                      </Menu>
                    </HStack>
                  </Td>
                </Tr>
              ))
            )}
          </Tbody>
          </Table>
        </Box>
      </Box>

      {/* Delete Confirmation Dialog */}
      <AlertDialog
        isOpen={isDeleteDialogOpen}
        leastDestructiveRef={cancelRef}
        onClose={onDeleteDialogClose}
      >
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Delete Game
            </AlertDialogHeader>

            <AlertDialogBody>
              Are you sure you want to delete "{selectedGame?.name}"? This action cannot be undone.
            </AlertDialogBody>

            <AlertDialogFooter>
              <Button ref={cancelRef} onClick={onDeleteDialogClose}>
                Cancel
              </Button>
              <Button colorScheme="red" onClick={handleDeleteGame} ml={3}>
                Delete
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </PageLayout>
  );
};

export default MixedGamesList;
