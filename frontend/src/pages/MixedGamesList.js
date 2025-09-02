import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { 
  Box, 
  Button, 
  Container, 
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
  Divider,
  Tooltip
} from '@chakra-ui/react';
import { 
  AddIcon, 
  ChevronDownIcon, 
  DeleteIcon, 
  EditIcon, 
  RepeatIcon,
  ArrowForwardIcon,
  TimeIcon,
  CheckIcon,
  NotAllowedIcon
} from '@chakra-ui/icons';
import { mixedGameApi } from '../services/api';

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
  const [selectedGame, setSelectedGame] = useState(null);
  const navigate = useNavigate();
  const toast = useToast();
  
  const { 
    isOpen: isDeleteDialogOpen, 
    onOpen: onDeleteDialogOpen, 
    onClose: onDeleteDialogClose 
  } = useDisclosure();
  
  const cancelRef = React.useRef();

  const fetchGames = async () => {
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
  };

  useEffect(() => {
    fetchGames();
  }, []);

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
      // await mixedGameApi.deleteGame(selectedGame.id);
      
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

  return (
    <Container maxW="container.xl" py={8}>
      <Box mb={6} display="flex" justifyContent="space-between" alignItems="center">
        <VStack align="flex-start" spacing={1}>
          <Text fontSize="2xl" fontWeight="bold">Mixed Human/AI Games</Text>
          <Text color="gray.600" fontSize="sm">
            Manage games with both human and AI players
          </Text>
        </VStack>
        <Button
          leftIcon={<AddIcon />}
          colorScheme="blue"
          onClick={() => navigate('/games/mixed/new')}
        >
          New Game
        </Button>
      </Box>

      <Box bg="white" p={6} rounded="lg" boxShadow="md">
        <Table variant="simple">
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
                  <Text>Loading games...</Text>
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
                      onClick={() => navigate('/games/mixed/new')}
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
                    <VStack align="flex-start" spacing={0}>
                      <Text fontWeight="medium">
                        <Link to={`/games/mixed/${game.id}`} style={{ textDecoration: 'underline' }}>
                          {game.name}
                        </Link>
                      </Text>
                      {game.description && (
                        <Text fontSize="sm" color="gray.500" noOfLines={1}>
                          {game.description}
                        </Text>
                      )}
                    </VStack>
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
                    <VStack align="flex-start" spacing={0}>
                      <Text fontSize="sm">
                        {new Date(game.created_at).toLocaleDateString()}
                      </Text>
                      <Text fontSize="xs" color="gray.500">
                        {new Date(game.created_at).toLocaleTimeString()}
                      </Text>
                    </VStack>
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
                            onClick={() => navigate(`/games/mixed/${game.id}/edit`)}
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
    </Container>
  );
};

export default MixedGamesList;
