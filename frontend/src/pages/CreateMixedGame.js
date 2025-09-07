import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Button, 
  FormControl, 
  FormLabel, 
  Input, 
  Select, 
  VStack, 
  HStack, 
  Text, 
  Checkbox, 
  NumberInput, 
  NumberInputField, 
  NumberInputStepper, 
  NumberIncrementStepper, 
  NumberDecrementStepper,
  useToast,
  Card,
  CardBody,
  CardHeader,
  Heading,
  Divider,
  useColorModeValue,
  Switch,
  FormHelperText,
  Grid,
  GridItem,
  Badge
} from '@chakra-ui/react';
import PageLayout, { PageSection } from '../components/PageLayout';
import api from '../services/api';

const playerRoles = [
  { value: 'retailer', label: 'Retailer' },
  { value: 'wholesaler', label: 'Wholesaler' },
  { value: 'distributor', label: 'Distributor' },
  { value: 'factory', label: 'Factory' },
];

const agentStrategies = [
  {
    group: 'Basic Strategies',
    options: [
      { value: 'NAIVE', label: 'Naive' },
      { value: 'BULLWHIP', label: 'Bullwhip' },
      { value: 'CONSERVATIVE', label: 'Conservative' },
      { value: 'RANDOM', label: 'Random' },
    ]
  },
  {
    group: 'Advanced Strategies',
    options: [
      { value: 'DEMAND_DRIVEN', label: 'Demand Driven' },
      { value: 'COST_OPTIMIZATION', label: 'Cost Optimization' },
    ]
  },
  {
    group: 'AI-Powered (LLM)',
    options: [
      { value: 'LLM_CONSERVATIVE', label: 'LLM - Conservative' },
      { value: 'LLM_BALANCED', label: 'LLM - Balanced' },
      { value: 'LLM_AGGRESSIVE', label: 'LLM - Aggressive' },
      { value: 'LLM_ADAPTIVE', label: 'LLM - Adaptive' },
    ]
  }
];

const demandPatterns = [
  { value: 'classic', label: 'Classic (Step Increase)' },
  { value: 'random', label: 'Random' },
  { value: 'seasonal', label: 'Seasonal' },
  { value: 'constant', label: 'Constant' },
];

const CreateMixedGame = () => {
  const [gameName, setGameName] = useState('');
  const [maxRounds, setMaxRounds] = useState(20);
  const [description, setDescription] = useState('');
  const [isPublic, setIsPublic] = useState(true);
  const [demandPattern, setDemandPattern] = useState(demandPatterns[0].value);
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const [players, setPlayers] = useState(
    playerRoles.map(role => ({
      role: role.value,
      playerType: 'human',
      strategy: 'NAIVE',
      canSeeDemand: role.value === 'retailer',
      userId: role.value === 'retailer' ? 'current-user-id' : ''
    }))
  );
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const toast = useToast();

  const handlePlayerTypeChange = (index, type) => {
    setPlayers(players.map((player, i) => 
      i === index 
        ? { 
            ...player, 
            playerType: type,
            // Reset strategy to default when changing to human
            ...(type === 'human' && { strategy: agentStrategies[0].options[0].value })
          } 
        : player
    ));
  };

  const handleStrategyChange = (index, strategy) => {
    setPlayers(players.map((player, i) => 
      i === index ? { ...player, strategy } : player
    ));
  };

  const handleCanSeeDemandChange = (index, canSeeDemand) => {
    setPlayers(players.map((player, i) => 
      i === index ? { ...player, canSeeDemand } : player
    ));
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    setIsLoading(true);
    
    try {
      const gameData = {
        name: gameName,
        max_rounds: maxRounds,
        description,
        is_public: isPublic,
        demand_pattern: {
          type: demandPattern,
          params: {}
        },
        player_assignments: players.map(player => ({
          role: player.role.toUpperCase(),
          player_type: player.playerType,
          strategy: player.strategy,
          can_see_demand: player.canSeeDemand,
          user_id: player.userId // In a real app, this would be set based on user selection
        }))
      };
      
      const newGame = await api.createGame(gameData);
      return newGame;
      
    } catch (error) {
      console.error('Error creating game:', error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const handleCreateGame = async (e) => {
    if (e) e.preventDefault();
    try {
      const response = await handleSubmit();
      
      // Show success toast
      toast({
        title: 'Game created!',
        description: 'The mixed game has been created successfully.',
        status: 'success',
        duration: 2000,
        isClosable: true,
      });
      
      // Navigate to the game board after a short delay
      setTimeout(() => {
        if (response && response.id) {
          navigate(`/games/mixed/${response.id}`);
        } else {
          navigate('/games');
        }
      }, 1500);
      
      return response;
    } catch (error) {
      console.error('Error creating game:', error);
      toast({
        title: 'Error creating game',
        description: error.response?.data?.detail || 'Failed to create game. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      throw error;
    }
  };

  return (
    <PageLayout title="Create Mixed Game">
      <VStack as="form" onSubmit={handleCreateGame} spacing={6} align="stretch" maxW="4xl" mx="auto">
        {/* Game Details Card */}
        <Card variant="outline" bg={cardBg} borderColor={borderColor}>
          <CardHeader pb={2}>
            <Heading size="md">Game Settings</Heading>
            <Text color="gray.500" fontSize="sm">Configure the basic settings for your game</Text>
          </CardHeader>
          <CardBody pt={0}>
            <VStack spacing={5}>
              <FormControl>
                <FormLabel>Game Name</FormLabel>
                <Input 
                  value={gameName}
                  onChange={(e) => setGameName(e.target.value)}
                  placeholder="Enter game name"
                  size="lg"
                />
              </FormControl>

              <Grid templateColumns={{ base: '1fr', md: '1fr 1fr' }} gap={6} w="full">
                <FormControl>
                  <FormLabel>Maximum Rounds</FormLabel>
                  <NumberInput 
                    min={1} 
                    max={999}
                    value={maxRounds}
                    onChange={(value) => setMaxRounds(parseInt(value) || 0)}
                  >
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                  <FormHelperText>Maximum 999 rounds</FormHelperText>
                </FormControl>

                <FormControl display="flex" flexDirection="column" justifyContent="flex-end">
                  <FormLabel mb={0}>Game Visibility</FormLabel>
                  <HStack spacing={4} align="center">
                    <Text fontSize="sm" color={isPublic ? 'gray.500' : 'gray.800'} fontWeight={isPublic ? 'normal' : 'medium'}>Private</Text>
                    <Switch 
                      isChecked={isPublic}
                      onChange={(e) => setIsPublic(e.target.checked)}
                      colorScheme="blue"
                      size="lg"
                    />
                    <Text fontSize="sm" color={isPublic ? 'blue.600' : 'gray.500'} fontWeight={isPublic ? 'medium' : 'normal'}>
                      {isPublic ? 'Public' : 'Private'}
                    </Text>
                  </HStack>
                  <FormHelperText>
                    {isPublic 
                      ? 'Anyone can join this game' 
                      : 'Only players with the link can join'}
                  </FormHelperText>
                </FormControl>
              </Grid>

              <FormControl>
                <FormLabel>Description (Optional)</FormLabel>
                <Input 
                  as="textarea"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Enter a brief description of the game"
                  rows={3}
                  size="lg"
                />
              </FormControl>
            </VStack>
          </CardBody>
        </Card>

        {/* Player Configuration Card */}
        <Card variant="outline" bg={cardBg} borderColor={borderColor} mt={6}>
          <CardHeader pb={2}>
            <Heading size="md">Player Configuration</Heading>
            <Text color="gray.500" fontSize="sm">Configure human and AI players for each role</Text>
          </CardHeader>
          <Divider />
          <CardBody>
            <VStack spacing={6}>
              {players.map((player, index) => (
                <Box 
                  key={player.role} 
                  w="full"
                  p={5} 
                  borderWidth="1px"
                  borderRadius="lg"
                  borderColor={borderColor}
                  bg={cardBg}
                  _hover={{
                    boxShadow: 'md',
                    transform: 'translateY(-2px)',
                    transition: 'all 0.2s',
                  }}
                >
                  <HStack justify="space-between" mb={4}>
                    <HStack>
                      <Text fontSize="lg" fontWeight="semibold" textTransform="capitalize">
                        {player.role}
                      </Text>
                      {player.role === 'retailer' && (
                        <Badge colorScheme="blue" variant="subtle" borderRadius="full" px={2}>
                          Required
                        </Badge>
                      )}
                    </HStack>
                    
                    <HStack 
                      spacing={0} 
                      bg="gray.100" 
                      p={1} 
                      borderRadius="md"
                      borderWidth="1px"
                      borderColor="gray.200"
                      _dark={{
                        bg: 'gray.700',
                        borderColor: 'gray.600'
                      }}
                    >
                      <Button 
                        size="sm" 
                        variant={player.playerType === 'human' ? 'solid' : 'ghost'}
                        colorScheme={player.playerType === 'human' ? 'blue' : 'gray'}
                        onClick={() => handlePlayerTypeChange(index, 'human')}
                        leftIcon={
                          <Box 
                            w={2} 
                            h={2} 
                            bg={player.playerType === 'human' ? 'white' : 'gray.500'} 
                            borderRadius="full" 
                          />
                        }
                        textTransform="none"
                        fontWeight="500"
                        borderRadius="md"
                        flex={1}
                        _active={{
                          transform: 'none',
                          bg: player.playerType === 'human' ? 'blue.600' : 'gray.200',
                          _dark: {
                            bg: player.playerType === 'human' ? 'blue.600' : 'gray.600'
                          }
                        }}
                        _hover={{
                          bg: player.playerType === 'human' ? 'blue.600' : 'gray.200',
                          _dark: {
                            bg: player.playerType === 'human' ? 'blue.600' : 'gray.600'
                          }
                        }}
                      >
                        Human
                      </Button>
                      <Button 
                        size="sm" 
                        variant={player.playerType === 'ai' ? 'solid' : 'ghost'}
                        colorScheme={player.playerType === 'ai' ? 'blue' : 'gray'}
                        onClick={() => handlePlayerTypeChange(index, 'ai')}
                        leftIcon={
                          <Box 
                            w={2} 
                            h={2} 
                            bg={player.playerType === 'ai' ? 'white' : 'gray.500'} 
                            borderRadius="full" 
                          />
                        }
                        textTransform="none"
                        fontWeight="500"
                        borderRadius="md"
                        flex={1}
                        _active={{
                          transform: 'none',
                          bg: player.playerType === 'ai' ? 'blue.600' : 'gray.200',
                          _dark: {
                            bg: player.playerType === 'ai' ? 'blue.600' : 'gray.600'
                          }
                        }}
                        _hover={{
                          bg: player.playerType === 'ai' ? 'blue.600' : 'gray.200',
                          _dark: {
                            bg: player.playerType === 'ai' ? 'blue.600' : 'gray.600'
                          }
                        }}
                      >
                        AI Agent
                      </Button>
                    </HStack>
                  </HStack>

                  {player.playerType === 'ai' && (
                    <FormControl mt={4}>
                      <FormLabel>AI Strategy</FormLabel>
                      <Select 
                        value={player.strategy}
                        onChange={(e) => handleStrategyChange(index, e.target.value)}
                        size="md"
                        bg="white"
                        _dark={{
                          bg: 'gray.700',
                          borderColor: 'gray.600',
                          _hover: { borderColor: 'gray.500' },
                          _focus: { borderColor: 'blue.500', boxShadow: '0 0 0 1px #3182ce' }
                        }}
                      >
                        {agentStrategies.map((group, i) => [
                          <option key={`group-${i}`} disabled style={{ fontWeight: 'bold', backgroundColor: '#f5f5f5' }}>
                            {group.group}
                          </option>,
                          ...group.options.map(option => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))
                        ])}
                      </Select>
                      <FormHelperText>
                        {player.strategy === 'NAIVE' && 'Simple strategy that matches orders to demand'}
                        {player.strategy === 'BULLWHIP' && 'Tends to overreact to demand changes'}
                        {player.strategy === 'CONSERVATIVE' && 'Maintains stable inventory levels'}
                        {player.strategy === 'RANDOM' && 'Makes random order decisions'}
                        {player.strategy === 'DEMAND_DRIVEN' && 'Advanced strategy that analyzes demand patterns'}
                        {player.strategy === 'COST_OPTIMIZATION' && 'Optimizes for lowest possible costs'}
                        {player.strategy === 'LLM_CONSERVATIVE' && 'AI-powered strategy using language models'}
                        {player.strategy === 'LLM_BALANCED' && 'Advanced AI with learning capabilities'}
                        {player.strategy === 'LLM_AGGRESSIVE' && 'Aggressive AI strategy'}
                        {player.strategy === 'LLM_ADAPTIVE' && 'Adaptive AI strategy'}
                      </FormHelperText>
                    </FormControl>
                  )}

                  <FormControl display="flex" alignItems="center" mt={4}>
                    <Switch
                      id={`demand-${index}`}
                      isChecked={player.canSeeDemand}
                      onChange={(e) => handleCanSeeDemandChange(index, e.target.checked)}
                      isDisabled={player.role === 'retailer'}
                      colorScheme="blue"
                      mr={3}
                    />
                    <FormLabel htmlFor={`demand-${index}`} mb={0} opacity={player.role === 'retailer' ? 0.7 : 1}>
                      Can see customer demand
                      {player.role === 'retailer' && ' (Always enabled for Retailer)'}
                    </FormLabel>
                  </FormControl>
                </Box>
              ))}
            </VStack>
          </CardBody>
        </Card>

        <Button 
          type="submit"
          isLoading={isLoading}
          loadingText="Creating Game..."
          colorScheme="blue"
          size="lg"
          mt={6}
          width="100%"
          textTransform="none"
          fontWeight="500"
          height="44px"
        >
          Create Game
        </Button>
      </VStack>
    </PageLayout>
  );
};

export default CreateMixedGame;
