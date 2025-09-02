import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Button, 
  Container, 
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
  useToast
} from '@chakra-ui/react';
import { mixedGameApi } from '../services/api';

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
  const [players, setPlayers] = useState(
    playerRoles.map(role => ({
      role: role.value,
      playerType: 'HUMAN',
      strategy: agentStrategies[0].value,
      canSeeDemand: false,
      userId: null // Will be set to current user or selected user
    }))
  );
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const toast = useToast();

  const handlePlayerTypeChange = (role, type) => {
    setPlayers(players.map(player => 
      player.role === role 
        ? { 
            ...player, 
            playerType: type,
            // Reset strategy to default when changing to human
            ...(type === 'HUMAN' && { strategy: agentStrategies[0].value })
          } 
        : player
    ));
  };

  const handleStrategyChange = (role, strategy) => {
    setPlayers(players.map(player => 
      player.role === role ? { ...player, strategy } : player
    ));
  };

  const handleDemandVisibilityChange = (role, canSeeDemand) => {
    setPlayers(players.map(player => 
      player.role === role ? { ...player, canSeeDemand } : player
    ));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
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
      
      const newGame = await mixedGameApi.createGame(gameData);
      
      toast({
        title: 'Game created!',
        description: 'The mixed game has been created successfully.',
        status: 'success',
        duration: 5000,
        isClosable: true,
      });
      
      // Navigate to the new game
      navigate(`/games/mixed/${newGame.id}`);
    } catch (error) {
      console.error('Error creating game:', error);
      toast({
        title: 'Error creating game',
        description: error.response?.data?.detail || 'Failed to create game',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxW="container.lg" py={8}>
      <Box bg="white" p={6} rounded="lg" boxShadow="md">
        <Text fontSize="2xl" fontWeight="bold" mb={6}>
          Create Mixed Human/AI Game
        </Text>
        
        <form onSubmit={handleSubmit}>
          <VStack spacing={6} align="stretch">
            <FormControl isRequired>
              <FormLabel>Game Name</FormLabel>
              <Input 
                value={gameName}
                onChange={(e) => setGameName(e.target.value)}
                placeholder="Enter game name"
              />
            </FormControl>
            
            <FormControl>
              <FormLabel>Description</FormLabel>
              <Input 
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Enter game description"
              />
            </FormControl>
            
            <FormControl isRequired>
              <FormLabel>Maximum Rounds</FormLabel>
              <NumberInput 
                min={1} 
                max={100} 
                value={maxRounds}
                onChange={(valueString) => setMaxRounds(parseInt(valueString) || 1)}
              >
                <NumberInputField />
                <NumberInputStepper>
                  <NumberIncrementStepper />
                  <NumberDecrementStepper />
                </NumberInputStepper>
              </NumberInput>
            </FormControl>
            
            <FormControl>
              <FormLabel>Demand Pattern</FormLabel>
              <Select 
                value={demandPattern}
                onChange={(e) => setDemandPattern(e.target.value)}
              >
                {demandPatterns.map(pattern => (
                  <option key={pattern.value} value={pattern.value}>
                    {pattern.label}
                  </option>
                ))}
              </Select>
            </FormControl>
            
            <FormControl>
              <Checkbox 
                isChecked={isPublic}
                onChange={(e) => setIsPublic(e.target.checked)}
              >
                Make this game public
              </Checkbox>
            </FormControl>
            
            <Box mt={6}>
              <Text fontSize="lg" fontWeight="semibold" mb={4}>
                Player Assignments
              </Text>
              
              <VStack spacing={6} align="stretch">
                {players.map((player) => (
                  <Box 
                    key={player.role}
                    p={4} 
                    borderWidth="1px" 
                    borderRadius="md"
                    bg="gray.50"
                  >
                    <HStack justify="space-between" mb={3}>
                      <Text fontWeight="medium">
                        {playerRoles.find(r => r.value === player.role)?.label}
                      </Text>
                      <HStack>
                        <Button
                          size="sm"
                          colorScheme={player.playerType === 'HUMAN' ? 'blue' : 'gray'}
                          variant={player.playerType === 'HUMAN' ? 'solid' : 'outline'}
                          onClick={() => handlePlayerTypeChange(player.role, 'HUMAN')}
                        >
                          Human
                        </Button>
                        <Button
                          size="sm"
                          colorScheme={player.playerType === 'AGENT' ? 'teal' : 'gray'}
                          variant={player.playerType === 'AGENT' ? 'solid' : 'outline'}
                          onClick={() => handlePlayerTypeChange(player.role, 'AGENT')}
                        >
                          AI Agent
                        </Button>
                      </HStack>
                    </HStack>
                    
                    {player.playerType === 'AGENT' && (
                      <VStack align="stretch" spacing={3} mt={3}>
                        <FormControl>
                          <FormLabel>Strategy</FormLabel>
                          <Select
                            value={player.strategy}
                            onChange={(e) => handleStrategyChange(player.role, e.target.value)}
                            size="sm"
                          >
                            <optgroup label="Basic Strategies">
                              {agentStrategies[0].options.map(strategy => (
                                <option key={strategy.value} value={strategy.value}>
                                  {strategy.label}
                                </option>
                              ))}
                            </optgroup>
                            <optgroup label="Advanced Strategies">
                              {agentStrategies[1].options.map(strategy => (
                                <option key={strategy.value} value={strategy.value}>
                                  {strategy.label}
                                </option>
                              ))}
                            </optgroup>
                            <optgroup label="AI-Powered (LLM)">
                              {agentStrategies[2].options.map(strategy => (
                                <option key={strategy.value} value={strategy.value}>
                                  {strategy.label}
                                </option>
                              ))}
                            </optgroup>
                          </Select>
                        </FormControl>
                        
                        <FormControl>
                          <Checkbox
                            isChecked={player.canSeeDemand}
                            onChange={(e) => handleDemandVisibilityChange(player.role, e.target.checked)}
                          >
                            Can see actual customer demand
                          </Checkbox>
                        </FormControl>
                      </VStack>
                    )}
                    
                    {player.playerType === 'HUMAN' && (
                      <Text fontSize="sm" color="gray.600" mt={2}>
                        A human player will need to join this role.
                      </Text>
                    )}
                  </Box>
                ))}
              </VStack>
            </Box>
            
            <HStack justify="flex-end" mt={8}>
              <Button 
                type="button" 
                variant="outline" 
                onClick={() => navigate(-1)}
                isDisabled={isLoading}
              >
                Cancel
              </Button>
              <Button 
                type="submit" 
                colorScheme="blue" 
                isLoading={isLoading}
                loadingText="Creating..."
              >
                Create Game
              </Button>
            </HStack>
          </VStack>
        </form>
      </Box>
    </Container>
  );
};

export default CreateMixedGame;
