import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
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
  useColorModeValue,
  Switch,
  FormHelperText,
  Grid,
  Badge,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription
} from '@chakra-ui/react';
import PageLayout from '../components/PageLayout';
import { getAllConfigs } from '../services/supplyChainConfigService';
import PricingConfigForm from '../components/PricingConfigForm';
import { api, mixedGameApi } from '../services/api';
import { getModelStatus } from '../services/modelService';
import { useSystemConfig } from '../contexts/SystemConfigContext.jsx';

const playerRoles = [
  { value: 'retailer', label: 'Retailer' },
  { value: 'wholesaler', label: 'Wholesaler' },
  { value: 'distributor', label: 'Distributor' },
  { value: 'factory', label: 'Factory' },
];

const agentStrategies = [
  {
    group: 'Basic',
    options: [
      { value: 'NAIVE', label: 'Naive (heuristic)' },
      { value: 'BULLWHIP', label: 'Bullwhip (heuristic)' },
      { value: 'CONSERVATIVE', label: 'Conservative (heuristic)' },
      { value: 'RANDOM', label: 'Random (heuristic)' },
    ]
  },
  {
    group: 'LLM',
    options: [
      { value: 'LLM_CONSERVATIVE', label: 'LLM - Conservative' },
      { value: 'LLM_BALANCED', label: 'LLM - Balanced' },
      { value: 'LLM_AGGRESSIVE', label: 'LLM - Aggressive' },
      { value: 'LLM_ADAPTIVE', label: 'LLM - Adaptive' },
    ]
  },
  {
    group: 'Daybreak',
    options: [
      { value: 'DAYBREAK_DTCE', label: 'Daybreak - Roles', requiresModel: true },
      { value: 'DAYBREAK_DTCE_CENTRAL', label: 'Daybreak - Roles + Supervisor', requiresModel: true },
      { value: 'DAYBREAK_DTCE_GLOBAL', label: 'Daybreak - SC Orchestrator', requiresModel: true },
    ]
  }
];

const HUMAN_ASSIGNMENT = 'HUMAN';

const strategyLabelMap = agentStrategies.reduce((acc, group) => {
  group.options.forEach((option) => {
    acc[option.value] = option.label;
  });
  return acc;
}, {});

const getStrategyLabel = (strategy) => strategyLabelMap[strategy] || strategy;

const daybreakStrategyDescriptions = {
  DAYBREAK_DTCE: 'Daybreak - Roles deploys one agent per supply chain role.',
  DAYBREAK_DTCE_CENTRAL:
    'Daybreak - Roles + Supervisor allows a network supervisor to adjust orders within the configured percentage.',
  DAYBREAK_DTCE_GLOBAL: 'Daybreak - SC Orchestrator runs a single agent across the entire supply chain.',
};

const strategyDescriptions = {
  NAIVE: 'Basic heuristic that matches orders to demand.',
  BULLWHIP: 'Tends to overreact to demand changes.',
  CONSERVATIVE: 'Maintains stable inventory levels.',
  RANDOM: 'Makes random order decisions.',
  DEMAND_DRIVEN: 'LLM: demand-driven analysis.',
  COST_OPTIMIZATION: 'LLM: optimizes for lower cost.',
  LLM_CONSERVATIVE: 'AI-powered strategy using language models.',
  LLM_BALANCED: 'Advanced AI with learning capabilities.',
  LLM_AGGRESSIVE: 'Aggressive AI strategy.',
  LLM_ADAPTIVE: 'Adaptive AI strategy.',
  DAYBREAK_DTCE: daybreakStrategyDescriptions.DAYBREAK_DTCE,
  DAYBREAK_DTCE_CENTRAL: daybreakStrategyDescriptions.DAYBREAK_DTCE_CENTRAL,
  DAYBREAK_DTCE_GLOBAL: daybreakStrategyDescriptions.DAYBREAK_DTCE_GLOBAL,
};

const demandPatterns = [
  { value: 'classic', label: 'Classic (Step Increase)' },
  { value: 'random', label: 'Random' },
  { value: 'seasonal', label: 'Seasonal' },
  { value: 'constant', label: 'Constant' },
];

const clampOverridePercent = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 5;
  }
  return Math.min(50, Math.max(5, numeric));
};

// Controlled per-node policy editor
const PerNodePolicyEditor = ({ value, onChange, ranges = {} }) => {
  const [selected, setSelected] = useState('retailer');
  const policies = value;
  const current = policies[selected] || {};
  const update = (field, val) => {
    const next = { ...policies, [selected]: { ...policies[selected], [field]: val } };
    onChange(next);
  };

  const margin = (current.price || 0) - (current.standard_cost || 0) - (current.variable_cost || 0);

  return (
    <Box>
      <HStack spacing={4} mb={4}>
        <FormControl maxW="xs">
          <FormLabel>Node Type</FormLabel>
          <Select value={selected} onChange={(e) => setSelected(e.target.value)}>
            <option value="retailer">Retailer</option>
            <option value="distributor">Distributor</option>
            <option value="manufacturer">Manufacturer</option>
            <option value="supplier">Supplier</option>
          </Select>
        </FormControl>
        <Text color="gray.500" fontSize="sm">Configure policy values for the selected node.</Text>
      </HStack>

      <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)' }} gap={6}>
        <FormControl>
          <FormLabel>Order Leadtime (weeks)</FormLabel>
          <NumberInput min={ranges.info_delay?.min ?? 0} max={ranges.info_delay?.max ?? 8} value={current.info_delay}
            onChange={(v) => update('info_delay', parseInt(v) || 0)}>
            <NumberInputField />
            <NumberInputStepper>
              <NumberIncrementStepper />
              <NumberDecrementStepper />
            </NumberInputStepper>
          </NumberInput>
        </FormControl>
        <FormControl>
          <FormLabel>Supply Leadtime (weeks)</FormLabel>
          <NumberInput min={ranges.ship_delay?.min ?? 0} max={ranges.ship_delay?.max ?? 8} value={current.ship_delay}
            onChange={(v) => update('ship_delay', parseInt(v) || 0)}>
            <NumberInputField />
            <NumberInputStepper>
              <NumberIncrementStepper />
              <NumberDecrementStepper />
            </NumberInputStepper>
          </NumberInput>
        </FormControl>
        <FormControl>
          <FormLabel>Initial Inventory (units)</FormLabel>
          <NumberInput min={ranges.init_inventory?.min ?? 0} max={ranges.init_inventory?.max ?? 1000} value={current.init_inventory}
            onChange={(v) => update('init_inventory', parseInt(v) || 0)}>
            <NumberInputField />
            <NumberInputStepper>
              <NumberIncrementStepper />
              <NumberDecrementStepper />
            </NumberInputStepper>
          </NumberInput>
        </FormControl>
        <FormControl>
          <FormLabel>Minimum Order Quantity</FormLabel>
          <NumberInput min={ranges.min_order_qty?.min ?? 0} max={ranges.min_order_qty?.max ?? 1000} value={current.min_order_qty || 0}
            onChange={(v) => update('min_order_qty', parseInt(v) || 0)}>
            <NumberInputField />
            <NumberInputStepper>
              <NumberIncrementStepper />
              <NumberDecrementStepper />
            </NumberInputStepper>
          </NumberInput>
        </FormControl>
        <FormControl>
          <FormLabel>Price</FormLabel>
          <NumberInput min={ranges.price?.min ?? 0} max={ranges.price?.max ?? 10000} step={0.01} precision={2} value={current.price}
            onChange={(v) => update('price', parseFloat(v) || 0)}>
            <NumberInputField />
            <NumberInputStepper>
              <NumberIncrementStepper />
              <NumberDecrementStepper />
            </NumberInputStepper>
          </NumberInput>
        </FormControl>
        <FormControl>
          <FormLabel>Standard Cost</FormLabel>
          <NumberInput min={ranges.standard_cost?.min ?? 0} max={ranges.standard_cost?.max ?? 10000} step={0.01} precision={2} value={current.standard_cost}
            onChange={(v) => update('standard_cost', parseFloat(v) || 0)}>
            <NumberInputField />
            <NumberInputStepper>
              <NumberIncrementStepper />
              <NumberDecrementStepper />
            </NumberInputStepper>
          </NumberInput>
        </FormControl>
        <FormControl>
          <FormLabel>Variable Cost</FormLabel>
          <NumberInput min={ranges.variable_cost?.min ?? 0} max={ranges.variable_cost?.max ?? 10000} step={0.01} precision={2} value={current.variable_cost}
            onChange={(v) => update('variable_cost', parseFloat(v) || 0)}>
            <NumberInputField />
            <NumberInputStepper>
              <NumberIncrementStepper />
              <NumberDecrementStepper />
            </NumberInputStepper>
          </NumberInput>
          <FormHelperText>Margin = price - std cost - variable cost</FormHelperText>
        </FormControl>
      </Grid>
      <Text mt={2} fontSize="sm" color={margin >= 0 ? 'green.600' : 'red.600'}>
        Margin: {margin.toFixed(2)}
      </Text>
    </Box>
  );
};

const CreateMixedGame = () => {
  const [searchParams] = useSearchParams();
  const [gameName, setGameName] = useState(searchParams.get('name') || '');
  const [systemConfig, setSystemConfig] = useState({
    // Default system configuration values
    min_order_quantity: 0,
    max_order_quantity: 100,
    min_holding_cost: 0,
    max_holding_cost: 10,
    min_backlog_cost: 0,
    max_backlog_cost: 20,
    min_demand: 0,
    max_demand: 100,
    min_lead_time: 0,
    max_lead_time: 4,
    min_starting_inventory: 0,
    max_starting_inventory: 100,
  });
  const [maxRounds, setMaxRounds] = useState(20);
  const [description, setDescription] = useState(searchParams.get('description') || '');
  const [isPublic, setIsPublic] = useState(true);
  const [demandPattern, setDemandPattern] = useState(demandPatterns[0].value);
  const [initialDemand, setInitialDemand] = useState(4);
  const [demandChangeWeek, setDemandChangeWeek] = useState(6);
  const [finalDemand, setFinalDemand] = useState(8);
  const [pricingConfig, setPricingConfig] = useState({
    retailer: { selling_price: 100.0, standard_cost: 80.0 },
    wholesaler: { selling_price: 75.0, standard_cost: 60.0 },
    distributor: { selling_price: 60.0, standard_cost: 45.0 },
    factory: { selling_price: 45.0, standard_cost: 30.0 }
  });
  // Missing local state for system configuration ranges and per-node policies
  // Node policies for each role in the supply chain
  const [nodePolicies, setNodePolicies] = useState({
    retailer: {
      // Default policy values for retailer
      policy_type: 'base_stock',
      base_stock_level: 50,
      reorder_point: 20,
      order_up_to: 100,
      smoothing_alpha: 0.3,
      smoothing_beta: 0.1,
      smoothing_gamma: 0.2,
      forecast_horizon: 4
    },
    wholesaler: {
      policy_type: 'base_stock',
      base_stock_level: 100,
      reorder_point: 40,
      order_up_to: 200,
      smoothing_alpha: 0.3,
      smoothing_beta: 0.1,
      smoothing_gamma: 0.2,
      forecast_horizon: 4
    },
    distributor: {
      policy_type: 'base_stock',
      base_stock_level: 150,
      reorder_point: 60,
      order_up_to: 300,
      smoothing_alpha: 0.3,
      smoothing_beta: 0.1,
      smoothing_gamma: 0.2,
      forecast_horizon: 4
    },
    factory: {
      policy_type: 'base_stock',
      base_stock_level: 200,
      reorder_point: 80,
      order_up_to: 400,
      smoothing_alpha: 0.3,
      smoothing_beta: 0.1,
      smoothing_gamma: 0.2,
      forecast_horizon: 4
    }
  });
  // Policy/Simulation settings (bounded)
  const [policy, setPolicy] = useState({
    info_delay: 2,         // order delay (weeks)
    ship_delay: 2,         // delivery delay (weeks)
    init_inventory: 12,    // starting inventory per role
    holding_cost: 0.5,     // per unit per week
    backlog_cost: 1.0,     // per unit per week
    max_inbound_per_link: 100, // shipment capacity
    max_order: 100,
  });
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const { user } = useAuth();
  const [modelStatus, setModelStatus] = useState(null);
  const { ranges: systemRanges } = useSystemConfig();
  const [availableUsers, setAvailableUsers] = useState([]);
  const [loadingUsers, setLoadingUsers] = useState(true);
  
  const [players, setPlayers] = useState(
    playerRoles.map(role => ({
      role: role.value,
      playerType: 'ai', // Default to AI for all roles initially
      strategy: 'NAIVE',
      canSeeDemand: role.value === 'retailer',
      userId: role.value === 'retailer' && user ? user.id : null,
      llmModel: 'gpt-4o-mini',
      daybreakOverridePct: 5,
    }))
  );
  const [isLoading, setIsLoading] = useState(false);
  const [configs, setConfigs] = useState([]);
  const navigate = useNavigate();
  const toast = useToast();

  // Fetch available users
  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const response = await api.get('/auth/users/');
        setAvailableUsers(response.data);
      } catch (error) {
        console.error('Error fetching users:', error);
        toast({
          title: 'Error',
          description: 'Failed to load users',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      } finally {
        setLoadingUsers(false);
      }
    };

    fetchUsers();
  }, [toast]);

  // Load available saved Game Configurations (core game setup)
  useEffect(() => {
    (async () => {
      try {
        const cfgs = await getAllConfigs();
        if (Array.isArray(cfgs)) setConfigs(cfgs);
      } catch (e) {
        // non-blocking
      }
    })();
  }, []);

  // Load Daybreak agent model status
  useEffect(() => {
    (async () => {
      try {
        const status = await getModelStatus();
        setModelStatus(status);
      } catch (e) {
        console.error('Failed to get model status', e);
      }
    })();
  }, []);

  // Optional prefill via query params for node policies (JSON-encoded)
  useEffect(() => {
    const np = searchParams.get('node_policies');
    if (np) {
      try {
        const parsed = JSON.parse(np);
        if (parsed && typeof parsed === 'object') setNodePolicies(parsed);
      } catch {}
    }
    const sc = searchParams.get('system_config');
    if (sc) {
      try {
        const parsed = JSON.parse(sc);
        if (parsed && typeof parsed === 'object') setSystemConfig(parsed);
      } catch {}
    }
    const pc = searchParams.get('pricing_config');
    if (pc) {
      try {
        const parsed = JSON.parse(pc);
        if (parsed && typeof parsed === 'object') setPricingConfig(parsed);
      } catch {}
    }
    // Load system ranges from backend, then merge local defaults and localStorage
    (async () => {
      try {
        const serverCfg = await mixedGameApi.getSystemConfig();
        if (serverCfg && typeof serverCfg === 'object') setSystemConfig((prev)=> ({...prev, ...serverCfg}));
      } catch {}
    })();
    // Load system ranges from localStorage if present
    const stored = localStorage.getItem('systemConfigRanges');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        if (parsed && typeof parsed === 'object') setSystemConfig((prev)=> ({...prev, ...parsed}));
      } catch {}
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // If context ranges become available later, merge them
  useEffect(() => {
    if (systemRanges && Object.keys(systemRanges).length) {
      setSystemConfig(prev => ({ ...prev, ...systemRanges }));
    }
  }, [systemRanges]);


  const handlePlayerTypeChange = (index, type) => {
    setPlayers((prevPlayers) =>
      prevPlayers.map((player, i) => {
        if (i !== index) {
          return player;
        }
        const updatedPlayer = {
          ...player,
          playerType: type,
        };
        if (type === 'human') {
          updatedPlayer.strategy = agentStrategies[0].options[0].value;
          if (player.role === 'retailer' && !player.userId && user) {
            updatedPlayer.userId = user.id;
          }
          updatedPlayer.daybreakOverridePct = undefined;
        } else if (type === 'ai' && !updatedPlayer.llmModel) {
          updatedPlayer.llmModel = 'gpt-4o-mini';
        }
        return updatedPlayer;
      })
    );
  };

  const handleStrategyChange = (index, strategy) => {
    setPlayers((prevPlayers) =>
      prevPlayers.map((player, i) => {
        if (i !== index) {
          return player;
        }
        const updated = { ...player, strategy };
        if (String(strategy).startsWith('LLM_') && !updated.llmModel) {
          updated.llmModel = 'gpt-4o-mini';
        }
        if (strategy === 'DAYBREAK_DTCE_CENTRAL') {
          const basePct = player.daybreakOverridePct ?? 5;
          updated.daybreakOverridePct = clampOverridePercent(basePct);
        }
        return updated;
      })
    );
  };

  const handleAssignmentSelect = (index, assignmentValue) => {
    if (assignmentValue === HUMAN_ASSIGNMENT) {
      handlePlayerTypeChange(index, 'human');
      return;
    }
    handlePlayerTypeChange(index, 'ai');
    handleStrategyChange(index, assignmentValue);
  };

  const handleUserChange = (index, userId) => {
    setPlayers((prevPlayers) =>
      prevPlayers.map((player, i) =>
        i === index ? { ...player, userId: userId || null } : player
      )
    );
  };

  const handleCanSeeDemandChange = (index, canSeeDemand) => {
    setPlayers((prevPlayers) =>
      prevPlayers.map((player, i) => (i === index ? { ...player, canSeeDemand } : player))
    );
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    
    // Validate that each human role has a user assigned
    const invalidPlayers = players.filter(
      p => p.playerType === 'human' && !p.userId
    );
    
    if (invalidPlayers.length > 0) {
      toast({
        title: 'Validation Error',
        description: 'Please assign a user to all human roles',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      return null;
    }
    
    // Validate pricing configuration
    const invalidPricing = Object.entries(pricingConfig).some(([role, prices]) => {
      return !prices.selling_price || !prices.standard_cost || 
             prices.selling_price <= prices.standard_cost;
    });
    
    if (invalidPricing) {
      toast({
        title: 'Validation Error',
        description: 'Please ensure selling price is greater than standard cost for all roles',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      return null;
    }
    
    setIsLoading(true);
    
    try {
      const gameData = {
        name: gameName,
        max_rounds: maxRounds,
        description,
        is_public: isPublic,
        demand_pattern: {
          type: demandPattern,
          params: demandPattern === 'classic'
            ? {
                initial_demand: Math.max(0, Number.isFinite(Number(initialDemand)) ? Number(initialDemand) : 0),
                change_week: Math.max(1, Number.isFinite(Number(demandChangeWeek)) ? Number(demandChangeWeek) : 1),
                final_demand: Math.max(0, Number.isFinite(Number(finalDemand)) ? Number(finalDemand) : 0),
              }
            : {}
        },
        node_policies: nodePolicies,
        system_config: systemConfig,
        global_policy: policy,
        pricing_config: {
          retailer: {
            selling_price: parseFloat(pricingConfig.retailer.selling_price),
            standard_cost: parseFloat(pricingConfig.retailer.standard_cost)
          },
          wholesaler: {
            selling_price: parseFloat(pricingConfig.wholesaler.selling_price),
            standard_cost: parseFloat(pricingConfig.wholesaler.standard_cost)
          },
          distributor: {
            selling_price: parseFloat(pricingConfig.distributor.selling_price),
            standard_cost: parseFloat(pricingConfig.distributor.standard_cost)
          },
          factory: {
            selling_price: parseFloat(pricingConfig.factory.selling_price),
            standard_cost: parseFloat(pricingConfig.factory.standard_cost)
          }
        },
        player_assignments: players.map(player => {
          const isAi = player.playerType === 'ai';
          const strategyValue = player.strategy ? player.strategy.toLowerCase() : null;
          const overridePercent = player.strategy === 'DAYBREAK_DTCE_CENTRAL'
            ? clampOverridePercent(player.daybreakOverridePct) / 100
            : null;

          return {
            role: player.role.toUpperCase(),
            player_type: isAi ? 'agent' : 'human',
            strategy: strategyValue,
            can_see_demand: player.canSeeDemand,
            user_id: player.userId || null,
            llm_model: (isAi && String(player.strategy).startsWith('LLM_'))
              ? player.llmModel
              : null,
            daybreak_override_pct: overridePercent,
          };
        })
      };
      
      const newGame = await mixedGameApi.createGame(gameData);
      return newGame;
      
    } catch (error) {
      console.error('Error creating game:', error);
      toast({
        title: 'Error',
        description: error.response?.data?.detail || 'Failed to create game',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
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
          navigate(`/games/${response.id}`);
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
    <PageLayout title="Mixed Game Definition">
      {/* Quick links to saved Game Configurations */}
      {configs?.length > 0 && (
        <Box mb={4}>
          <Alert status="info" borderRadius="md" mb={3}>
            <AlertIcon />
            <Box>
              <AlertTitle mr={2}>Saved Game Configurations</AlertTitle>
              <AlertDescription fontSize="sm">Use a configuration to prefill this form.</AlertDescription>
            </Box>
          </Alert>
          <HStack spacing={2} wrap="wrap">
            {configs.map((c) => (
              <Button as={Link} key={c.id} to={`/games/new-from-config/${c.id}`} size="sm" variant="outline">
                Use: {c.name}
              </Button>
            ))}
          </HStack>
        </Box>
      )}
      {modelStatus && !modelStatus.is_trained && (
        <Alert status="error" variant="left-accent" mb={6} borderRadius="md">
          <AlertIcon boxSize="16px" color="red.500" />
          <Box>
            <AlertTitle>Daybreak Agent Not Trained</AlertTitle>
            <AlertDescription fontSize="sm">
              The Daybreak agent has not yet been trained, so it cannot be used until training completes. You may still select Basic (heuristics) or LLM agents.
            </AlertDescription>
          </Box>
        </Alert>
      )}
      {(!systemConfig || Object.keys(systemConfig||{}).length === 0) && (
        <Alert status="info" variant="left-accent" mb={4} borderRadius="md">
          <AlertIcon />
          <Box>
            <AlertTitle>Using default ranges</AlertTitle>
            <AlertDescription fontSize="sm">
              You can define system-wide ranges for configuration variables in the System Configuration page.
              <Button ml={3} size="sm" colorScheme="blue" variant="ghost" onClick={() => navigate('/system-config')}>Open System Config</Button>
            </AlertDescription>
          </Box>
        </Alert>
      )}
      <VStack as="form" onSubmit={handleCreateGame} spacing={6} align="stretch" maxW="4xl" mx="auto">
        <Tabs variant="enclosed" isFitted>
          <TabList mb="1em">
            <Tab>Game Settings</Tab>
            <Tab>Pricing</Tab>
            <Tab>Players</Tab>
          </TabList>
          
          <TabPanels>
            <TabPanel p={0}>
              <VStack spacing={6}>
                {/* Game Details Card */}
                <Card variant="outline" bg={cardBg} borderColor={borderColor} w="100%" className="card-surface pad-6">
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
                          isRequired
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
                            isRequired
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
                            <Text fontSize="sm" color={isPublic ? 'gray.800' : 'gray.500'} fontWeight={isPublic ? 'medium' : 'normal'}>Public</Text>
                          </HStack>
                          <FormHelperText>
                            {isPublic 
                              ? 'Anyone can join this game' 
                              : 'Only invited players can join this game'}
                          </FormHelperText>
                        </FormControl>
                      </Grid>

                      <FormControl>
                        <FormLabel>Description (Optional)</FormLabel>
                        <Input 
                          as="textarea"
                          value={description}
                          onChange={(e) => setDescription(e.target.value)}
                          placeholder="Enter a description for your game"
                          size="lg"
                          minH="100px"
                          p={3}
                        />
                      </FormControl>
                    </VStack>
                  </CardBody>
                </Card>

                {/* Market Demand Settings */}
                <Card variant="outline" bg={cardBg} borderColor={borderColor} w="100%" className="card-surface pad-6">
                  <CardHeader pb={2}>
                    <Heading size="md">Market Demand</Heading>
                    <Text color="gray.500" fontSize="sm">Configure how customer demand evolves during the game</Text>
                  </CardHeader>
                  <CardBody pt={0}>
                    <VStack spacing={5} align="stretch">
                      <FormControl>
                        <FormLabel>Demand Pattern</FormLabel>
                        <Select value={demandPattern} onChange={(e) => setDemandPattern(e.target.value)}>
                          {demandPatterns.map((pattern) => (
                            <option key={pattern.value} value={pattern.value}>
                              {pattern.label}
                            </option>
                          ))}
                        </Select>
                        <FormHelperText>Select the demand model to use for this game</FormHelperText>
                      </FormControl>
                      {demandPattern === 'classic' && (
                        <Grid templateColumns={{ base: '1fr', md: 'repeat(3, 1fr)' }} gap={6}>
                          <FormControl>
                            <FormLabel>Initial demand</FormLabel>
                            <NumberInput min={0} value={initialDemand} onChange={(value) => setInitialDemand(parseInt(value) || 0)}>
                              <NumberInputField />
                              <NumberInputStepper>
                                <NumberIncrementStepper />
                                <NumberDecrementStepper />
                              </NumberInputStepper>
                            </NumberInput>
                            <FormHelperText>Customer demand before the change occurs</FormHelperText>
                          </FormControl>
                          <FormControl>
                            <FormLabel>Change occurs in week</FormLabel>
                            <NumberInput min={1} value={demandChangeWeek} onChange={(value) => setDemandChangeWeek(parseInt(value) || 1)}>
                              <NumberInputField />
                              <NumberInputStepper>
                                <NumberIncrementStepper />
                                <NumberDecrementStepper />
                              </NumberInputStepper>
                            </NumberInput>
                            <FormHelperText>Week when demand switches to the new level</FormHelperText>
                          </FormControl>
                          <FormControl>
                            <FormLabel>Final demand</FormLabel>
                            <NumberInput min={0} value={finalDemand} onChange={(value) => setFinalDemand(parseInt(value) || 0)}>
                              <NumberInputField />
                              <NumberInputStepper>
                                <NumberIncrementStepper />
                                <NumberDecrementStepper />
                              </NumberInputStepper>
                            </NumberInput>
                            <FormHelperText>Customer demand after the change</FormHelperText>
                          </FormControl>
                        </Grid>
                      )}
                      {demandPattern !== 'classic' && (
                        <Text color="gray.500" fontSize="sm">
                          Additional configuration for this pattern will be added in a future update.
                        </Text>
                      )}
                    </VStack>
                  </CardBody>
                </Card>

                {/* Policy Settings (bounded) */}
                <Card variant="outline" bg={cardBg} borderColor={borderColor} w="100%" className="card-surface pad-6">
                  <CardHeader pb={2}>
                    <Heading size="md">Policy Settings</Heading>
                    <Text color="gray.500" fontSize="sm">Lead times, inventory and cost parameters</Text>
                  </CardHeader>
                  <CardBody pt={0}>
                    <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)' }} gap={6}>
                      <FormControl>
                        <FormLabel>Order Leadtime (weeks)</FormLabel>
                        <NumberInput min={systemConfig.info_delay?.min ?? 0} max={systemConfig.info_delay?.max ?? 8} value={policy.info_delay}
                          onChange={(v) => setPolicy((p) => ({ ...p, info_delay: parseInt(v) || 0 }))}>
                          <NumberInputField />
                          <NumberInputStepper>
                            <NumberIncrementStepper />
                            <NumberDecrementStepper />
                          </NumberInputStepper>
                        </NumberInput>
                      </FormControl>

                      <FormControl>
                        <FormLabel>Supply Leadtime (weeks)</FormLabel>
                        <NumberInput min={systemConfig.ship_delay?.min ?? 0} max={systemConfig.ship_delay?.max ?? 8} value={policy.ship_delay}
                          onChange={(v) => setPolicy((p) => ({ ...p, ship_delay: parseInt(v) || 0 }))}>
                          <NumberInputField />
                          <NumberInputStepper>
                            <NumberIncrementStepper />
                            <NumberDecrementStepper />
                          </NumberInputStepper>
                        </NumberInput>
                      </FormControl>

                      <FormControl>
                        <FormLabel>Initial Inventory (units)</FormLabel>
                        <NumberInput min={systemConfig.init_inventory?.min ?? 0} max={systemConfig.init_inventory?.max ?? 1000} value={policy.init_inventory}
                          onChange={(v) => setPolicy((p) => ({ ...p, init_inventory: parseInt(v) || 0 }))}>
                          <NumberInputField />
                          <NumberInputStepper>
                            <NumberIncrementStepper />
                            <NumberDecrementStepper />
                          </NumberInputStepper>
                        </NumberInput>
                      </FormControl>

                      <FormControl>
                        <FormLabel>Holding Cost (per unit/week)</FormLabel>
                        <NumberInput min={systemConfig.holding_cost?.min ?? 0} max={systemConfig.holding_cost?.max ?? 100} step={0.1} precision={2} value={policy.holding_cost}
                          onChange={(v) => setPolicy((p) => ({ ...p, holding_cost: parseFloat(v) || 0 }))}>
                          <NumberInputField />
                          <NumberInputStepper>
                            <NumberIncrementStepper />
                            <NumberDecrementStepper />
                          </NumberInputStepper>
                        </NumberInput>
                      </FormControl>

                      <FormControl>
                        <FormLabel>Backlog Cost (per unit/week)</FormLabel>
                        <NumberInput min={systemConfig.backlog_cost?.min ?? 0} max={systemConfig.backlog_cost?.max ?? 200} step={0.1} precision={2} value={policy.backlog_cost}
                          onChange={(v) => setPolicy((p) => ({ ...p, backlog_cost: parseFloat(v) || 0 }))}>
                          <NumberInputField />
                          <NumberInputStepper>
                            <NumberIncrementStepper />
                            <NumberDecrementStepper />
                          </NumberInputStepper>
                        </NumberInput>
                      </FormControl>

                      <FormControl>
                        <FormLabel>Shipment Capacity (per link)</FormLabel>
                        <NumberInput min={systemConfig.max_inbound_per_link?.min ?? 10} max={systemConfig.max_inbound_per_link?.max ?? 2000} value={policy.max_inbound_per_link}
                          onChange={(v) => setPolicy((p) => ({ ...p, max_inbound_per_link: parseInt(v) || 0 }))}>
                          <NumberInputField />
                          <NumberInputStepper>
                            <NumberIncrementStepper />
                            <NumberDecrementStepper />
                          </NumberInputStepper>
                        </NumberInput>
                      </FormControl>

                      <FormControl>
                        <FormLabel>Max Order (units)</FormLabel>
                        <NumberInput min={systemConfig.max_order?.min ?? 10} max={systemConfig.max_order?.max ?? 2000} value={policy.max_order}
                          onChange={(v) => setPolicy((p) => ({ ...p, max_order: parseInt(v) || 0 }))}>
                          <NumberInputField />
                          <NumberInputStepper>
                            <NumberIncrementStepper />
                            <NumberDecrementStepper />
                          </NumberInputStepper>
                        </NumberInput>
                      </FormControl>
                    </Grid>
                    <Text color="gray.500" fontSize="xs" mt={2}>
                      Bounds chosen to reflect common Beer Game settings; adjust as needed.
                    </Text>

                    <Box mt={6}>
                      <Heading size="sm" mb={2}>Per-Node Policies</Heading>
                      <Text color="gray.500" fontSize="sm" mb={4}>
                        Select a node type and edit its leadtimes and costs. Daybreak agent availability depends on training status.
                      </Text>
                      <HStack justify="flex-end" mb={2}>
                        <Button size="sm" variant="outline" onClick={() => {
                          const mid = (k, fb=0) => { const r = systemConfig[k] || {}; const min = Number(r.min ?? fb); const max = Number(r.max ?? fb); return Math.round((min + (max - min)/2)); };
                          const priceMid = () => { const r = systemConfig.price || {}; const min = Number(r.min ?? 0); const max = Number(r.max ?? 0); return (min + (max - min)/2); };
                          const p = Math.round(priceMid() * 100)/100;
                          const stdMin = Number(systemConfig.standard_cost?.min ?? 0);
                          const vcMin = Number(systemConfig.variable_cost?.min ?? 0);
                          const defaults = (price) => ({ info_delay: mid('info_delay', 0), ship_delay: mid('ship_delay', 0), init_inventory: mid('init_inventory', 12), min_order_qty: mid('min_order_qty', 0), price, standard_cost: Math.max(stdMin, Math.round(price * 0.8 * 100)/100), variable_cost: Math.max(vcMin, Math.round(price * 0.1 * 100)/100) });
                          setNodePolicies({
                            retailer: defaults(p),
                            distributor: defaults(p),
                            manufacturer: defaults(p),
                            supplier: defaults(p),
                          });
                        }}>
                          Reset Node Policies to Server Defaults
                        </Button>
                      </HStack>
                      <PerNodePolicyEditor value={nodePolicies} onChange={setNodePolicies} ranges={systemConfig} />
                    </Box>
                  </CardBody>
                </Card>

                {/* System Configuration Ranges */}
                <Card variant="outline" bg={cardBg} borderColor={borderColor} w="100%" className="card-surface pad-6">
                  <CardHeader pb={2}>
                    <Heading size="md">System Configuration</Heading>
                    <Text color="gray.500" fontSize="sm">Define permissible ranges for configuration variables</Text>
                    <HStack mt={2}>
                      <Button size="sm" variant="outline" onClick={() => setSystemConfig(prev => ({ ...prev, ...systemRanges }))}>
                        Use System Ranges
                      </Button>
                      <Button size="sm" variant="ghost" onClick={() => navigate('/system-config')}>
                        Edit in System Config
                      </Button>
                    </HStack>
                  </CardHeader>
                  <CardBody pt={0}>
                    <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)' }} gap={6}>
                      {Object.entries(systemConfig).map(([key, rng]) => (
                        <HStack key={key} spacing={3} align="flex-end">
                          <FormControl>
                            <FormLabel textTransform="capitalize">{key.replaceAll('_',' ') } Min</FormLabel>
                            <NumberInput value={rng.min} onChange={(v)=> setSystemConfig(s=> ({...s, [key]: { ...s[key], min: parseFloat(v)||0 }}))}>
                              <NumberInputField />
                            </NumberInput>
                          </FormControl>
                          <FormControl>
                            <FormLabel textTransform="capitalize">{key.replaceAll('_',' ') } Max</FormLabel>
                            <NumberInput value={rng.max} onChange={(v)=> setSystemConfig(s=> ({...s, [key]: { ...s[key], max: parseFloat(v)||0 }}))}>
                              <NumberInputField />
                            </NumberInput>
                          </FormControl>
                        </HStack>
                      ))}
                    </Grid>
                  </CardBody>
                </Card>
              </VStack>
            </TabPanel>
            
            <TabPanel p={0}>
              <HStack justify="space-between" mb={3}>
                <Box />
                <Button size="sm" variant="outline" onClick={() => {
                  const priceMid = (k) => { const r = systemConfig[k] || {}; const min = Number(r.min ?? 0); const max = Number(r.max ?? 0) || min; return min + (max - min)/2; };
                  const p = Math.round(priceMid('price') * 100) / 100;
                  const stdMin = Number(systemConfig.standard_cost?.min ?? 0);
                  const std = Math.max(stdMin, Math.round(p * 0.8 * 100)/100);
                  setPricingConfig({
                    retailer: { selling_price: p, standard_cost: std },
                    wholesaler: { selling_price: p, standard_cost: std },
                    distributor: { selling_price: p, standard_cost: std },
                    factory: { selling_price: p, standard_cost: std },
                  });
                }}>
                  Reset Pricing to Server Defaults
                </Button>
              </HStack>
              <PricingConfigForm pricingConfig={pricingConfig} onChange={setPricingConfig} />
            </TabPanel>
            
            <TabPanel p={0}>

              {/* Player Configuration Card */}
              <Card variant="outline" bg={cardBg} borderColor={borderColor} className="card-surface pad-6">
                <CardHeader>
                  <Heading size="md">Player Configuration</Heading>
                  <Text color="gray.500" fontSize="sm">
                    Configure players and AI agents for each role
                  </Text>
                </CardHeader>
                <CardBody>
                  <VStack spacing={6} align="stretch">
              {players.map((player, index) => {
                const assignmentValue =
                  player.playerType === 'ai'
                    ? player.strategy || agentStrategies[0].options[0].value
                    : HUMAN_ASSIGNMENT;
                const assignmentLabel =
                  assignmentValue === HUMAN_ASSIGNMENT
                    ? 'Human Player'
                    : getStrategyLabel(assignmentValue);
                const isDaybreakSelection =
                  ['DAYBREAK_DTCE', 'DAYBREAK_DTCE_CENTRAL', 'DAYBREAK_DTCE_GLOBAL'].includes(assignmentValue);
                const daybreakTrainingLocked =
                  isDaybreakSelection && !(modelStatus && modelStatus.is_trained);
                const descriptionText =
                  assignmentValue === HUMAN_ASSIGNMENT
                    ? 'Assign a participant to control this role.'
                    : strategyDescriptions[assignmentValue] || 'AI agent will manage ordering for this role.';
                const helperText = daybreakTrainingLocked
                  ? `${descriptionText} Training must complete before using Daybreak agents.`
                  : descriptionText;

                return (
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
                    <VStack align="stretch" spacing={4}>
                      <HStack justify="space-between" align="flex-start">
                        <HStack spacing={3} align="center">
                          <Text fontSize="lg" fontWeight="semibold" textTransform="capitalize">
                            {player.role}
                          </Text>
                          {player.role === 'retailer' && (
                            <Badge colorScheme="blue" variant="subtle" borderRadius="full" px={2}>
                              Required
                            </Badge>
                          )}
                          <Badge
                            colorScheme={assignmentValue === HUMAN_ASSIGNMENT ? 'green' : 'purple'}
                            variant="subtle"
                            borderRadius="full"
                            px={2}
                          >
                            {assignmentLabel}
                          </Badge>
                        </HStack>
                      </HStack>

                      <FormControl>
                        <FormLabel>Assignment</FormLabel>
                        <Select
                          value={assignmentValue}
                          onChange={(e) => handleAssignmentSelect(index, e.target.value)}
                          size="md"
                          bg="white"
                          _dark={{
                            bg: 'gray.700',
                            borderColor: 'gray.600',
                            _hover: { borderColor: 'gray.500' },
                            _focus: { borderColor: 'blue.500', boxShadow: '0 0 0 1px #3182ce' }
                          }}
                        >
                          <option value={HUMAN_ASSIGNMENT}>Human Player</option>
                          {agentStrategies.map((group, groupIndex) => (
                            <optgroup key={groupIndex} label={group.group}>
                              {group.options.map((option) => (
                                <option
                                  key={option.value}
                                  value={option.value}
                                  disabled={option.requiresModel && !(modelStatus && modelStatus.is_trained)}
                                >
                                  {option.label}
                                </option>
                              ))}
                            </optgroup>
                          ))}
                        </Select>
                        <FormHelperText>{helperText}</FormHelperText>
                      </FormControl>

                      {player.playerType === 'ai' && (
                        <VStack align="stretch" spacing={3}>
                          {String(player.strategy).startsWith('LLM_') && (
                            <Box>
                              <FormLabel>Choose LLM</FormLabel>
                              <Select
                                value={player.llmModel}
                                onChange={(e) =>
                                  setPlayers((prev) =>
                                    prev.map((p, i) => (i === index ? { ...p, llmModel: e.target.value } : p))
                                  )
                                }
                              >
                                <option value="gpt-4o">GPT-4o</option>
                                <option value="gpt-4o-mini">GPT-4o Mini</option>
                                <option value="claude-3-5-sonnet">Claude 3.5 Sonnet</option>
                                <option value="claude-3-5-haiku">Claude 3.5 Haiku</option>
                              </Select>
                              <FormHelperText>Pick the LLM backend for this agent.</FormHelperText>
                            </Box>
                          )}

                          {player.strategy === 'DAYBREAK_DTCE_CENTRAL' && (
                            <Box>
                              <FormLabel>Supervisor Override (±%)</FormLabel>
                              <NumberInput
                                min={5}
                                max={50}
                                step={1}
                                value={clampOverridePercent(player.daybreakOverridePct)}
                                onChange={(valueString, valueNumber) => {
                                  const raw = Number.isFinite(valueNumber)
                                    ? valueNumber
                                    : parseFloat(valueString);
                                  const next = clampOverridePercent(
                                    Number.isFinite(raw) ? raw : player.daybreakOverridePct
                                  );
                                  setPlayers((prev) =>
                                    prev.map((p, i) =>
                                      i === index ? { ...p, daybreakOverridePct: next } : p
                                    )
                                  );
                                }}
                              >
                                <NumberInputField />
                                <NumberInputStepper>
                                  <NumberIncrementStepper />
                                  <NumberDecrementStepper />
                                </NumberInputStepper>
                              </NumberInput>
                              <FormHelperText>
                                Supervisor may adjust the Daybreak recommendation by up to this percentage.
                              </FormHelperText>
                            </Box>
                          )}
                        </VStack>
                      )}

                      {player.playerType === 'human' && (
                        <FormControl>
                          <FormLabel>Assign User</FormLabel>
                          <Select
                            placeholder="Select a user"
                            value={player.userId || ''}
                            onChange={(e) => handleUserChange(index, e.target.value || null)}
                            isDisabled={loadingUsers}
                            bg="white"
                            _dark={{
                              bg: 'gray.700',
                              borderColor: 'gray.600',
                              _hover: { borderColor: 'gray.500' },
                              _focus: { borderColor: 'blue.500', boxShadow: '0 0 0 1px #3182ce' }
                            }}
                          >
                            <option value="">-- Select User --</option>
                            {availableUsers.map((user) => (
                              <option
                                key={user.id}
                                value={user.id}
                                disabled={players.some(p => p.userId === user.id && p.role !== player.role)}
                              >
                                {user.username} {(user.is_superuser || (Array.isArray(user.roles) && user.roles.includes('admin'))) ? '(Admin)' : ''}
                                {players.some(p => p.userId === user.id && p.role !== player.role) ? ' (Assigned)' : ''}
                              </option>
                            ))}
                          </Select>
                          <FormHelperText>
                            {loadingUsers
                              ? 'Loading users...'
                              : player.userId
                                ? `Assigned to: ${availableUsers.find(u => u.id === player.userId)?.username || 'Unknown'}`
                                : 'Select a user to assign to this role'}
                          </FormHelperText>
                        </FormControl>
                      )}

                      <FormControl display="flex" alignItems="center">
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
                    </VStack>
                  </Box>
                );
              })}
                  </VStack>
                </CardBody>
              </Card>
            </TabPanel>
          </TabPanels>
        </Tabs>

        {/* Action Buttons */}
        <HStack spacing={4} justify="flex-end" mt={4}>
          <Button 
            onClick={() => navigate('/games')}
            variant="outline"
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
    </PageLayout>
  );
};

export default CreateMixedGame;
