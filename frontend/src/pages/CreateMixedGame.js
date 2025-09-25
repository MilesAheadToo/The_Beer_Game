import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useNavigate, useSearchParams, Link, useParams } from 'react-router-dom';
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
  AlertDescription,
  RadioGroup,
  Radio,
  Spinner,
  Checkbox,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Tag,
  Wrap,
  WrapItem
} from '@chakra-ui/react';
import PageLayout from '../components/PageLayout';
import { getAllConfigs, getSupplyChainConfigById } from '../services/supplyChainConfigService';
import { getUserType as resolveUserType } from '../utils/authUtils';
import PricingConfigForm from '../components/PricingConfigForm';
import { api, mixedGameApi } from '../services/api';
import { getModelStatus } from '../services/modelService';
import { useSystemConfig } from '../contexts/SystemConfigContext.jsx';
import {
  PRIMARY_FONT,
  CARD_BG_LIGHT,
  CARD_BG_DARK,
  BORDER_LIGHT,
  BORDER_DARK,
  TIMELINE_BG_LIGHT,
  TIMELINE_BG_DARK,
} from '../theme/constants';

const playerRoles = [
  { value: 'retailer', label: 'Retailer' },
  { value: 'wholesaler', label: 'Wholesaler' },
  { value: 'distributor', label: 'Distributor' },
  { value: 'manufacturer', label: 'Manufacturer' },
];

const agentStrategies = [
  {
    group: 'Basic',
    options: [
      { value: 'NAIVE', label: 'Naive (heuristic)' },
      { value: 'BULLWHIP', label: 'Bullwhip (heuristic)' },
      { value: 'CONSERVATIVE', label: 'Conservative (heuristic)' },
      { value: 'RANDOM', label: 'Random (heuristic)' },
      { value: 'PI_HEURISTIC', label: 'PI Heuristic (control)' },
    ]
  },
  {
    group: 'Daybreak LLM',
    options: [
      { value: 'LLM_CONSERVATIVE', label: 'Daybreak LLM - Conservative' },
      { value: 'LLM_BALANCED', label: 'Daybreak LLM - Balanced' },
      { value: 'LLM_AGGRESSIVE', label: 'Daybreak LLM - Aggressive' },
      { value: 'LLM_ADAPTIVE', label: 'Daybreak LLM - Adaptive' },
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

const DEFAULT_CLASSIC_PARAMS = {
  initial_demand: 4,
  change_week: 6,
  final_demand: 8,
};

const strategyDescriptions = {
  NAIVE: 'Basic heuristic that matches orders to demand.',
  BULLWHIP: 'Tends to overreact to demand changes.',
  CONSERVATIVE: 'Maintains stable inventory levels.',
  RANDOM: 'Makes random order decisions.',
  PI_HEURISTIC: 'Uses a proportional-integral controller to balance demand forecast and inventory error.',
  DEMAND_DRIVEN: 'Daybreak LLM: demand-driven analysis.',
  COST_OPTIMIZATION: 'Daybreak LLM: optimizes for lower cost.',
  LLM_CONSERVATIVE: 'Daybreak LLM strategy focused on stable inventory.',
  LLM_BALANCED: 'Daybreak LLM strategy balancing service and cost.',
  LLM_AGGRESSIVE: 'Daybreak LLM strategy that minimizes inventory aggressively.',
  LLM_ADAPTIVE: 'Daybreak LLM strategy that adapts to observed trends.',
  DAYBREAK_DTCE: daybreakStrategyDescriptions.DAYBREAK_DTCE,
  DAYBREAK_DTCE_CENTRAL: daybreakStrategyDescriptions.DAYBREAK_DTCE_CENTRAL,
  DAYBREAK_DTCE_GLOBAL: daybreakStrategyDescriptions.DAYBREAK_DTCE_GLOBAL,
};

const StyledFormLabel = (props) => (
  <FormLabel fontWeight="semibold" fontSize="md" {...props} />
);

const HelperText = ({ children, ...props }) => (
  <Text fontSize="0.75rem" color="gray.600" mt={1} {...props}>
    {children}
  </Text>
);

const progressionOptions = [
  {
    value: 'supervised',
    label: 'Supervised',
    description: 'Group Admin advances rounds manually.',
  },
  {
    value: 'unsupervised',
    label: 'Unsupervised',
    description: 'Advance automatically when every player submits.',
  },
];

const DEFAULT_SYSTEM_CONFIG = {
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
};

const DEFAULT_POLICY = {
  info_delay: 2,
  ship_delay: 2,
  init_inventory: 12,
  holding_cost: 0.5,
  backlog_cost: 1.0,
  max_inbound_per_link: 100,
  max_order: 100,
};

const DEFAULT_PRICING_CONFIG = {
  retailer: { selling_price: 100.0, standard_cost: 80.0 },
  wholesaler: { selling_price: 75.0, standard_cost: 60.0 },
  distributor: { selling_price: 60.0, standard_cost: 45.0 },
  manufacturer: { selling_price: 45.0, standard_cost: 30.0 },
};

const DEFAULT_DAYBREAK_LLM_CONFIG = {
  toggles: {
    customer_demand_history_sharing: false,
    volatility_signal_sharing: false,
    downstream_inventory_visibility: false,
  },
  shared_history_weeks: null,
  volatility_window: null,
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

const normalizeStrategyForPayload = (strategy) => {
  if (!strategy) {
    return null;
  }
  const raw = String(strategy).toLowerCase();
  if (raw === 'pi' || raw === 'pi_controller') {
    return 'pi_heuristic';
  }
  return raw;
};

const CreateMixedGame = () => {
  const { gameId } = useParams();
  const isEditing = Boolean(gameId);
  const [searchParams] = useSearchParams();
  const [gameName, setGameName] = useState(searchParams.get('name') || '');
  const [systemConfig, setSystemConfig] = useState(() => ({
    ...DEFAULT_SYSTEM_CONFIG,
  }));
  const [maxRounds, setMaxRounds] = useState(20);
  const [description, setDescription] = useState(searchParams.get('description') || '');
  const [isPublic, setIsPublic] = useState(true);
  const [progressionMode, setProgressionMode] = useState('supervised');
  const [demandPattern, setDemandPattern] = useState(demandPatterns[0].value);
  const [initialDemand, setInitialDemand] = useState(4);
  const [demandChangeWeek, setDemandChangeWeek] = useState(6);
  const [finalDemand, setFinalDemand] = useState(8);
  const [pricingConfig, setPricingConfig] = useState(() => ({ ...DEFAULT_PRICING_CONFIG }));
  const [nodePolicies, setNodePolicies] = useState({});
  // Policy/Simulation settings (bounded)
  const [policy, setPolicy] = useState(() => ({ ...DEFAULT_POLICY }));
  const [daybreakLlmConfig, setDaybreakLlmConfig] = useState(() => ({
    toggles: { ...DEFAULT_DAYBREAK_LLM_CONFIG.toggles },
    shared_history_weeks: DEFAULT_DAYBREAK_LLM_CONFIG.shared_history_weeks,
    volatility_window: DEFAULT_DAYBREAK_LLM_CONFIG.volatility_window,
  }));
  const cardBg = useColorModeValue(CARD_BG_LIGHT, CARD_BG_DARK);
  const borderColor = useColorModeValue(BORDER_LIGHT, BORDER_DARK);
  const { user } = useAuth();
  const [modelStatus, setModelStatus] = useState(null);
  const { ranges: systemRanges } = useSystemConfig();
  const [availableUsers, setAvailableUsers] = useState([]);
  const [loadingUsers, setLoadingUsers] = useState(true);

  const [players, setPlayers] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [configs, setConfigs] = useState([]);
  const [activeConfigId, setActiveConfigId] = useState(null);
  const [activeSupplyChainConfig, setActiveSupplyChainConfig] = useState(null);
  const [loadingSupplyChain, setLoadingSupplyChain] = useState(true);
  const [supplyChainError, setSupplyChainError] = useState(null);
  const [selectedNodeType, setSelectedNodeType] = useState(null);
  const navigate = useNavigate();
  const toast = useToast();
  const [initializing, setInitializing] = useState(isEditing);

  const usesDaybreakStrategist = useMemo(
    () =>
      players.some(
        (player) => player.playerType === 'ai' && String(player.strategy || '').startsWith('LLM_')
      ),
    [players]
  );

  const hasDaybreakOverrides = useMemo(() => {
    const toggles = daybreakLlmConfig?.toggles || {};
    return Object.values(toggles).some(Boolean);
  }, [daybreakLlmConfig]);

  const showDaybreakSharingCard = usesDaybreakStrategist || hasDaybreakOverrides;

  const normalizeNodeName = useCallback((value) => String(value || '').trim().toLowerCase(), []);
  const normalizeNodeType = useCallback((value) => String(value || '').trim().toLowerCase(), []);

  const playableTypeToRole = {
    retailer: 'retailer',
    wholesaler: 'wholesaler',
    distributor: 'distributor',
    manufacturer: 'manufacturer',
  };

  const nodeTypeLabels = {
    market_supply: 'Market Supply',
    manufacturer: 'Manufacturer',
    distributor: 'Distributor',
    wholesaler: 'Wholesaler',
    retailer: 'Retailer',
    market_demand: 'Market Demand',
  };

  const computeRangeMidpoint = useCallback((range, fallback = 0) => {
    if (!range || (range.min == null && range.max == null)) {
      return fallback;
    }
    const min = Number(range.min ?? range.max ?? fallback);
    const max = Number(range.max ?? range.min ?? fallback);
    if (!Number.isFinite(min) && !Number.isFinite(max)) {
      return fallback;
    }
    if (!Number.isFinite(min)) return max;
    if (!Number.isFinite(max)) return min;
    return (min + max) / 2;
  }, []);

  const clampToRange = useCallback((value, range) => {
    if (!Number.isFinite(value)) return value;
    if (!range) return value;
    let result = value;
    if (range.min != null) {
      const min = Number(range.min);
      if (Number.isFinite(min)) {
        result = Math.max(result, min);
      }
    }
    if (range.max != null) {
      const max = Number(range.max);
      if (Number.isFinite(max)) {
        result = Math.min(result, max);
      }
    }
    return result;
  }, []);

  const buildDefaultNodePolicy = useCallback(
    (node, config) => {
      const key = normalizeNodeName(node?.name);
      if (!key) {
        return null;
      }
      const nodeType = normalizeNodeType(node?.type);
      const itemConfig = node?.item_configs && node.item_configs.length > 0 ? node.item_configs[0] : null;
      const inboundLanes = (config?.lanes || []).filter((lane) => lane?.downstream_node_id === node.id);
      const leadRanges = inboundLanes.map((lane) => lane?.lead_time_days || {});
      const leadMin = leadRanges.reduce((acc, range) => {
        const value = Number(range?.min);
        if (!Number.isFinite(value)) return acc;
        if (acc == null || value < acc) return value;
        return acc;
      }, null);
      const leadMax = leadRanges.reduce((acc, range) => {
        const value = Number(range?.max);
        if (!Number.isFinite(value)) return acc;
        if (acc == null || value > acc) return value;
        return acc;
      }, null);
      const defaultInit = itemConfig?.initial_inventory_range
        ? computeRangeMidpoint(itemConfig.initial_inventory_range, 12)
        : 12;
      const shipDelay = leadMin != null && leadMax != null
        ? Math.round((leadMin + leadMax) / 2)
        : leadMin != null
          ? leadMin
          : leadMax != null
            ? leadMax
            : 0;

      const isMarketNode = ['market_supply', 'market_demand'].includes(nodeType);
      return {
        info_delay: isMarketNode ? 0 : 1,
        ship_delay: isMarketNode ? 0 : shipDelay,
        init_inventory: isMarketNode ? 0 : Math.round(defaultInit),
        min_order_qty: isMarketNode ? 0 : 0,
        variable_cost: 0,
      };
    },
    [computeRangeMidpoint, normalizeNodeName, normalizeNodeType]
  );

  const normalizeLoadedPolicies = useCallback((rawPolicies = {}) => {
    const normalised = {};
    Object.entries(rawPolicies || {}).forEach(([name, value]) => {
      if (!value) {
        return;
      }
      const key = normalizeNodeName(name);
      normalised[key] = {
        info_delay: Number.isFinite(Number(value.info_delay)) ? Number(value.info_delay) : 0,
        ship_delay: Number.isFinite(Number(value.ship_delay)) ? Number(value.ship_delay) : 0,
        init_inventory: Number.isFinite(Number(value.init_inventory)) ? Number(value.init_inventory) : 0,
        min_order_qty: Number.isFinite(Number(value.min_order_qty)) ? Number(value.min_order_qty) : 0,
        variable_cost: Number.isFinite(Number(value.variable_cost)) ? Number(value.variable_cost) : 0,
      };
    });
    return normalised;
  }, [normalizeNodeName]);

  const formatDemandPatternSummary = useCallback((pattern, params) => {
    if (!pattern) {
      return '—';
    }
    const type = String(pattern.type || params?.type || 'unknown').toLowerCase();
    const payload = params || pattern.params || {};
    switch (type) {
      case 'constant':
        return `Constant at ${payload.value ?? payload.mean ?? payload.demand ?? '?'}`;
      case 'random':
        return `Random between ${payload.min ?? '?'} and ${payload.max ?? '?'}`;
      case 'seasonal':
        return `Seasonal base ${payload.value ?? '?'} (period ${payload?.seasonality?.period ?? '?'} weeks)`;
      case 'trending':
        return `Trending base ${payload.value ?? '?'} (trend ${payload.trend ?? 0})`;
      case 'classic':
        return `Classic: starts ${payload.initial_demand ?? '?'}, week ${payload.change_week ?? '?'}, final ${payload.final_demand ?? payload.new_demand ?? '?'}`;
      default:
        return type.replace(/_/g, ' ');
    }
  }, []);

  const formatLeadTimeRange = useCallback((range) => {
    if (!range) {
      return '—';
    }
    const min = Number(range.min);
    const max = Number(range.max);
    if (Number.isFinite(min) && Number.isFinite(max)) {
      if (min === max) {
        return `${min} week${min === 1 ? '' : 's'}`;
      }
      return `${min}–${max} weeks`;
    }
    if (Number.isFinite(min)) {
      return `${min}+ weeks`;
    }
    if (Number.isFinite(max)) {
      return `≤ ${max} weeks`;
    }
    return '—';
  }, []);

  const getPolicyValue = useCallback(
    (nodeKey, field, fallback = 0) => {
      const policy = nodePolicies[nodeKey];
      if (policy && policy[field] != null) {
        return policy[field];
      }
      const defaultPolicy = buildDefaultNodePolicy(
        (activeSupplyChainConfig?.nodes || []).find((node) => normalizeNodeName(node?.name) === nodeKey),
        activeSupplyChainConfig
      );
      if (defaultPolicy && defaultPolicy[field] != null) {
        return defaultPolicy[field];
      }
      return fallback;
    },
    [activeSupplyChainConfig, buildDefaultNodePolicy, nodePolicies, normalizeNodeName]
  );

  const describeRange = useCallback((range, unit = '') => {
    if (!range) {
      return null;
    }
    const min = range.min != null ? Number(range.min) : null;
    const max = range.max != null ? Number(range.max) : null;
    if (Number.isFinite(min) && Number.isFinite(max)) {
      return `${min}${unit ? ` ${unit}` : ''} – ${max}${unit ? ` ${unit}` : ''}`;
    }
    if (Number.isFinite(min)) {
      return `≥ ${min}${unit ? ` ${unit}` : ''}`;
    }
    if (Number.isFinite(max)) {
      return `≤ ${max}${unit ? ` ${unit}` : ''}`;
    }
    return null;
  }, []);

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

  useEffect(() => {
    if (!Array.isArray(configs)) {
      return;
    }
    const groupIdRaw = user?.group_id;
    const groupId = Number.isFinite(Number(groupIdRaw)) ? Number(groupIdRaw) : null;
    const candidates = configs.filter((config) => {
      if (groupId == null) return true;
      return Number(config?.group_id) === groupId;
    });
    const chosen = candidates.find((config) => Boolean(config?.is_active)) || candidates[0] || configs[0] || null;
    setActiveConfigId(chosen ? chosen.id : null);
  }, [configs, user]);

  useEffect(() => {
    let ignore = false;
    const loadActiveConfig = async () => {
      if (!activeConfigId) {
        setActiveSupplyChainConfig(null);
        setLoadingSupplyChain(false);
        return;
      }
      try {
        setLoadingSupplyChain(true);
        setSupplyChainError(null);
        const detailed = await getSupplyChainConfigById(activeConfigId);
        if (!ignore) {
          setActiveSupplyChainConfig(detailed || null);
        }
      } catch (error) {
        if (!ignore) {
          console.error('Failed to load supply chain configuration', error);
          setSupplyChainError('Unable to load supply chain configuration.');
          setActiveSupplyChainConfig(null);
        }
      } finally {
        if (!ignore) {
          setLoadingSupplyChain(false);
        }
      }
    };
    loadActiveConfig();
    return () => {
      ignore = true;
    };
  }, [activeConfigId]);

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

  useEffect(() => {
    if (!isEditing) {
      return;
    }

    let cancelled = false;

    const loadGame = async () => {
      try {
        setInitializing(true);
        const state = await mixedGameApi.getGameState(gameId);
        if (cancelled) return;

        const game = state?.game || {};
        const config = game?.config || {};
        setGameName(game.name || '');
        setDescription(game.description || '');
        setMaxRounds(game.max_rounds || 20);
        setIsPublic(game.is_public !== undefined ? Boolean(game.is_public) : true);

        const nextProgression = state?.progression_mode || config.progression_mode || 'supervised';
        setProgressionMode(nextProgression);

        const demand = game.demand_pattern || config.demand_pattern || {};
        const demandType = String(demand.type || 'classic').toLowerCase();
        setDemandPattern(demandType);
        const demandParams = demand.params || {};
        setInitialDemand(
          Number.isFinite(Number(demandParams.initial_demand))
            ? Number(demandParams.initial_demand)
            : DEFAULT_CLASSIC_PARAMS.initial_demand
        );
        setDemandChangeWeek(
          Number.isFinite(Number(demandParams.change_week))
            ? Number(demandParams.change_week)
            : DEFAULT_CLASSIC_PARAMS.change_week
        );
        setFinalDemand(
          Number.isFinite(Number(demandParams.final_demand))
            ? Number(demandParams.final_demand)
            : DEFAULT_CLASSIC_PARAMS.final_demand
        );

        const pricing = config.pricing_config || {};
        setPricingConfig({
          retailer: {
            selling_price: Number(pricing.retailer?.selling_price ?? DEFAULT_PRICING_CONFIG.retailer.selling_price),
            standard_cost: Number(pricing.retailer?.standard_cost ?? DEFAULT_PRICING_CONFIG.retailer.standard_cost),
          },
          wholesaler: {
            selling_price: Number(pricing.wholesaler?.selling_price ?? DEFAULT_PRICING_CONFIG.wholesaler.selling_price),
            standard_cost: Number(pricing.wholesaler?.standard_cost ?? DEFAULT_PRICING_CONFIG.wholesaler.standard_cost),
          },
          distributor: {
            selling_price: Number(pricing.distributor?.selling_price ?? DEFAULT_PRICING_CONFIG.distributor.selling_price),
            standard_cost: Number(pricing.distributor?.standard_cost ?? DEFAULT_PRICING_CONFIG.distributor.standard_cost),
          },
          manufacturer: {
            selling_price: Number(pricing.manufacturer?.selling_price ?? DEFAULT_PRICING_CONFIG.manufacturer.selling_price),
            standard_cost: Number(pricing.manufacturer?.standard_cost ?? DEFAULT_PRICING_CONFIG.manufacturer.standard_cost),
          },
        });

        setNodePolicies(normalizeLoadedPolicies(config.node_policies));

        setSystemConfig({ ...DEFAULT_SYSTEM_CONFIG, ...(config.system_config || {}) });
        setPolicy({ ...DEFAULT_POLICY, ...(config.global_policy || {}) });

        const rawDaybreak = config.daybreak_llm || {};
        const toggleBlock = rawDaybreak.toggles || {};
        setDaybreakLlmConfig({
          toggles: {
            customer_demand_history_sharing: Boolean(toggleBlock.customer_demand_history_sharing),
            volatility_signal_sharing: Boolean(toggleBlock.volatility_signal_sharing),
            downstream_inventory_visibility: Boolean(toggleBlock.downstream_inventory_visibility),
          },
          shared_history_weeks:
            rawDaybreak.shared_history_weeks != null ? rawDaybreak.shared_history_weeks : null,
          volatility_window:
            rawDaybreak.volatility_window != null ? rawDaybreak.volatility_window : null,
        });

        const overrides = config.daybreak_overrides || {};
        const statePlayers = Array.isArray(state?.players) ? state.players : [];
        const mappedPlayers = statePlayers.reduce((acc, record) => {
          const roleValue = String(record.role || '').toLowerCase();
          if (!roleValue) {
            return acc;
          }
          const isAi = Boolean(record.is_ai);
          const rawStrategy = record.ai_strategy
            ? String(record.ai_strategy).toUpperCase()
            : 'NAIVE';
          const overrideDecimal =
            overrides?.[roleValue] ??
            overrides?.[roleValue.toLowerCase()] ??
            overrides?.[roleValue.toUpperCase()];
          acc[roleValue] = {
            role: roleValue,
            playerType: isAi ? 'ai' : 'human',
            strategy: isAi ? rawStrategy : 'NAIVE',
            canSeeDemand: Boolean(record.can_see_demand),
            userId: record.user_id || null,
            llmModel: record.llm_model || 'gpt-4o-mini',
            daybreakOverridePct:
              overrideDecimal != null
                ? clampOverridePercent(Number(overrideDecimal) * 100)
                : undefined,
          };
          return acc;
        }, {});
        const orderedPlayers = playerRoles.map(({ value: roleValue }) => {
          const player = mappedPlayers[roleValue];
          if (player) {
            return player;
          }
          return {
            role: roleValue,
            playerType: 'ai',
            strategy: 'NAIVE',
            canSeeDemand: roleValue === 'retailer',
            userId: roleValue === 'retailer' && user ? user.id : null,
            llmModel: 'gpt-4o-mini',
            daybreakOverridePct: undefined,
          };
        });
        setPlayers(orderedPlayers);
      } catch (error) {
        console.error('Failed to load game configuration', error);
        toast({
          title: 'Error loading game',
          description:
            error?.response?.data?.detail || 'Unable to load existing game configuration.',
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
        navigate('/games');
      } finally {
        if (!cancelled) {
          setInitializing(false);
        }
      }
    };

    loadGame();
    return () => {
      cancelled = true;
    };
  }, [isEditing, gameId, toast, navigate, user, normalizeLoadedPolicies]);

  // Optional prefill via query params for node policies (JSON-encoded)
  useEffect(() => {
    const np = searchParams.get('node_policies');
    if (np) {
      try {
        const parsed = JSON.parse(np);
        if (parsed && typeof parsed === 'object') setNodePolicies(normalizeLoadedPolicies(parsed));
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
    if (!systemRanges || !Object.keys(systemRanges).length) {
      return;
    }
    if (isEditing && !initializing) {
      return;
    }
    setSystemConfig(prev => ({ ...prev, ...systemRanges }));
  }, [systemRanges, isEditing, initializing]);


  const nodesByType = useMemo(() => {
    if (!activeSupplyChainConfig?.nodes) {
      return {};
    }
    return activeSupplyChainConfig.nodes.reduce((acc, node) => {
      const type = normalizeNodeType(node?.type);
      if (!type) {
        return acc;
      }
      if (!acc[type]) {
        acc[type] = [];
      }
      acc[type].push(node);
      return acc;
    }, {});
  }, [activeSupplyChainConfig, normalizeNodeType]);

  const nodeTypeOptions = useMemo(() => {
    const types = Object.keys(nodesByType);
    const order = ['market_supply', 'manufacturer', 'distributor', 'wholesaler', 'retailer', 'market_demand'];
    return order.filter((type) => types.includes(type));
  }, [nodesByType]);

  const playableNodes = useMemo(() => {
    const order = ['retailer', 'wholesaler', 'distributor', 'manufacturer'];
    return order
      .map((type) => {
        const nodes = nodesByType[type] || [];
        return nodes.length > 0 ? nodes[0] : null;
      })
      .filter(Boolean);
  }, [nodesByType]);

  useEffect(() => {
    if (!activeSupplyChainConfig) {
      return;
    }
    setNodePolicies((prev) => {
      const updated = { ...prev };
      (activeSupplyChainConfig.nodes || []).forEach((node) => {
        const key = normalizeNodeName(node?.name);
        if (!key) {
          return;
        }
        if (!updated[key]) {
          const defaults = buildDefaultNodePolicy(node, activeSupplyChainConfig);
          if (defaults) {
            updated[key] = defaults;
          }
        }
      });
      return updated;
    });
  }, [activeSupplyChainConfig, buildDefaultNodePolicy, normalizeNodeName]);

  useEffect(() => {
    if (!activeSupplyChainConfig) {
      return;
    }
    const availableTypes = Object.keys(nodesByType);
    if (availableTypes.length === 0) {
      setSelectedNodeType(null);
      return;
    }
    if (!selectedNodeType || !availableTypes.includes(selectedNodeType)) {
      const ordered = ['market_supply', 'manufacturer', 'distributor', 'wholesaler', 'retailer', 'market_demand'];
      const nextType = ordered.find((type) => availableTypes.includes(type)) || availableTypes[0];
      setSelectedNodeType(nextType);
    }
  }, [activeSupplyChainConfig, nodesByType, selectedNodeType]);

  useEffect(() => {
    if (!activeSupplyChainConfig) {
      return;
    }
    setPlayers((prevPlayers) => {
      const existingByRole = new Map(prevPlayers.map((player) => [player.role, player]));
      const defaults = playableNodes.map((node) => {
        const nodeType = normalizeNodeType(node?.type);
        const roleKey = playableTypeToRole[nodeType];
        if (!roleKey) {
          return null;
        }
        const base = {
          role: roleKey,
          playerType: 'ai',
          strategy: 'NAIVE',
          canSeeDemand: roleKey === 'retailer',
          userId: roleKey === 'retailer' && user ? user.id : null,
          llmModel: 'gpt-4o-mini',
          daybreakOverridePct: 5,
          displayName: node?.name || roleKey,
          nodeId: node?.id,
          nodeType,
        };
        const existing = existingByRole.get(roleKey);
        return existing ? { ...base, ...existing, displayName: node?.name || existing.displayName } : base;
      }).filter(Boolean);
      return defaults;
    });
  }, [activeSupplyChainConfig, playableNodes, normalizeNodeType, playableTypeToRole, user]);

  const nodesById = useMemo(() => {
    if (!activeSupplyChainConfig?.nodes) {
      return new Map();
    }
    return new Map(activeSupplyChainConfig.nodes.map((node) => [node.id, node]));
  }, [activeSupplyChainConfig]);

  const itemsById = useMemo(() => {
    if (!activeSupplyChainConfig?.items) {
      return new Map();
    }
    return new Map(activeSupplyChainConfig.items.map((item) => [item.id, item]));
  }, [activeSupplyChainConfig]);

  const nodePolicyBounds = useMemo(() => {
    if (!activeSupplyChainConfig) {
      return {};
    }
    const bounds = {};
    (activeSupplyChainConfig.nodes || []).forEach((node) => {
      const key = normalizeNodeName(node?.name);
      if (!key) {
        return;
      }
      const itemConfig = node?.item_configs && node.item_configs.length > 0 ? node.item_configs[0] : null;
      const inboundLanes = (activeSupplyChainConfig.lanes || []).filter((lane) => lane?.downstream_node_id === node.id);
      const leadRange = inboundLanes.reduce(
        (acc, lane) => {
          const range = lane?.lead_time_days || {};
          const min = Number(range.min);
          const max = Number(range.max);
          return {
            min: Number.isFinite(min) ? (acc.min == null ? min : Math.min(acc.min, min)) : acc.min,
            max: Number.isFinite(max) ? (acc.max == null ? max : Math.max(acc.max, max)) : acc.max,
          };
        },
        { min: null, max: null }
      );
      bounds[key] = {
        init_inventory: itemConfig?.initial_inventory_range,
        min_order_qty: itemConfig?.inventory_target_range,
        ship_delay: leadRange,
        info_delay: { min: 0, max: 12 },
        variable_cost: itemConfig?.selling_price_range
          ? { min: 0, max: Number(itemConfig.selling_price_range.max) }
          : { min: 0, max: 100 },
      };
    });
    return bounds;
  }, [activeSupplyChainConfig, normalizeNodeName]);

  const marketDemandRows = useMemo(() => {
    const demands = activeSupplyChainConfig?.market_demands || [];
    const lanes = activeSupplyChainConfig?.lanes || [];
    if (!demands.length) {
      return [];
    }
    const laneByUpstream = lanes.reduce((acc, lane) => {
      const upstream = lane?.upstream_node_id;
      const downstreamNode = nodesById.get(lane?.downstream_node_id);
      if (!upstream || !downstreamNode) {
        return acc;
      }
      const downstreamType = normalizeNodeType(downstreamNode.type);
      if (downstreamType !== 'market_demand') {
        return acc;
      }
      acc[upstream] = { lane, marketNode: downstreamNode };
      return acc;
    }, {});
    return demands.map((demand) => {
      const item = itemsById.get(demand.item_id);
      const retailer = nodesById.get(demand.retailer_id);
      const laneInfo = retailer ? laneByUpstream[retailer.id] : null;
      const pattern = demand.demand_pattern || {};
      const params = pattern.params || pattern;
      return {
        id: demand.id,
        itemName: item?.name || `Item ${demand.item_id}`,
        pattern,
        params,
        retailerName: retailer?.name || 'Unknown',
        marketNodeName: laneInfo?.marketNode?.name || 'Market Demand',
        laneLeadTime: laneInfo?.lane?.lead_time_days || null,
      };
    });
  }, [activeSupplyChainConfig, itemsById, nodesById, normalizeNodeType]);

  const marketSupplyRows = useMemo(() => {
    if (!activeSupplyChainConfig) {
      return [];
    }
    const lanes = activeSupplyChainConfig.lanes || [];
    return (nodesByType.market_supply || []).map((node) => {
      const outbound = lanes.filter((lane) => lane?.upstream_node_id === node.id);
      return {
        node,
        outbound,
      };
    });
  }, [activeSupplyChainConfig, nodesByType]);


  const handleDaybreakToggleChange = (key) => (event) => {
    const checked = event.target.checked;
    setDaybreakLlmConfig((prev) => ({
      ...prev,
      toggles: { ...prev.toggles, [key]: checked },
    }));
  };

  const handleNodePolicyNumberChange = useCallback(
    (nodeKey, field) => (valueString, valueNumber) => {
      const rawValue = Number.isFinite(valueNumber) ? valueNumber : parseFloat(valueString);
      setNodePolicies((prev) => {
        const prevPolicy = prev[nodeKey] || {};
        const bounds = nodePolicyBounds[nodeKey]?.[field];
        const defaultPolicy = prevPolicy.init_inventory != null ? prevPolicy : buildDefaultNodePolicy(
          (activeSupplyChainConfig?.nodes || []).find((node) => normalizeNodeName(node?.name) === nodeKey),
          activeSupplyChainConfig
        ) || {};
        let nextValue = Number.isFinite(rawValue) ? rawValue : prevPolicy[field] ?? defaultPolicy[field] ?? 0;
        nextValue = Number.isFinite(nextValue) ? clampToRange(nextValue, bounds) : nextValue;
        return {
          ...prev,
          [nodeKey]: {
            ...defaultPolicy,
            ...prevPolicy,
            [field]: Number.isFinite(nextValue) ? nextValue : prevPolicy[field] ?? defaultPolicy[field] ?? 0,
          },
        };
      });
    },
    [activeSupplyChainConfig, buildDefaultNodePolicy, clampToRange, nodePolicyBounds, normalizeNodeName]
  );

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
        } else if (type === 'ai') {
          updatedPlayer.userId = null;
          updatedPlayer.strategy = updatedPlayer.strategy || agentStrategies[0].options[0].value;
          if (!updatedPlayer.llmModel) {
            updatedPlayer.llmModel = 'gpt-4o-mini';
          }
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
      const nodePolicyPayload = (() => {
        if (activeSupplyChainConfig?.nodes) {
          const payload = {};
          activeSupplyChainConfig.nodes.forEach((node) => {
            const key = normalizeNodeName(node?.name);
            if (!key) {
              return;
            }
            const base = buildDefaultNodePolicy(node, activeSupplyChainConfig) || {};
            const stored = nodePolicies[key] || {};
            const merged = { ...base, ...stored };
            const roleKey = playableTypeToRole[normalizeNodeType(node?.type)];
            const pricing = roleKey ? pricingConfig[roleKey] : null;
            payload[key] = {
              info_delay: Number.isFinite(Number(merged.info_delay)) ? Number(merged.info_delay) : 0,
              ship_delay: Number.isFinite(Number(merged.ship_delay)) ? Number(merged.ship_delay) : 0,
              init_inventory: Number.isFinite(Number(merged.init_inventory)) ? Number(merged.init_inventory) : 0,
              min_order_qty: Number.isFinite(Number(merged.min_order_qty)) ? Number(merged.min_order_qty) : 0,
              variable_cost: Number.isFinite(Number(merged.variable_cost)) ? Number(merged.variable_cost) : 0,
              price: pricing ? Number(pricing.selling_price) : 0,
              standard_cost: pricing ? Number(pricing.standard_cost) : 0,
            };
          });
          return payload;
        }
        const fallback = {};
        Object.entries(nodePolicies || {}).forEach(([key, value]) => {
          fallback[key] = {
            info_delay: Number.isFinite(Number(value.info_delay)) ? Number(value.info_delay) : 0,
            ship_delay: Number.isFinite(Number(value.ship_delay)) ? Number(value.ship_delay) : 0,
            init_inventory: Number.isFinite(Number(value.init_inventory)) ? Number(value.init_inventory) : 0,
            min_order_qty: Number.isFinite(Number(value.min_order_qty)) ? Number(value.min_order_qty) : 0,
            variable_cost: Number.isFinite(Number(value.variable_cost)) ? Number(value.variable_cost) : 0,
            price: 0,
            standard_cost: 0,
          };
        });
        return fallback;
      })();

      const gameData = {
        name: gameName,
        max_rounds: maxRounds,
        description,
        is_public: isPublic,
        progression_mode: progressionMode,
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
        node_policies: nodePolicyPayload,
        system_config: systemConfig,
        global_policy: policy,
        supply_chain_config_id: activeSupplyChainConfig?.id || null,
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
          manufacturer: {
            selling_price: parseFloat(pricingConfig.manufacturer.selling_price),
            standard_cost: parseFloat(pricingConfig.manufacturer.standard_cost)
          }
        },
        player_assignments: players.map((player) => {
          const role = String(player?.role || '').toLowerCase();
          const isAi = player.playerType === 'ai';
          const normalizedStrategy = isAi ? normalizeStrategyForPayload(player.strategy) : null;
          const overridePercent =
            normalizedStrategy === 'daybreak_dtce_central'
              ? clampOverridePercent(player.daybreakOverridePct) / 100
              : null;

          return {
            role,
            player_type: isAi ? 'agent' : 'human',
            strategy: normalizedStrategy,
            can_see_demand: Boolean(player.canSeeDemand),
            user_id: isAi ? null : (player.userId || null),
            llm_model:
              isAi && String(player.strategy || '')
                .toUpperCase()
                .startsWith('LLM_')
                ? player.llmModel
                : null,
            daybreak_override_pct: overridePercent,
          };
        })
      };

      if (usesDaybreakStrategist) {
        const toggles = daybreakLlmConfig?.toggles || {};
        const daybreakPayload = {
          toggles: {
            customer_demand_history_sharing: Boolean(toggles.customer_demand_history_sharing),
            volatility_signal_sharing: Boolean(toggles.volatility_signal_sharing),
            downstream_inventory_visibility: Boolean(toggles.downstream_inventory_visibility),
          },
        };
        if (daybreakLlmConfig?.shared_history_weeks != null) {
          daybreakPayload.shared_history_weeks = daybreakLlmConfig.shared_history_weeks;
        }
        if (daybreakLlmConfig?.volatility_window != null) {
          daybreakPayload.volatility_window = daybreakLlmConfig.volatility_window;
        }
        gameData.daybreak_llm = daybreakPayload;
      }

      const response = isEditing
        ? await mixedGameApi.updateGame(gameId, gameData)
        : await mixedGameApi.createGame(gameData);
      return response;
      
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

  const handleFormSubmit = async (e) => {
    if (e) e.preventDefault();
    try {
      const response = await handleSubmit();
      if (!response) {
        return null;
      }

      toast({
        title: isEditing ? 'Game updated!' : 'Game created!',
        description: isEditing
          ? 'The game configuration has been saved.'
          : 'The mixed game has been created successfully.',
        status: 'success',
        duration: 2000,
        isClosable: true,
      });

      setTimeout(() => {
        if (isEditing) {
          navigate('/games');
        } else if (response && response.id) {
          navigate(`/games/${response.id}`);
        } else {
          navigate('/games');
        }
      }, 1500);

      return response;
    } catch (error) {
      console.error(isEditing ? 'Error updating game:' : 'Error creating game:', error);
      toast({
        title: isEditing ? 'Error updating game' : 'Error creating game',
        description:
          error?.response?.data?.detail || 'Failed to save game configuration. Please try again.',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
      throw error;
    }
  };

  if (isEditing && initializing) {
    return (
      <PageLayout title="Edit Mixed Game">
        <Box fontFamily={PRIMARY_FONT}>
          <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
            <Spinner size="xl" />
          </Box>
        </Box>
      </PageLayout>
    );
  }

  return (
    <PageLayout title={isEditing ? 'Edit Mixed Game' : 'Mixed Game Definition'}>
      <Box fontFamily={PRIMARY_FONT}>
      {/* Quick links to saved Game Configurations */}
      {!isEditing && configs?.length > 0 && (
        <Box mb={4}>
          <Alert status="info" borderRadius="md" mb={3}>
            <AlertIcon boxSize="1em" />
            <Box>
              <AlertTitle mr={2}>Saved Game Configurations</AlertTitle>
              <HelperText mt={0} mb={0}>Use a configuration to prefill this form.</HelperText>
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
          <AlertIcon boxSize="1em" color="red.500" />
          <Box>
            <AlertTitle>Daybreak Agent Not Trained</AlertTitle>
            <HelperText mt={0} mb={0}>
              The Daybreak agent has not yet been trained, so it cannot be used until training completes. You may still select Basic (heuristics) or Daybreak LLM agents.
            </HelperText>
          </Box>
        </Alert>
      )}
      <VStack as="form" onSubmit={handleFormSubmit} spacing={6} align="stretch" maxW="4xl" mx="auto">
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
                    <Heading size="md" fontFamily="inherit">Game Settings</Heading>
                    <Text color="gray.500" fontSize="sm">Configure the basic settings for your game</Text>
                  </CardHeader>
                  <CardBody pt={0}>
                    <VStack spacing={5}>
                      <FormControl>
                        <StyledFormLabel>Game Name</StyledFormLabel>
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
                        <StyledFormLabel>Maximum Rounds</StyledFormLabel>
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
                          <HelperText>Maximum 999 rounds</HelperText>
                        </FormControl>

                        <FormControl display="flex" flexDirection="column" justifyContent="flex-end">
                          <StyledFormLabel mb={0}>Game Visibility</StyledFormLabel>
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
                          <HelperText>
                            {isPublic 
                              ? 'Anyone can join this game' 
                              : 'Only invited players can join this game'}
                          </HelperText>
                        </FormControl>
                      </Grid>

                      <FormControl>
                        <StyledFormLabel>Game Orchestration</StyledFormLabel>
                        <VStack align="stretch" spacing={3}>
                          {progressionOptions.map((option) => (
                            <Box
                              key={option.value}
                              borderWidth="1px"
                              borderColor={progressionMode === option.value ? 'blue.400' : borderColor}
                              bg={progressionMode === option.value ? 'blue.50' : 'whiteAlpha.0'}
                              borderRadius="md"
                              px={4}
                              py={3}
                              cursor="pointer"
                              onClick={() => setProgressionMode(option.value)}
                              _hover={{ borderColor: 'blue.400' }}
                            >
                              <HStack align="flex-start" spacing={3}>
                                <Text fontSize="md">{progressionMode === option.value ? '☒' : '☐'}</Text>
                                <Box>
                                  <Text fontWeight="semibold">{option.label}</Text>
                                  <HelperText>{option.description}</HelperText>
                                </Box>
                              </HStack>
                            </Box>
                          ))}
                        </VStack>
                        <HelperText>Select how rounds should progress.</HelperText>
                      </FormControl>

                      {progressionMode === 'unsupervised' && (
                        <Alert status="info" variant="left-accent" borderRadius="md">
                          <AlertIcon boxSize="1em" />
                          <Box>
                            <AlertTitle fontSize="sm">Unsupervised mode</AlertTitle>
                            <HelperText mt={0} mb={0}>
                              Rounds advance automatically once all players submit their orders. Use this for self-paced games.
                            </HelperText>
                          </Box>
                        </Alert>
                      )}

                      <FormControl>
                        <StyledFormLabel>Description (Optional)</StyledFormLabel>
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
                    <Heading size="md" fontFamily="inherit">Market Demand</Heading>
                    <Text color="gray.500" fontSize="sm">Configure how customer demand evolves during the game</Text>
                  </CardHeader>
                  <CardBody pt={0}>
                    <VStack spacing={5} align="stretch">
                      <FormControl>
                        <StyledFormLabel>Demand Pattern</StyledFormLabel>
                        <Select value={demandPattern} onChange={(e) => setDemandPattern(e.target.value)}>
                          {demandPatterns.map((pattern) => (
                            <option key={pattern.value} value={pattern.value}>
                              {pattern.label}
                            </option>
                          ))}
                        </Select>
                        <HelperText>Select the demand model to use for this game</HelperText>
                      </FormControl>
                      {demandPattern === 'classic' && (
                        <Grid templateColumns={{ base: '1fr', md: 'repeat(3, 1fr)' }} gap={6}>
                          <FormControl>
                            <StyledFormLabel>Initial demand</StyledFormLabel>
                            <NumberInput min={0} value={initialDemand} onChange={(value) => setInitialDemand(parseInt(value) || 0)}>
                              <NumberInputField />
                              <NumberInputStepper>
                                <NumberIncrementStepper />
                                <NumberDecrementStepper />
                              </NumberInputStepper>
                            </NumberInput>
                            <HelperText>Customer demand before the change occurs</HelperText>
                          </FormControl>
                          <FormControl>
                            <StyledFormLabel>Change occurs in week</StyledFormLabel>
                            <NumberInput min={1} value={demandChangeWeek} onChange={(value) => setDemandChangeWeek(parseInt(value) || 1)}>
                              <NumberInputField />
                              <NumberInputStepper>
                                <NumberIncrementStepper />
                                <NumberDecrementStepper />
                              </NumberInputStepper>
                            </NumberInput>
                            <HelperText>Week when demand switches to the new level</HelperText>
                          </FormControl>
                          <FormControl>
                            <StyledFormLabel>Final demand</StyledFormLabel>
                            <NumberInput min={0} value={finalDemand} onChange={(value) => setFinalDemand(parseInt(value) || 0)}>
                              <NumberInputField />
                              <NumberInputStepper>
                                <NumberIncrementStepper />
                                <NumberDecrementStepper />
                              </NumberInputStepper>
                            </NumberInput>
                            <HelperText>Customer demand after the change</HelperText>
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

                <Card variant="outline" bg={cardBg} borderColor={borderColor} w="100%" className="card-surface pad-6">
                  <CardHeader pb={2}>
                    <Heading size="md" fontFamily="inherit">Supply Chain Network</Heading>
                    <Text color="gray.500" fontSize="sm">
                      Configure node-level parameters using the active supply chain configuration.
                    </Text>
                  </CardHeader>
                  <CardBody pt={0}>
                    {loadingSupplyChain ? (
                      <Box py={10} display="flex" justifyContent="center">
                        <Spinner size="lg" />
                      </Box>
                    ) : supplyChainError ? (
                      <Alert status="error" borderRadius="md">
                        <AlertIcon />
                        <Box>
                          <AlertTitle fontSize="sm">Unable to load supply chain</AlertTitle>
                          <AlertDescription fontSize="sm">{supplyChainError}</AlertDescription>
                        </Box>
                      </Alert>
                    ) : !activeSupplyChainConfig ? (
                      <Text color="gray.500" fontSize="sm">
                        No supply chain configuration is linked to this group yet.
                      </Text>
                    ) : (
                      <VStack align="stretch" spacing={5}>
                        <FormControl>
                          <StyledFormLabel>Node Type</StyledFormLabel>
                          {nodeTypeOptions.length > 0 ? (
                            <Wrap spacing={2}>
                              {nodeTypeOptions.map((type) => (
                                <WrapItem key={type}>
                                  <Button
                                    size="sm"
                                    variant={selectedNodeType === type ? 'solid' : 'outline'}
                                    colorScheme="blue"
                                    onClick={() => setSelectedNodeType(type)}
                                  >
                                    {nodeTypeLabels[type] || type}
                                  </Button>
                                </WrapItem>
                              ))}
                            </Wrap>
                          ) : (
                            <Text color="gray.500" fontSize="sm">No nodes available in this configuration.</Text>
                          )}
                        </FormControl>

                        {selectedNodeType ? (
                          (() => {
                            const nodes = nodesByType[selectedNodeType] || [];
                            if (selectedNodeType === 'market_demand') {
                              if (marketDemandRows.length === 0) {
                                return (
                                  <Text color="gray.500" fontSize="sm">
                                    No market demand records are defined for this configuration.
                                  </Text>
                                );
                              }
                              return (
                                <Box overflowX="auto">
                                  <Table size="sm" variant="simple">
                                    <Thead>
                                      <Tr>
                                        <Th>Item</Th>
                                        <Th>Demand Pattern</Th>
                                        <Th>Supplied By</Th>
                                        <Th>Lane Lead Time</Th>
                                      </Tr>
                                    </Thead>
                                    <Tbody>
                                      {marketDemandRows.map((row) => (
                                        <Tr key={row.id}>
                                          <Td>{row.itemName}</Td>
                                          <Td>{formatDemandPatternSummary(row.pattern, row.params)}</Td>
                                          <Td>{row.retailerName} → {row.marketNodeName}</Td>
                                          <Td>{formatLeadTimeRange(row.laneLeadTime)}</Td>
                                        </Tr>
                                      ))}
                                    </Tbody>
                                  </Table>
                                  <HelperText mt={2}>Demand patterns are defined in the linked supply chain configuration.</HelperText>
                                </Box>
                              );
                            }

                            if (selectedNodeType === 'market_supply') {
                              if (marketSupplyRows.length === 0) {
                                return (
                                  <Text color="gray.500" fontSize="sm">
                                    No market supply nodes are present in this configuration.
                                  </Text>
                                );
                              }
                              return (
                                <VStack align="stretch" spacing={4}>
                                  {marketSupplyRows.map(({ node, outbound }) => (
                                    <Box key={node.id} p={4} borderWidth="1px" borderRadius="md" borderColor={borderColor}>
                                      <Text fontWeight="semibold">{node.name}</Text>
                                      {outbound.length === 0 ? (
                                        <HelperText>No outbound lanes configured.</HelperText>
                                      ) : (
                                        <VStack align="stretch" spacing={1} mt={2}>
                                          {outbound.map((lane) => {
                                            const downstream = nodesById.get(lane.downstream_node_id);
                                            return (
                                              <Text key={lane.id} fontSize="sm">
                                                Supplies <strong>{downstream?.name || lane.downstream_node_id}</strong> • Lead time {formatLeadTimeRange(lane.lead_time_days)}
                                              </Text>
                                            );
                                          })}
                                        </VStack>
                                      )}
                                      <HelperText>Market supply nodes provide infinite supply with the configured lead times.</HelperText>
                                    </Box>
                                  ))}
                                </VStack>
                              );
                            }

                            if (nodes.length === 0) {
                              return <Text color="gray.500" fontSize="sm">No nodes of this type are defined.</Text>;
                            }

                            return (
                              <VStack align="stretch" spacing={4}>
                                {nodes.map((node) => {
                                  const nodeKey = normalizeNodeName(node?.name);
                                  const bounds = nodePolicyBounds[nodeKey] || {};
                                  return (
                                    <Box key={node.id} p={4} borderWidth="1px" borderRadius="md" borderColor={borderColor}>
                                      <Text fontWeight="semibold" mb={2}>{node.name}</Text>
                                      <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)' }} gap={4}>
                                        <FormControl>
                                          <StyledFormLabel>Information Delay (weeks)</StyledFormLabel>
                                          <NumberInput
                                            min={bounds.info_delay?.min ?? 0}
                                            max={bounds.info_delay?.max ?? 12}
                                            step={1}
                                            value={getPolicyValue(nodeKey, 'info_delay')}
                                            onChange={handleNodePolicyNumberChange(nodeKey, 'info_delay')}
                                          >
                                            <NumberInputField />
                                            <NumberInputStepper>
                                              <NumberIncrementStepper />
                                              <NumberDecrementStepper />
                                            </NumberInputStepper>
                                          </NumberInput>
                                          {describeRange(bounds.info_delay, 'weeks') && (
                                            <HelperText>Allowed: {describeRange(bounds.info_delay, 'weeks')}</HelperText>
                                          )}
                                        </FormControl>
                                        <FormControl>
                                          <StyledFormLabel>Shipment Delay (weeks)</StyledFormLabel>
                                          <NumberInput
                                            min={bounds.ship_delay?.min ?? 0}
                                            max={bounds.ship_delay?.max ?? 12}
                                            step={1}
                                            value={getPolicyValue(nodeKey, 'ship_delay')}
                                            onChange={handleNodePolicyNumberChange(nodeKey, 'ship_delay')}
                                          >
                                            <NumberInputField />
                                            <NumberInputStepper>
                                              <NumberIncrementStepper />
                                              <NumberDecrementStepper />
                                            </NumberInputStepper>
                                          </NumberInput>
                                          {describeRange(bounds.ship_delay, 'weeks') && (
                                            <HelperText>Allowed: {describeRange(bounds.ship_delay, 'weeks')}</HelperText>
                                          )}
                                        </FormControl>
                                        <FormControl>
                                          <StyledFormLabel>Initial Inventory</StyledFormLabel>
                                          <NumberInput
                                            min={bounds.init_inventory?.min ?? 0}
                                            max={bounds.init_inventory?.max ?? 9999}
                                            step={1}
                                            value={getPolicyValue(nodeKey, 'init_inventory')}
                                            onChange={handleNodePolicyNumberChange(nodeKey, 'init_inventory')}
                                          >
                                            <NumberInputField />
                                            <NumberInputStepper>
                                              <NumberIncrementStepper />
                                              <NumberDecrementStepper />
                                            </NumberInputStepper>
                                          </NumberInput>
                                          {describeRange(bounds.init_inventory, 'units') && (
                                            <HelperText>Allowed: {describeRange(bounds.init_inventory, 'units')}</HelperText>
                                          )}
                                        </FormControl>
                                        <FormControl>
                                          <StyledFormLabel>Minimum Order Quantity</StyledFormLabel>
                                          <NumberInput
                                            min={bounds.min_order_qty?.min ?? 0}
                                            max={bounds.min_order_qty?.max ?? 9999}
                                            step={1}
                                            value={getPolicyValue(nodeKey, 'min_order_qty')}
                                            onChange={handleNodePolicyNumberChange(nodeKey, 'min_order_qty')}
                                          >
                                            <NumberInputField />
                                            <NumberInputStepper>
                                              <NumberIncrementStepper />
                                              <NumberDecrementStepper />
                                            </NumberInputStepper>
                                          </NumberInput>
                                          {describeRange(bounds.min_order_qty, 'units') && (
                                            <HelperText>Allowed: {describeRange(bounds.min_order_qty, 'units')}</HelperText>
                                          )}
                                        </FormControl>
                                        <FormControl>
                                          <StyledFormLabel>Variable Cost</StyledFormLabel>
                                          <NumberInput
                                            min={bounds.variable_cost?.min ?? 0}
                                            max={bounds.variable_cost?.max ?? 1000}
                                            step={0.1}
                                            precision={2}
                                            value={getPolicyValue(nodeKey, 'variable_cost')}
                                            onChange={handleNodePolicyNumberChange(nodeKey, 'variable_cost')}
                                          >
                                            <NumberInputField />
                                            <NumberInputStepper>
                                              <NumberIncrementStepper />
                                              <NumberDecrementStepper />
                                            </NumberInputStepper>
                                          </NumberInput>
                                          {describeRange(bounds.variable_cost, 'cost') && (
                                            <HelperText>Suggested range: {describeRange(bounds.variable_cost)}</HelperText>
                                          )}
                                        </FormControl>
                                      </Grid>
                                      <HelperText mt={2}>Selling prices are configured on the Pricing tab.</HelperText>
                                    </Box>
                                  );
                                })}
                              </VStack>
                            );
                          })()
                        ) : (
                          <Text color="gray.500" fontSize="sm">Select a node type to view details.</Text>
                        )}
                      </VStack>
                    )}
                  </CardBody>
                </Card>

              {showDaybreakSharingCard && (
                <Card variant="outline" bg={cardBg} borderColor={borderColor} w="100%" className="card-surface pad-6">
                  <CardHeader pb={2}>
                    <Heading size="md">Daybreak Strategist Sharing</Heading>
                    <Text color="gray.500" fontSize="sm">
                      Choose which information the Daybreak Beer Game Strategist can see when any role uses a Daybreak LLM strategy.
                    </Text>
                  </CardHeader>
                  <CardBody pt={0}>
                    <VStack align="stretch" spacing={3}>
                      <Checkbox
                        isChecked={daybreakLlmConfig.toggles.customer_demand_history_sharing}
                        onChange={handleDaybreakToggleChange('customer_demand_history_sharing')}
                      >
                        Share retailer demand history with upstream roles
                      </Checkbox>
                      <Checkbox
                        isChecked={daybreakLlmConfig.toggles.volatility_signal_sharing}
                        onChange={handleDaybreakToggleChange('volatility_signal_sharing')}
                      >
                        Share volatility signal (variance + trend) from the retailer
                      </Checkbox>
                      <Checkbox
                        isChecked={daybreakLlmConfig.toggles.downstream_inventory_visibility}
                        onChange={handleDaybreakToggleChange('downstream_inventory_visibility')}
                      >
                        Allow visibility into the immediate downstream inventory/backlog
                      </Checkbox>
                      <Text fontSize="xs" color="gray.500">
                        Disable a toggle to keep the strategist limited to local information for that scope.
                      </Text>
                    </VStack>
                  </CardBody>
                </Card>
              )}
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
                    manufacturer: { selling_price: p, standard_cost: std },
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
                  <Heading size="md" fontFamily="inherit">Player Configuration</Heading>
                  <Text color="gray.500" fontSize="sm">
                    Configure players and AI agents for each role
                  </Text>
                </CardHeader>
                <CardBody>
                  <VStack spacing={6} align="stretch">
              {players.map((player, index) => {
                const selectedStrategy = player.strategy || agentStrategies[0].options[0].value;
                const badgeLabel =
                  player.playerType === 'human'
                    ? 'Human Player'
                    : getStrategyLabel(selectedStrategy);
                const isDaybreakSelection =
                  player.playerType === 'ai' &&
                  ['DAYBREAK_DTCE', 'DAYBREAK_DTCE_CENTRAL', 'DAYBREAK_DTCE_GLOBAL'].includes(selectedStrategy);
                const daybreakTrainingLocked =
                  isDaybreakSelection && !(modelStatus && modelStatus.is_trained);
                const humanHelper = 'Assign a participant to control this role.';
                const agentHelper =
                  strategyDescriptions[selectedStrategy] || 'AI agent will manage ordering for this role.';
                const agentHelperText = daybreakTrainingLocked
                  ? `${agentHelper} Training must complete before using Daybreak agents.`
                  : agentHelper;

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
                        <VStack align="flex-start" spacing={0}>
                          <HStack spacing={3} align="center">
                            <Text fontSize="lg" fontWeight="semibold">
                              {player.displayName || player.role}
                            </Text>
                            {player.role === 'retailer' && (
                              <Badge colorScheme="blue" variant="subtle" borderRadius="full" px={2}>
                                Required
                              </Badge>
                            )}
                          </HStack>
                          <HStack spacing={2} align="center">
                            <Tag size="sm" colorScheme="gray" variant="subtle">
                              Role: {player.role}
                            </Tag>
                            <Badge
                              colorScheme={player.playerType === 'human' ? 'green' : 'purple'}
                              variant="subtle"
                              borderRadius="full"
                              px={2}
                            >
                              {badgeLabel}
                            </Badge>
                          </HStack>
                        </VStack>
                      </HStack>

                      <FormControl>
                        <StyledFormLabel>Player Type</StyledFormLabel>
                        <HStack spacing={3}>
                          <Button
                            variant={player.playerType === 'human' ? 'solid' : 'outline'}
                            colorScheme="green"
                            onClick={() => handlePlayerTypeChange(index, 'human')}
                            size="sm"
                          >
                            Human
                          </Button>
                          <Button
                            variant={player.playerType === 'ai' ? 'solid' : 'outline'}
                            colorScheme="purple"
                            onClick={() => handlePlayerTypeChange(index, 'ai')}
                            size="sm"
                          >
                            Agent
                          </Button>
                        </HStack>
                        <HelperText>
                          {player.playerType === 'human' ? humanHelper : agentHelper}
                        </HelperText>
                      </FormControl>

                      {player.playerType === 'ai' && (
                        <FormControl>
                          <StyledFormLabel>Agent Strategy</StyledFormLabel>
                          <Select
                            value={selectedStrategy}
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
                          <HelperText>{agentHelperText}</HelperText>
                        </FormControl>
                      )}

                      {player.playerType === 'ai' && (
                        <VStack align="stretch" spacing={3}>
                          {String(player.strategy).startsWith('LLM_') && (
                            <Box>
                              <StyledFormLabel>Choose Daybreak LLM</StyledFormLabel>
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
                              <HelperText>Pick the Daybreak LLM backend for this agent.</HelperText>
                            </Box>
                          )}

                          {player.strategy === 'DAYBREAK_DTCE_CENTRAL' && (
                            <Box>
                              <StyledFormLabel>Supervisor Override (±%)</StyledFormLabel>
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
                              <HelperText>
                                Supervisor may adjust the Daybreak recommendation by up to this percentage.
                              </HelperText>
                            </Box>
                          )}
                        </VStack>
                      )}

                      {player.playerType === 'human' && (
                        <FormControl>
                          <StyledFormLabel>Assign User</StyledFormLabel>
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
                                {user.username} {resolveUserType(user) === 'systemadmin' ? '(Admin)' : ''}
                                {players.some(p => p.userId === user.id && p.role !== player.role) ? ' (Assigned)' : ''}
                              </option>
                            ))}
                          </Select>
                          <HelperText>
                            {loadingUsers
                              ? 'Loading users...'
                              : player.userId
                                ? `Assigned to: ${availableUsers.find(u => u.id === player.userId)?.username || 'Unknown'}`
                                : 'Select a user to assign to this role'}
                          </HelperText>
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
                        <StyledFormLabel htmlFor={`demand-${index}`} mb={0} opacity={player.role === 'retailer' ? 0.7 : 1}>
                          Can see customer demand
                          {player.role === 'retailer' && ' (Always enabled for Retailer)'}
                        </StyledFormLabel>
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
            loadingText={isEditing ? 'Saving...' : 'Creating...'}
          >
            {isEditing ? 'Save Changes' : 'Create Game'}
          </Button>
        </HStack>
      </VStack>
      </Box>
    </PageLayout>
  );
};

export default CreateMixedGame;
