import api from './api';

const SUPPLY_CHAIN_CONFIG_BASE_URL = '/api/v1/supply-chain-config';

// Supply Chain Config CRUD
export const getSupplyChainConfigs = async () => {
  const response = await api.get(SUPPLY_CHAIN_CONFIG_BASE_URL);
  return response.data;
};

export const getSupplyChainConfigById = async (id) => {
  const response = await api.get(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${id}`);
  return response.data;
};

export const createSupplyChainConfig = async (configData) => {
  const response = await api.post(SUPPLY_CHAIN_CONFIG_BASE_URL, configData);
  return response.data;
};

export const updateSupplyChainConfig = async (id, configData) => {
  const response = await api.put(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${id}`, configData);
  return response.data;
};

export const deleteSupplyChainConfig = async (id) => {
  await api.delete(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${id}`);
};

// Items CRUD
export const getItems = async (configId) => {
  const response = await api.get(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/items`);
  return response.data;
};

export const createItem = async (configId, itemData) => {
  const response = await api.post(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/items`, itemData);
  return response.data;
};

export const updateItem = async (configId, itemId, itemData) => {
  const response = await api.put(
    `${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/items/${itemId}`, 
    itemData
  );
  return response.data;
};

export const deleteItem = async (configId, itemId) => {
  await api.delete(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/items/${itemId}`);
};

// Nodes CRUD
export const getNodes = async (configId, nodeType = null) => {
  const url = nodeType 
    ? `${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/nodes?node_type=${nodeType}`
    : `${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/nodes`;
  
  const response = await api.get(url);
  return response.data;
};

export const createNode = async (configId, nodeData) => {
  const response = await api.post(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/nodes`, nodeData);
  return response.data;
};

export const updateNode = async (configId, nodeId, nodeData) => {
  const response = await api.put(
    `${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/nodes/${nodeId}`, 
    nodeData
  );
  return response.data;
};

export const deleteNode = async (configId, nodeId) => {
  await api.delete(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/nodes/${nodeId}`);
};

// Lanes CRUD
export const getLanes = async (configId) => {
  const response = await api.get(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/lanes`);
  return response.data;
};

export const createLane = async (configId, laneData) => {
  const response = await api.post(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/lanes`, laneData);
  return response.data;
};

export const updateLane = async (configId, laneId, laneData) => {
  const response = await api.put(
    `${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/lanes/${laneId}`, 
    laneData
  );
  return response.data;
};

export const deleteLane = async (configId, laneId) => {
  await api.delete(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/lanes/${laneId}`);
};

// Item-Node Configs CRUD
export const getItemNodeConfigs = async (configId) => {
  const response = await api.get(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/item-node-configs`);
  return response.data;
};

export const createItemNodeConfig = async (configId, itemNodeData) => {
  const response = await api.post(
    `${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/item-node-configs`, 
    itemNodeData
  );
  return response.data;
};

export const updateItemNodeConfig = async (configId, itemNodeId, itemNodeData) => {
  const response = await api.put(
    `${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/item-node-configs/${itemNodeId}`, 
    itemNodeData
  );
  return response.data;
};

// Market Demands CRUD
export const getMarketDemands = async (configId) => {
  const response = await api.get(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/market-demands`);
  return response.data;
};

export const createMarketDemand = async (configId, demandData) => {
  const response = await api.post(
    `${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/market-demands`, 
    demandData
  );
  return response.data;
};

export const updateMarketDemand = async (configId, demandId, demandData) => {
  const response = await api.put(
    `${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/market-demands/${demandId}`, 
    demandData
  );
  return response.data;
};

export const deleteMarketDemand = async (configId, demandId) => {
  await api.delete(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/market-demands/${demandId}`);
};

// Helper functions
export const getNodeTypeDisplayName = (nodeType) => {
  const displayNames = {
    retailer: 'Retailer',
    distributor: 'Distributor',
    manufacturer: 'Manufacturer',
    supplier: 'Supplier',
  };
  return displayNames[nodeType] || nodeType;
};

export const getNodeTypeColor = (nodeType) => {
  const colors = {
    retailer: 'success',
    distributor: 'info',
    manufacturer: 'warning',
    supplier: 'error',
  };
  return colors[nodeType] || 'default';
};

// Game creation from config
export const createGameFromConfig = async (configId, gameData) => {
  const response = await api.post(`${SUPPLY_CHAIN_CONFIG_BASE_URL}/${configId}/create-game`, gameData);
  return response.data;
};

// Get all configurations with minimal data
export const getAllConfigs = async () => {
  const response = await api.get(SUPPLY_CHAIN_CONFIG_BASE_URL);
  return response.data;
};

// Default values for new entities
export const DEFAULT_CONFIG = {
  name: '',
  description: '',
  is_active: false,
};

export const DEFAULT_ITEM = {
  name: '',
  description: '',
  unit_cost_range: { min: 0, max: 100 },
};

export const DEFAULT_NODE = {
  name: '',
  type: 'retailer',
};

export const DEFAULT_LANE = {
  upstream_node_id: null,
  downstream_node_id: null,
  capacity: 100,
  lead_time_days: { min: 1, max: 3 },
};

export const DEFAULT_ITEM_NODE_CONFIG = {
  item_id: null,
  node_id: null,
  inventory_target_range: { min: 10, max: 50 },
  initial_inventory_range: { min: 20, max: 40 },
  holding_cost_range: { min: 0.5, max: 2 },
  backlog_cost_range: { min: 1, max: 5 },
  selling_price_range: { min: 5, max: 50 },
};

export const DEFAULT_MARKET_DEMAND = {
  item_id: null,
  retailer_id: null,
  demand_pattern: {
    type: 'constant', // 'constant', 'random', 'seasonal', 'trending'
    value: 10,
    min: 5,
    max: 15,
    seasonality: 1.0,
    trend: 0,
  },
};
