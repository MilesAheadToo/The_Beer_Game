import React, { useState, useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { 
  Box, 
  Button,
  FormControlLabel,
  Switch,
  Chip, 
  Card, 
  CardContent, 
  CardHeader, 
  Divider, 
  Grid, 
  TextField, 
  Typography,
  CircularProgress,
  Alert,
  Stepper,
  Step,
  StepLabel,
  Paper,
  IconButton,
  Tooltip
} from '@mui/material';
import { 
  Save as SaveIcon, 
  ArrowBack as ArrowBackIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import { useSnackbar } from 'notistack';

// Import services
import { 
  getSupplyChainConfigById,
  createSupplyChainConfig,
  updateSupplyChainConfig,
  getItems,
  getNodes,
  getLanes,
  getItemNodeConfigs,
  getMarketDemands,
  createItem,
  updateItem,
  deleteItem,
  createNode,
  updateNode,
  deleteNode,
  createLane,
  updateLane,
  deleteLane,
  createItemNodeConfig,
  updateItemNodeConfig,
  createMarketDemand,
  updateMarketDemand,
  deleteMarketDemand,
  getNodeTypeDisplayName,
  getNodeTypeColor,
  DEFAULT_CONFIG,
  DEFAULT_ITEM,
  DEFAULT_NODE,
  DEFAULT_LANE,
  DEFAULT_ITEM_NODE_CONFIG,
  DEFAULT_MARKET_DEMAND
} from '../../services/supplyChainConfigService';

// Import sub-components
import RangeInput from './RangeInput';
import DemandPatternInput from './DemandPatternInput';
import NodeForm from './NodeForm';
import LaneForm from './LaneForm';
import ItemForm from './ItemForm';
import ItemNodeConfigForm from './ItemNodeConfigForm';
import MarketDemandForm from './MarketDemandForm';

const STEPS = [
  'Basic Information',
  'Items',
  'Nodes',
  'Lanes',
  'Item-Node Configurations',
  'Market Demands',
  'Review & Save'
];

const SupplyChainConfigForm = () => {
  const { id } = useParams();
  const isEditMode = Boolean(id);
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(isEditMode);
  const [error, setError] = useState(null);
  
  // Data state
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [items, setItems] = useState([]);
  const [nodes, setNodes] = useState([]);
  const [lanes, setLanes] = useState([]);
  const [itemNodeConfigs, setItemNodeConfigs] = useState([]);
  const [marketDemands, setMarketDemands] = useState([]);
  
  // Formik form
  const formik = useFormik({
    initialValues: {
      name: '',
      description: '',
      is_active: false,
    },
    validationSchema: Yup.object({
      name: Yup.string().required('Name is required'),
      description: Yup.string(),
      is_active: Yup.boolean(),
    }),
    onSubmit: async (values) => {
      try {
        setLoading(true);
        
        if (isEditMode) {
          await updateSupplyChainConfig(id, values);
          enqueueSnackbar('Configuration updated successfully', { variant: 'success' });
        } else {
          const newConfig = await createSupplyChainConfig(values);
          enqueueSnackbar('Configuration created successfully', { variant: 'success' });
          navigate(`/supply-chain-config/edit/${newConfig.id}`);
        }
        
        // Refresh data
        if (isEditMode) {
          fetchConfig();
        }
      } catch (err) {
        console.error('Error saving configuration:', err);
        enqueueSnackbar('Failed to save configuration', { variant: 'error' });
      } finally {
        setLoading(false);
      }
    },
  });

  // Fetch configuration data if in edit mode
  useEffect(() => {
    if (isEditMode) {
      fetchConfig();
    }
  }, [id]);

  const fetchConfig = async () => {
    try {
      setLoading(true);
      const configData = await getSupplyChainConfigById(id);
      setConfig(configData);
      
      // Set form values
      formik.setValues({
        name: configData.name,
        description: configData.description || '',
        is_active: configData.is_active || false,
      });
      
      // Fetch related data
      await Promise.all([
        fetchItems(),
        fetchNodes(),
        fetchLanes(),
        fetchItemNodeConfigs(),
        fetchMarketDemands(),
      ]);
      
      setError(null);
    } catch (err) {
      console.error('Error fetching configuration:', err);
      setError('Failed to load configuration data');
      enqueueSnackbar('Failed to load configuration', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const fetchItems = async () => {
    const itemsData = await getItems(id);
    setItems(itemsData);
    return itemsData;
  };

  const fetchNodes = async () => {
    const nodesData = await getNodes(id);
    setNodes(nodesData);
    return nodesData;
  };

  const fetchLanes = async () => {
    const lanesData = await getLanes(id);
    setLanes(lanesData);
    return lanesData;
  };

  const fetchItemNodeConfigs = async () => {
    const configsData = await getItemNodeConfigs(id);
    setItemNodeConfigs(configsData);
    return configsData;
  };

  const fetchMarketDemands = async () => {
    const demandsData = await getMarketDemands(id);
    setMarketDemands(demandsData);
    return demandsData;
  };

  const handleBack = () => {
    if (activeStep === 0) {
      navigate(-1);
    } else {
      setActiveStep((prevStep) => prevStep - 1);
    }
  };

  const handleNext = async () => {
    if (activeStep === STEPS.length - 1) {
      await formik.submitForm();
    } else {
      setActiveStep((prevStep) => prevStep + 1);
    }
  };

  const handleStep = (step) => {
    if (step < activeStep) {
      setActiveStep(step);
    }
  };

  // Render step content
  const renderStepContent = (step) => {
    switch (step) {
      case 0: // Basic Information
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                id="name"
                name="name"
                label="Configuration Name"
                value={formik.values.name}
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                error={formik.touched.name && Boolean(formik.errors.name)}
                helperText={formik.touched.name && formik.errors.name}
                disabled={loading}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={4}
                id="description"
                name="description"
                label="Description"
                value={formik.values.description}
                onChange={formik.handleChange}
                onBlur={formik.handleBlur}
                error={formik.touched.description && Boolean(formik.errors.description)}
                helperText={formik.touched.description && formik.errors.description}
                disabled={loading}
              />
            </Grid>
            {isEditMode && (
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={formik.values.is_active}
                      onChange={formik.handleChange}
                      name="is_active"
                      color="primary"
                      disabled={loading}
                    />
                  }
                  label="Active Configuration"
                />
              </Grid>
            )}
          </Grid>
        );
        
      case 1: // Items
        return (
          <ItemForm
            items={items}
            onAdd={handleAddItem}
            onUpdate={handleUpdateItem}
            onDelete={handleDeleteItem}
            loading={loading}
          />
        );
        
      case 2: // Nodes
        return (
          <NodeForm
            nodes={nodes}
            onAdd={handleAddNode}
            onUpdate={handleUpdateNode}
            onDelete={handleDeleteNode}
            loading={loading}
          />
        );
        
      case 3: // Lanes
        return (
          <LaneForm
            lanes={lanes}
            nodes={nodes}
            onAdd={handleAddLane}
            onUpdate={handleUpdateLane}
            onDelete={handleDeleteLane}
            loading={loading}
          />
        );
        
      case 4: // Item-Node Configs
        return (
          <ItemNodeConfigForm
            configs={itemNodeConfigs}
            items={items}
            nodes={nodes}
            onAdd={handleAddItemNodeConfig}
            onUpdate={handleUpdateItemNodeConfig}
            loading={loading}
          />
        );
        
      case 5: // Market Demands
        return (
          <MarketDemandForm
            demands={marketDemands}
            items={items}
            retailers={nodes.filter(node => node.type === 'retailer')}
            onAdd={handleAddMarketDemand}
            onUpdate={handleUpdateMarketDemand}
            onDelete={handleDeleteMarketDemand}
            loading={loading}
          />
        );
        
      case 6: // Review
        return (
          <Box>
            <Typography variant="h6" gutterBottom>Review Configuration</Typography>
            <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom>Basic Information</Typography>
              <Typography>Name: {formik.values.name}</Typography>
              <Typography>Description: {formik.values.description || 'N/A'}</Typography>
              <Typography>Status: {formik.values.is_active ? 'Active' : 'Inactive'}</Typography>
            </Paper>
            
            <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom>Items ({items.length})</Typography>
              {items.length > 0 ? (
                <ul>
                  {items.map(item => (
                    <li key={item.id}>{item.name}</li>
                  ))}
                </ul>
              ) : (
                <Typography>No items configured</Typography>
              )}
            </Paper>
            
            <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom>Nodes ({nodes.length})</Typography>
              {nodes.length > 0 ? (
                <ul>
                  {nodes.map(node => (
                    <li key={node.id}>
                      {node.name} <Chip label={getNodeTypeDisplayName(node.type)} size="small" />
                    </li>
                  ))}
                </ul>
              ) : (
                <Typography>No nodes configured</Typography>
              )}
            </Paper>
            
            <Button 
              variant="contained" 
              color="primary" 
              onClick={handleSubmit}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : <SaveIcon />}
            >
              {isEditMode ? 'Update Configuration' : 'Create Configuration'}
            </Button>
          </Box>
        );
        
      default:
        return <div>Unknown step</div>;
    }
  };

  // Handler functions for CRUD operations
  const handleAddItem = async (itemData) => {
    try {
      setLoading(true);
      await createItem(id, itemData);
      await fetchItems();
      enqueueSnackbar('Item added successfully', { variant: 'success' });
    } catch (err) {
      console.error('Error adding item:', err);
      enqueueSnackbar('Failed to add item', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateItem = async (itemId, itemData) => {
    try {
      setLoading(true);
      await updateItem(id, itemId, itemData);
      await fetchItems();
      enqueueSnackbar('Item updated successfully', { variant: 'success' });
    } catch (err) {
      console.error('Error updating item:', err);
      enqueueSnackbar('Failed to update item', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteItem = async (itemId) => {
    try {
      setLoading(true);
      await deleteItem(id, itemId);
      await fetchItems();
      enqueueSnackbar('Item deleted successfully', { variant: 'success' });
    } catch (err) {
      console.error('Error deleting item:', err);
      enqueueSnackbar('Failed to delete item', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleAddNode = async (nodeData) => {
    try {
      setLoading(true);
      await createNode(id, nodeData);
      await fetchNodes();
      enqueueSnackbar('Node added successfully', { variant: 'success' });
    } catch (err) {
      console.error('Error adding node:', err);
      enqueueSnackbar('Failed to add node', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateNode = async (nodeId, nodeData) => {
    try {
      setLoading(true);
      await updateNode(id, nodeId, nodeData);
      await fetchNodes();
      enqueueSnackbar('Node updated successfully', { variant: 'success' });
    } catch (err) {
      console.error('Error updating node:', err);
      enqueueSnackbar('Failed to update node', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteNode = async (nodeId) => {
    try {
      setLoading(true);
      await deleteNode(id, nodeId);
      await fetchNodes();
      enqueueSnackbar('Node deleted successfully', { variant: 'success' });
    } catch (err) {
      console.error('Error deleting node:', err);
      enqueueSnackbar('Failed to delete node', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleAddLane = async (laneData) => {
    try {
      setLoading(true);
      await createLane(id, laneData);
      await fetchLanes();
      enqueueSnackbar('Lane added successfully', { variant: 'success' });
    } catch (err) {
      console.error('Error adding lane:', err);
      enqueueSnackbar('Failed to add lane', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateLane = async (laneId, laneData) => {
    try {
      setLoading(true);
      await updateLane(id, laneId, laneData);
      await fetchLanes();
      enqueueSnackbar('Lane updated successfully', { variant: 'success' });
    } catch (err) {
      console.error('Error updating lane:', err);
      enqueueSnackbar('Failed to update lane', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteLane = async (laneId) => {
    try {
      setLoading(true);
      await deleteLane(id, laneId);
      await fetchLanes();
      enqueueSnackbar('Lane deleted successfully', { variant: 'success' });
    } catch (err) {
      console.error('Error deleting lane:', err);
      enqueueSnackbar('Failed to delete lane', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleAddItemNodeConfig = async (configData) => {
    try {
      setLoading(true);
      await createItemNodeConfig(id, configData);
      await fetchItemNodeConfigs();
      enqueueSnackbar('Item-Node configuration added successfully', { variant: 'success' });
    } catch (err) {
      console.error('Error adding item-node config:', err);
      enqueueSnackbar('Failed to add item-node configuration', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateItemNodeConfig = async (configId, configData) => {
    try {
      setLoading(true);
      await updateItemNodeConfig(id, configId, configData);
      await fetchItemNodeConfigs();
      enqueueSnackbar('Item-Node configuration updated successfully', { variant: 'success' });
    } catch (err) {
      console.error('Error updating item-node config:', err);
      enqueueSnackbar('Failed to update item-node configuration', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleAddMarketDemand = async (demandData) => {
    try {
      setLoading(true);
      await createMarketDemand(id, demandData);
      await fetchMarketDemands();
      enqueueSnackbar('Market demand added successfully', { variant: 'success' });
    } catch (err) {
      console.error('Error adding market demand:', err);
      enqueueSnackbar('Failed to add market demand', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleUpdateMarketDemand = async (demandId, demandData) => {
    try {
      setLoading(true);
      await updateMarketDemand(id, demandId, demandData);
      await fetchMarketDemands();
      enqueueSnackbar('Market demand updated successfully', { variant: 'success' });
    } catch (err) {
      console.error('Error updating market demand:', err);
      enqueueSnackbar('Failed to update market demand', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteMarketDemand = async (demandId) => {
    try {
      setLoading(true);
      await deleteMarketDemand(id, demandId);
      await fetchMarketDemands();
      enqueueSnackbar('Market demand deleted successfully', { variant: 'success' });
    } catch (err) {
      console.error('Error deleting market demand:', err);
      enqueueSnackbar('Failed to delete market demand', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = () => {
    formik.handleSubmit();
  };

  if (loading && !isEditMode) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="300px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 3 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Box display="flex" alignItems="center" mb={3}>
        <Tooltip title="Go back">
          <IconButton onClick={() => navigate(-1)} sx={{ mr: 2 }}>
            <ArrowBackIcon />
          </IconButton>
        </Tooltip>
        <Typography variant="h5" component="h1">
          {isEditMode ? 'Edit Configuration' : 'New Configuration'}
        </Typography>
      </Box>
      
      <Stepper activeStep={activeStep} alternativeLabel sx={{ mb: 4 }}>
        {STEPS.map((label, index) => (
          <Step key={label} onClick={() => handleStep(index)} sx={{ cursor: 'pointer' }}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>
      
      <Card>
        <CardContent>
          {renderStepContent(activeStep)}
        </CardContent>
      </Card>
      
      <Box display="flex" justifyContent="space-between" mt={3}>
        <Button
          onClick={handleBack}
          disabled={loading}
          startIcon={<ArrowBackIcon />}
        >
          {activeStep === 0 ? 'Back to List' : 'Back'}
        </Button>
        
        {activeStep < STEPS.length - 1 && (
          <Button
            variant="contained"
            color="primary"
            onClick={handleNext}
            disabled={loading}
            endIcon={loading ? <CircularProgress size={24} /> : null}
          >
            Next
          </Button>
        )}
      </Box>
    </Box>
  );
};

export default SupplyChainConfigForm;
