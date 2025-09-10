import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Divider,
  Grid,
  TextField,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
  Chip,
  FormHelperText,
  TablePagination,
  Tabs,
  Tab,
  Alert,
  AlertTitle
} from '@mui/material';
import { 
  Add as AddIcon, 
  Edit as EditIcon, 
  Delete as DeleteIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Warning as WarningIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import DemandPatternInput from './DemandPatternInput';

const MarketDemandForm = ({ 
  demands = [], 
  items = [],
  nodes = [],
  onAdd, 
  onUpdate, 
  onDelete, 
  loading = false 
}) => {
  const [openDialog, setOpenDialog] = useState(false);
  const [editingDemand, setEditingDemand] = useState(null);
  const [formData, setFormData] = useState({
    item_id: '',
    node_id: '',
    pattern: {
      type: 'constant',
      value: 10,
      min: 5,
      max: 15,
      seasonality: 1.0,
      trend: 0
    }
  });
  const [errors, setErrors] = useState({});
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [activeTab, setActiveTab] = useState(0);

  // Filter to only show retailer nodes that have is_retailer=true in their item-node config
  const retailerNodes = nodes.filter(node => 
    demands.some(d => d.node_id === node.id && d.is_retailer)
  );

  // Reset form when dialog opens/closes
  useEffect(() => {
    if (openDialog) {
      if (editingDemand) {
        setFormData({
          item_id: editingDemand.item_id,
          node_id: editingDemand.node_id,
          pattern: editingDemand.pattern || {
            type: 'constant',
            value: 10
          }
        });
      } else {
        setFormData({
          item_id: '',
          node_id: '',
          pattern: {
            type: 'constant',
            value: 10,
            min: 5,
            max: 15,
            seasonality: 1.0,
            trend: 0
          }
        });
      }
      setErrors({});
    }
  }, [openDialog, editingDemand]);

  // Filter available items and nodes based on selected values
  const [availableItems, setAvailableItems] = useState([]);
  const [availableNodes, setAvailableNodes] = useState([]);

  useEffect(() => {
    // Get items that have at least one retailer node configured
    const itemsWithRetailers = [...new Set(
      demands
        .filter(d => d.is_retailer)
        .map(d => d.item_id)
    )];
    
    setAvailableItems(
      items.filter(item => itemsWithRetailers.includes(item.id))
    );
  }, [items, demands]);

  useEffect(() => {
    if (formData.item_id) {
      // Get retailer nodes for the selected item
      const itemDemands = demands.filter(d => 
        d.item_id === formData.item_id && d.is_retailer
      );
      
      const itemNodeIds = itemDemands.map(d => d.node_id);
      const available = nodes.filter(node => 
        itemNodeIds.includes(node.id) && 
        (!formData.node_id || node.id === formData.node_id || 
         !demands.some(d => d.node_id === node.id && d.item_id === formData.item_id))
      );
      
      setAvailableNodes(available);
      
      // Reset node_id if current selection is no longer valid
      if (formData.node_id && !available.some(n => n.id === formData.node_id)) {
        setFormData(prev => ({ ...prev, node_id: '' }));
      }
    } else {
      setAvailableNodes([]);
    }
  }, [formData.item_id, formData.node_id, nodes, demands]);

  const handleOpenDialog = (demand = null) => {
    setEditingDemand(demand);
    setActiveTab(0);
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setEditingDemand(null);
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.item_id) {
      newErrors.item_id = 'Item is required';
    }
    
    if (!formData.node_id) {
      newErrors.node_id = 'Retailer node is required';
    }
    
    // Validate pattern based on type
    if (!formData.pattern) {
      newErrors.pattern = 'Demand pattern is required';
    } else {
      const { type, value, min, max, seasonality, trend } = formData.pattern;
      
      if (type === 'constant' && (value === undefined || value === '' || value < 0)) {
        newErrors.pattern = 'Value must be a non-negative number';
      } else if (type === 'random' && (min === undefined || max === undefined || min < 0 || max < min)) {
        newErrors.pattern = 'Invalid range: min must be less than or equal to max, and both must be non-negative';
      } else if (type === 'seasonal' && (value === undefined || seasonality === undefined || value < 0 || seasonality < 0 || seasonality > 2)) {
        newErrors.pattern = 'Base value must be non-negative and seasonality must be between 0 and 2';
      } else if (type === 'trending' && (value === undefined || value < 0)) {
        newErrors.pattern = 'Base value must be non-negative';
      }
    }
    
    // Check for duplicate demand configuration
    const isDuplicate = demands.some(
      d => 
        d.item_id === formData.item_id && 
        d.node_id === formData.node_id &&
        (!editingDemand || d.id !== editingDemand.id)
    );
    
    if (isDuplicate) {
      newErrors.duplicate = 'A demand configuration for this item and node already exists';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    const demandData = {
      item_id: formData.item_id,
      node_id: formData.node_id,
      pattern: { ...formData.pattern }
    };
    
    // Ensure pattern only contains relevant fields for its type
    const { type } = demandData.pattern;
    const cleanPattern = { type };
    
    if (type === 'constant') {
      cleanPattern.value = parseFloat(demandData.pattern.value) || 0;
    } else if (type === 'random') {
      cleanPattern.min = parseFloat(demandData.pattern.min) || 0;
      cleanPattern.max = Math.max(parseFloat(demandData.pattern.max) || 0, cleanPattern.min);
    } else if (type === 'seasonal') {
      cleanPattern.value = parseFloat(demandData.pattern.value) || 0;
      cleanPattern.seasonality = Math.min(Math.max(parseFloat(demandData.pattern.seasonality) || 1.0, 0), 2);
    } else if (type === 'trending') {
      cleanPattern.value = parseFloat(demandData.pattern.value) || 0;
      cleanPattern.trend = parseFloat(demandData.pattern.trend) || 0;
    }
    
    demandData.pattern = cleanPattern;
    
    if (editingDemand) {
      onUpdate(editingDemand.id, demandData);
    } else {
      onAdd(demandData);
    }
    
    handleCloseDialog();
  };

  const handleDelete = (demandId) => {
    if (window.confirm('Are you sure you want to delete this demand configuration? This action cannot be undone.')) {
      onDelete(demandId);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handlePatternChange = (pattern) => {
    setFormData(prev => ({
      ...prev,
      pattern: { ...pattern }
    }));
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const getItemName = (itemId) => {
    const item = items.find(i => i.id === itemId);
    return item ? item.name : 'Unknown';
  };

  const getNodeName = (nodeId) => {
    const node = nodes.find(n => n.id === nodeId);
    return node ? node.name : 'Unknown';
  };

  const NODE_TYPE_LABELS = {
    retailer: 'Retailer',
    distributor: 'Distributor',
    manufacturer: 'Manufacturer',
    supplier: 'Supplier'
  };

  const NODE_TYPE_COLORS = {
    retailer: 'success',
    distributor: 'info',
    manufacturer: 'warning',
    supplier: 'error'
  };

  // Filter demands based on active tab
  const filteredDemands = activeTab === 0 
    ? demands
    : demands.filter(d => d.item_id === availableItems[activeTab - 1]?.id);

  // Pagination
  const startIndex = page * rowsPerPage;
  const endIndex = startIndex + rowsPerPage;
  const paginatedDemands = filteredDemands.slice(startIndex, endIndex);

  // Check for items without demand configurations
  const itemsWithoutDemands = availableItems.filter(item => 
    !demands.some(d => d.item_id === item.id)
  );

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">Market Demand Configurations</Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={() => handleOpenDialog()}
          disabled={loading || availableItems.length === 0}
        >
          Add Demand
        </Button>
      </Box>

      {availableItems.length === 0 ? (
        <Alert severity="info" sx={{ mb: 2 }}>
          <AlertTitle>No Retailer Nodes Configured</AlertTitle>
          Configure at least one retailer node in the Item-Node Configurations section before setting up market demand.
        </Alert>
      ) : (
        <>
          <Tabs 
            value={activeTab} 
            onChange={handleTabChange} 
            variant="scrollable"
            scrollButtons="auto"
            sx={{ mb: 2 }}
          >
            <Tab label="All Items" />
            {availableItems.map(item => (
              <Tab 
                key={item.id} 
                label={
                  <Box display="flex" alignItems="center">
                    {item.name}
                    {!demands.some(d => d.item_id === item.id) && (
                      <WarningIcon color="warning" fontSize="small" sx={{ ml: 1 }} />
                    )}
                  </Box>
                } 
              />
            ))}
          </Tabs>

          {itemsWithoutDemands.length > 0 && activeTab === 0 && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              <AlertTitle>Missing Demand Configurations</AlertTitle>
              The following items don't have demand configurations: {itemsWithoutDemands.map(i => i.name).join(', ')}.
              Add demand configurations for these items to ensure proper simulation behavior.
            </Alert>
          )}

          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Item</TableCell>
                  <TableCell>Retailer Node</TableCell>
                  <TableCell>Demand Pattern</TableCell>
                  <TableCell>Parameters</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {paginatedDemands.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} align="center">
                      {activeTab === 0 
                        ? 'No demand configurations found. Click "Add Demand" to get started.'
                        : `No demand configurations found for ${getItemName(availableItems[activeTab - 1]?.id)}.`}
                    </TableCell>
                  </TableRow>
                ) : (
                  paginatedDemands.map((demand) => (
                    <TableRow key={`${demand.item_id}-${demand.node_id}`}>
                      <TableCell>{getItemName(demand.item_id)}</TableCell>
                      <TableCell>
                        <Box display="flex" alignItems="center">
                          <Chip 
                            label={NODE_TYPE_LABELS.retailer}
                            size="small"
                            color={NODE_TYPE_COLORS.retailer}
                            sx={{ mr: 1 }}
                          />
                          {getNodeName(demand.node_id)}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Box textTransform="capitalize">
                          {demand.pattern?.type || 'Not specified'}
                        </Box>
                      </TableCell>
                      <TableCell>
                        {demand.pattern && (
                          <Box component="span" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                            {JSON.stringify(
                              Object.entries(demand.pattern)
                                .filter(([key]) => key !== 'type')
                                .reduce((obj, [key, val]) => ({ ...obj, [key]: val }), {}),
                              null,
                              2
                            ).replace(/[{"}]/g, '').trim()}
                          </Box>
                        )}
                      </TableCell>
                      <TableCell align="right">
                        <Tooltip title="Edit">
                          <IconButton 
                            size="small" 
                            onClick={() => handleOpenDialog(demand)}
                            disabled={loading}
                          >
                            <EditIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton 
                            size="small" 
                            color="error"
                            onClick={() => handleDelete(demand.id)}
                            disabled={loading}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
            <TablePagination
              rowsPerPageOptions={[5, 10, 25]}
              component="div"
              count={filteredDemands.length}
              rowsPerPage={rowsPerPage}
              page={page}
              onPageChange={handleChangePage}
              onRowsPerPageChange={handleChangeRowsPerPage}
            />
          </TableContainer>
        </>
      )}

      <Dialog 
        open={openDialog} 
        onClose={handleCloseDialog} 
        maxWidth="sm" 
        fullWidth
        TransitionProps={{ onEntered: () => setActiveTab(0) }}
      >
        <form onSubmit={handleSubmit}>
          <DialogTitle>
            {editingDemand ? 'Edit Demand Configuration' : 'Add New Demand Configuration'}
          </DialogTitle>
          <DialogContent>
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={12}>
                <FormControl fullWidth error={!!errors.item_id} size="small">
                  <InputLabel id="demand-item-label">Item</InputLabel>
                  <Select
                    labelId="demand-item-label"
                    name="item_id"
                    value={formData.item_id}
                    label="Item"
                    onChange={handleChange}
                    disabled={loading || !!editingDemand}
                  >
                    {availableItems.map((item) => (
                      <MenuItem key={item.id} value={item.id}>
                        {item.name}
                      </MenuItem>
                    ))}
                  </Select>
                  {errors.item_id && (
                    <FormHelperText>{errors.item_id}</FormHelperText>
                  )}
                </FormControl>
              </Grid>
              
              <Grid item xs={12}>
                <FormControl fullWidth error={!!errors.node_id} size="small">
                  <InputLabel id="demand-node-label">Retailer Node</InputLabel>
                  <Select
                    labelId="demand-node-label"
                    name="node_id"
                    value={formData.node_id}
                    label="Retailer Node"
                    onChange={handleChange}
                    disabled={loading || !formData.item_id || !!editingDemand}
                  >
                    {availableNodes.map((node) => (
                      <MenuItem key={node.id} value={node.id}>
                        <Box display="flex" alignItems="center">
                          <Chip 
                            label={NODE_TYPE_LABELS.retailer}
                            size="small"
                            color={NODE_TYPE_COLORS.retailer}
                            sx={{ mr: 1 }}
                          />
                          {node.name}
                        </Box>
                      </MenuItem>
                    ))}
                  </Select>
                  {errors.node_id ? (
                    <FormHelperText>{errors.node_id}</FormHelperText>
                  ) : !formData.item_id ? (
                    <FormHelperText>Select an item first</FormHelperText>
                  ) : availableNodes.length === 0 ? (
                    <FormHelperText>No available retailer nodes for the selected item</FormHelperText>
                  ) : null}
                </FormControl>
              </Grid>
              
              <Grid item xs={12}>
                <DemandPatternInput
                  value={formData.pattern}
                  onChange={handlePatternChange}
                  disabled={loading}
                />
                {errors.pattern && (
                  <Typography color="error" variant="body2" sx={{ mt: 1 }}>
                    {errors.pattern}
                  </Typography>
                )}
              </Grid>
              
              {errors.duplicate && (
                <Grid item xs={12}>
                  <Alert severity="error">
                    <AlertTitle>Duplicate Configuration</AlertTitle>
                    {errors.duplicate}
                  </Alert>
                </Grid>
              )}
              
              <Grid item xs={12}>
                <Alert severity="info" icon={<InfoIcon />}>
                  <AlertTitle>Demand Pattern Information</AlertTitle>
                  <ul style={{ margin: 0, paddingLeft: '16px' }}>
                    <li><strong>Constant:</strong> Fixed demand per period</li>
                    <li><strong>Random:</strong> Uniform random demand between min and max</li>
                    <li><strong>Seasonal:</strong> Varies with a seasonal pattern</li>
                    <li><strong>Trending:</strong> Increases or decreases over time</li>
                  </ul>
                </Alert>
              </Grid>
            </Grid>
          </DialogContent>
          <DialogActions>
            <Button 
              onClick={handleCloseDialog}
              disabled={loading}
              startIcon={<CancelIcon />}
            >
              Cancel
            </Button>
            <Button 
              type="submit" 
              color="primary" 
              variant="contained"
              disabled={loading || !formData.item_id || !formData.node_id || !formData.pattern}
              startIcon={editingDemand ? <SaveIcon /> : <AddIcon />}
            >
              {editingDemand ? 'Update' : 'Add'} Demand
            </Button>
          </DialogActions>
        </form>
      </Dialog>
    </Box>
  );
};

export default MarketDemandForm;
