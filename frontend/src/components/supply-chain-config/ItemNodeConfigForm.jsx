import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
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
  Checkbox,
  TablePagination
} from '@mui/material';
import { 
  Add as AddIcon, 
  Edit as EditIcon, 
  Delete as DeleteIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Warning as WarningIcon
} from '@mui/icons-material';

const ItemNodeConfigForm = ({ 
  configs = [], 
  items = [],
  nodes = [],
  onAdd, 
  onUpdate, 
  onDelete, 
  loading = false 
}) => {
  const [openDialog, setOpenDialog] = useState(false);
  const [editingConfig, setEditingConfig] = useState(null);
  const [formData, setFormData] = useState({
    item_id: '',
    node_id: '',
    initial_inventory: 0,
    holding_cost: 0.1,
    backorder_cost: 1.0,
    service_level: 0.95,
    is_retailer: false
  });
  const [errors, setErrors] = useState({});
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  // Filter nodes based on selected item type and existing configs
  const [filteredNodes, setFilteredNodes] = useState([]);
  
  // Reset form when dialog opens/closes
  useEffect(() => {
    if (openDialog) {
      if (editingConfig) {
        setFormData({
          item_id: editingConfig.item_id,
          node_id: editingConfig.node_id,
          initial_inventory: editingConfig.initial_inventory || 0,
          holding_cost: editingConfig.holding_cost || 0.1,
          backorder_cost: editingConfig.backorder_cost || 1.0,
          service_level: editingConfig.service_level || 0.95,
          is_retailer: editingConfig.is_retailer || false
        });
      } else {
        setFormData({
          item_id: '',
          node_id: '',
          initial_inventory: 0,
          holding_cost: 0.1,
          backorder_cost: 1.0,
          service_level: 0.95,
          is_retailer: false
        });
      }
      setErrors({});
    }
  }, [openDialog, editingConfig]);

  // Filter available nodes when item changes
  useEffect(() => {
    if (formData.item_id) {
      // Get nodes that don't already have a config for this item
      const configuredNodeIds = configs
        .filter(c => c.item_id === formData.item_id)
        .map(c => c.node_id);
      
      const availableNodes = nodes.filter(node => 
        !configuredNodeIds.includes(node.id) || 
        (editingConfig && node.id === editingConfig.node_id)
      );
      
      setFilteredNodes(availableNodes);
      
      // Reset node_id if current selection is no longer valid
      if (formData.node_id && !availableNodes.some(n => n.id === formData.node_id)) {
        setFormData(prev => ({ ...prev, node_id: '' }));
      }
    } else {
      setFilteredNodes([]);
    }
  }, [formData.item_id, formData.node_id, nodes, configs, editingConfig]);

  const handleOpenDialog = (config = null) => {
    setEditingConfig(config);
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setEditingConfig(null);
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.item_id) {
      newErrors.item_id = 'Item is required';
    }
    
    if (!formData.node_id) {
      newErrors.node_id = 'Node is required';
    }
    
    if (formData.initial_inventory < 0) {
      newErrors.initial_inventory = 'Cannot be negative';
    }
    
    if (formData.holding_cost < 0) {
      newErrors.holding_cost = 'Cannot be negative';
    }
    
    if (formData.backorder_cost < 0) {
      newErrors.backorder_cost = 'Cannot be negative';
    }
    
    if (formData.service_level < 0 || formData.service_level > 1) {
      newErrors.service_level = 'Must be between 0 and 1';
    }
    
    // Check for duplicate config
    const isDuplicate = configs.some(
      config => 
        config.item_id === formData.item_id && 
        config.node_id === formData.node_id &&
        (!editingConfig || config.id !== editingConfig.id)
    );
    
    if (isDuplicate) {
      newErrors.duplicate = 'A configuration for this item and node already exists';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    const configData = {
      item_id: formData.item_id,
      node_id: formData.node_id,
      initial_inventory: parseInt(formData.initial_inventory, 10) || 0,
      holding_cost: parseFloat(formData.holding_cost) || 0.1,
      backorder_cost: parseFloat(formData.backorder_cost) || 1.0,
      service_level: parseFloat(formData.service_level) || 0.95,
      is_retailer: formData.is_retailer || false
    };
    
    if (editingConfig) {
      onUpdate(editingConfig.id, configData);
    } else {
      onAdd(configData);
    }
    
    handleCloseDialog();
  };

  const handleDelete = (configId) => {
    if (window.confirm('Are you sure you want to delete this configuration? This action cannot be undone.')) {
      onDelete(configId);
    }
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const getItemName = (itemId) => {
    const item = items.find(i => i.id === itemId);
    return item ? item.name : 'Unknown';
  };

  const getNodeName = (nodeId) => {
    const node = nodes.find(n => n.id === nodeId);
    return node ? node.name : 'Unknown';
  };

  const getNodeType = (nodeId) => {
    const node = nodes.find(n => n.id === nodeId);
    return node ? node.type : 'unknown';
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

  // Pagination
  const startIndex = page * rowsPerPage;
  const endIndex = startIndex + rowsPerPage;
  const paginatedConfigs = configs.slice(startIndex, endIndex);

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">Item-Node Configurations</Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={() => handleOpenDialog()}
          disabled={loading || items.length === 0 || nodes.length === 0}
        >
          Add Configuration
        </Button>
      </Box>

      {items.length === 0 || nodes.length === 0 ? (
        <Paper variant="outlined" sx={{ p: 3, textAlign: 'center' }}>
          <Typography color="textSecondary">
            {items.length === 0 && nodes.length === 0 
              ? 'Add items and nodes first to create configurations.'
              : items.length === 0 
                ? 'Add items first to create configurations.'
                : 'Add nodes first to create configurations.'}
          </Typography>
        </Paper>
      ) : (
        <>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Item</TableCell>
                  <TableCell>Node</TableCell>
                  <TableCell align="right">Initial Inventory</TableCell>
                  <TableCell align="right">Holding Cost</TableCell>
                  <TableCell align="right">Backorder Cost</TableCell>
                  <TableCell align="right">Service Level</TableCell>
                  <TableCell>Retailer</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {paginatedConfigs.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={8} align="center">
                      No configurations added yet. Click "Add Configuration" to get started.
                    </TableCell>
                  </TableRow>
                ) : (
                  paginatedConfigs.map((config) => (
                    <TableRow key={`${config.item_id}-${config.node_id}`}>
                      <TableCell>{getItemName(config.item_id)}</TableCell>
                      <TableCell>
                        <Box display="flex" alignItems="center">
                          <Chip 
                            label={NODE_TYPE_LABELS[getNodeType(config.node_id)]}
                            size="small"
                            color={NODE_TYPE_COLORS[getNodeType(config.node_id)]}
                            sx={{ mr: 1 }}
                          />
                          {getNodeName(config.node_id)}
                        </Box>
                      </TableCell>
                      <TableCell align="right">{config.initial_inventory}</TableCell>
                      <TableCell align="right">${config.holding_cost.toFixed(2)}</TableCell>
                      <TableCell align="right">${config.backorder_cost.toFixed(2)}</TableCell>
                      <TableCell align="right">{(config.service_level * 100).toFixed(1)}%</TableCell>
                      <TableCell>
                        <Checkbox 
                          checked={!!config.is_retailer} 
                          disabled 
                          size="small" 
                        />
                      </TableCell>
                      <TableCell align="right">
                        <Tooltip title="Edit">
                          <IconButton 
                            size="small" 
                            onClick={() => handleOpenDialog(config)}
                            disabled={loading}
                          >
                            <EditIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton 
                            size="small" 
                            color="error"
                            onClick={() => handleDelete(config.id)}
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
              count={configs.length}
              rowsPerPage={rowsPerPage}
              page={page}
              onPageChange={handleChangePage}
              onRowsPerPageChange={handleChangeRowsPerPage}
            />
          </TableContainer>
          
          {configs.length > 0 && (
            <Box mt={1} display="flex" alignItems="center" color="text.secondary">
              <WarningIcon fontSize="small" sx={{ mr: 0.5 }} />
              <Typography variant="caption">
                {configs.filter(c => c.is_retailer).length === 0 
                  ? 'No retailer nodes configured. At least one node should be marked as a retailer.'
                  : `Retailer nodes: ${configs.filter(c => c.is_retailer).length}`}
              </Typography>
            </Box>
          )}
        </>
      )}

      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <form onSubmit={handleSubmit}>
          <DialogTitle>
            {editingConfig ? 'Edit Configuration' : 'Add New Configuration'}
          </DialogTitle>
          <DialogContent>
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={12}>
                <FormControl fullWidth error={!!errors.item_id} size="small">
                  <InputLabel id="item-label">Item</InputLabel>
                  <Select
                    labelId="item-label"
                    name="item_id"
                    value={formData.item_id}
                    label="Item"
                    onChange={handleChange}
                    disabled={loading || !!editingConfig}
                  >
                    {items.map((item) => (
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
                  <InputLabel id="node-label">Node</InputLabel>
                  <Select
                    labelId="node-label"
                    name="node_id"
                    value={formData.node_id}
                    label="Node"
                    onChange={handleChange}
                    disabled={loading || !formData.item_id || !!editingConfig}
                  >
                    {filteredNodes.map((node) => (
                      <MenuItem key={node.id} value={node.id}>
                        <Box display="flex" alignItems="center">
                          <Chip 
                            label={NODE_TYPE_LABELS[node.type]}
                            size="small"
                            color={NODE_TYPE_COLORS[node.type]}
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
                  ) : filteredNodes.length === 0 ? (
                    <FormHelperText>No available nodes for the selected item</FormHelperText>
                  ) : null}
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Initial Inventory"
                  name="initial_inventory"
                  type="number"
                  value={formData.initial_inventory}
                  onChange={handleChange}
                  error={!!errors.initial_inventory}
                  helperText={errors.initial_inventory}
                  disabled={loading}
                  inputProps={{ min: 0, step: 1 }}
                  size="small"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Holding Cost ($/unit/period)"
                  name="holding_cost"
                  type="number"
                  value={formData.holding_cost}
                  onChange={handleChange}
                  error={!!errors.holding_cost}
                  helperText={errors.holding_cost}
                  disabled={loading}
                  inputProps={{ min: 0, step: 0.01 }}
                  size="small"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Backorder Cost ($/unit/period)"
                  name="backorder_cost"
                  type="number"
                  value={formData.backorder_cost}
                  onChange={handleChange}
                  error={!!errors.backorder_cost}
                  helperText={errors.backorder_cost}
                  disabled={loading}
                  inputProps={{ min: 0, step: 0.01 }}
                  size="small"
                />
              </Grid>
              
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Service Level (0-1)"
                  name="service_level"
                  type="number"
                  value={formData.service_level}
                  onChange={handleChange}
                  error={!!errors.service_level}
                  helperText={errors.service_level || 'Probability of meeting demand'}
                  disabled={loading}
                  inputProps={{ min: 0, max: 1, step: 0.01 }}
                  size="small"
                />
              </Grid>
              
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Checkbox
                      name="is_retailer"
                      checked={formData.is_retailer}
                      onChange={handleChange}
                      disabled={loading}
                      color="primary"
                    />
                  }
                  label="This is a retailer node (where customer demand occurs)"
                />
              </Grid>
              
              {errors.duplicate && (
                <Grid item xs={12}>
                  <Typography color="error" variant="body2">
                    {errors.duplicate}
                  </Typography>
                </Grid>
              )}
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
              disabled={loading || !formData.item_id || !formData.node_id}
              startIcon={editingConfig ? <SaveIcon /> : <AddIcon />}
            >
              {editingConfig ? 'Update' : 'Add'} Configuration
            </Button>
          </DialogActions>
        </form>
      </Dialog>
    </Box>
  );
};

export default ItemNodeConfigForm;
