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
  FormHelperText
} from '@mui/material';
import { 
  Add as AddIcon, 
  Edit as EditIcon, 
  Delete as DeleteIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  ArrowForward as ArrowForwardIcon
} from '@mui/icons-material';
import RangeInput from './RangeInput';

const LaneForm = ({ 
  lanes = [], 
  nodes = [],
  onAdd, 
  onUpdate, 
  onDelete, 
  loading = false 
}) => {
  const [openDialog, setOpenDialog] = useState(false);
  const [editingLane, setEditingLane] = useState(null);
  const [formData, setFormData] = useState({
    from_node_id: '',
    to_node_id: '',
    lead_time: 1,
    capacity: 100,
    cost_per_unit: 1.0
  });
  const [errors, setErrors] = useState({});
  const [filteredToNodes, setFilteredToNodes] = useState([]);

  // Filter available destination nodes based on selected source node
  useEffect(() => {
    if (formData.from_node_id) {
      const fromNode = nodes.find(n => n.id === formData.from_node_id);
      
      // Filter out the source node and any nodes that would create a cycle
      const availableNodes = nodes.filter(node => {
        // Can't connect to self
        if (node.id === formData.from_node_id) return false;
        
        // Get node types
        const fromType = fromNode?.type;
        const toType = node.type;
        
        // Enforce supply chain flow: supplier -> manufacturer -> distributor -> retailer
        switch (fromType) {
          case 'supplier':
            return toType === 'manufacturer';
          case 'manufacturer':
            return toType === 'distributor';
          case 'distributor':
            return toType === 'retailer';
          default:
            return false; // Retailers can't have outgoing lanes
        }
      });
      
      setFilteredToNodes(availableNodes);
      
      // Reset to_node_id if current selection is no longer valid
      if (formData.to_node_id && !availableNodes.some(n => n.id === formData.to_node_id)) {
        setFormData(prev => ({ ...prev, to_node_id: '' }));
      }
    } else {
      setFilteredToNodes([]);
    }
  }, [formData.from_node_id, nodes]);

  const handleOpenDialog = (lane = null) => {
    if (lane) {
      setEditingLane(lane);
      setFormData({
        from_node_id: lane.from_node_id,
        to_node_id: lane.to_node_id,
        lead_time: lane.lead_time || 1,
        capacity: lane.capacity || 100,
        cost_per_unit: lane.cost_per_unit || 1.0
      });
    } else {
      setEditingLane(null);
      setFormData({
        from_node_id: '',
        to_node_id: '',
        lead_time: 1,
        capacity: 100,
        cost_per_unit: 1.0
      });
    }
    setErrors({});
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setEditingLane(null);
    setFormData({
      from_node_id: '',
      to_node_id: '',
      lead_time: 1,
      capacity: 100,
      cost_per_unit: 1.0
    });
    setErrors({});
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.from_node_id) {
      newErrors.from_node_id = 'Source node is required';
    }
    
    if (!formData.to_node_id) {
      newErrors.to_node_id = 'Destination node is required';
    } else if (formData.from_node_id === formData.to_node_id) {
      newErrors.to_node_id = 'Source and destination cannot be the same';
    }
    
    if (formData.lead_time < 0) {
      newErrors.lead_time = 'Lead time cannot be negative';
    }
    
    if (formData.capacity <= 0) {
      newErrors.capacity = 'Capacity must be greater than 0';
    }
    
    if (formData.cost_per_unit < 0) {
      newErrors.cost_per_unit = 'Cost cannot be negative';
    }
    
    // Check for duplicate lanes
    const isDuplicate = lanes.some(
      lane => 
        lane.from_node_id === formData.from_node_id && 
        lane.to_node_id === formData.to_node_id &&
        (!editingLane || lane.id !== editingLane.id)
    );
    
    if (isDuplicate) {
      newErrors.duplicate = 'A lane between these nodes already exists';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    const laneData = {
      from_node_id: formData.from_node_id,
      to_node_id: formData.to_node_id,
      lead_time: parseInt(formData.lead_time, 10),
      capacity: parseInt(formData.capacity, 10),
      cost_per_unit: parseFloat(formData.cost_per_unit)
    };
    
    if (editingLane) {
      onUpdate(editingLane.id, laneData);
    } else {
      onAdd(laneData);
    }
    
    handleCloseDialog();
  };

  const handleDelete = (laneId) => {
    if (window.confirm('Are you sure you want to delete this lane? This action cannot be undone.')) {
      onDelete(laneId);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
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

  // Get source nodes that can have outgoing lanes (all except retailers)
  const sourceNodes = nodes.filter(node => node.type !== 'retailer');

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">Lanes</Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={() => handleOpenDialog()}
          disabled={loading || nodes.length < 2}
        >
          Add Lane
        </Button>
      </Box>

      {nodes.length < 2 ? (
        <Paper variant="outlined" sx={{ p: 3, textAlign: 'center' }}>
          <Typography color="textSecondary">
            Add at least 2 nodes to create lanes between them.
          </Typography>
        </Paper>
      ) : (
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>From Node</TableCell>
                <TableCell>To Node</TableCell>
                <TableCell align="right">Lead Time</TableCell>
                <TableCell align="right">Capacity</TableCell>
                <TableCell align="right">Cost/Unit</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {lanes.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} align="center">
                    No lanes added yet. Click "Add Lane" to connect nodes.
                  </TableCell>
                </TableRow>
              ) : (
                lanes.map((lane) => (
                  <TableRow key={lane.id}>
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        <Chip 
                          label={NODE_TYPE_LABELS[getNodeType(lane.from_node_id)]}
                          size="small"
                          color={NODE_TYPE_COLORS[getNodeType(lane.from_node_id)]}
                          sx={{ mr: 1 }}
                        />
                        {getNodeName(lane.from_node_id)}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        <Chip 
                          label={NODE_TYPE_LABELS[getNodeType(lane.to_node_id)]}
                          size="small"
                          color={NODE_TYPE_COLORS[getNodeType(lane.to_node_id)]}
                          sx={{ mr: 1 }}
                        />
                        {getNodeName(lane.to_node_id)}
                      </Box>
                    </TableCell>
                    <TableCell align="right">{lane.lead_time} days</TableCell>
                    <TableCell align="right">{lane.capacity.toLocaleString()}</TableCell>
                    <TableCell align="right">${lane.cost_per_unit.toFixed(2)}</TableCell>
                    <TableCell align="right">
                      <Tooltip title="Edit">
                        <IconButton 
                          size="small" 
                          onClick={() => handleOpenDialog(lane)}
                          disabled={loading}
                        >
                          <EditIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <IconButton 
                          size="small" 
                          color="error"
                          onClick={() => handleDelete(lane.id)}
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
        </TableContainer>
      )}

      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <form onSubmit={handleSubmit}>
          <DialogTitle>
            {editingLane ? 'Edit Lane' : 'Add New Lane'}
          </DialogTitle>
          <DialogContent>
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={12}>
                <FormControl fullWidth error={!!errors.from_node_id} size="small">
                  <InputLabel id="from-node-label">From Node</InputLabel>
                  <Select
                    labelId="from-node-label"
                    name="from_node_id"
                    value={formData.from_node_id}
                    label="From Node"
                    onChange={handleChange}
                    disabled={loading || !!editingLane}
                  >
                    {sourceNodes.map((node) => (
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
                  {errors.from_node_id && (
                    <FormHelperText>{errors.from_node_id}</FormHelperText>
                  )}
                </FormControl>
              </Grid>
              
              <Grid item xs={12}>
                <FormControl fullWidth error={!!errors.to_node_id} size="small">
                  <InputLabel id="to-node-label">To Node</InputLabel>
                  <Select
                    labelId="to-node-label"
                    name="to_node_id"
                    value={formData.to_node_id}
                    label="To Node"
                    onChange={handleChange}
                    disabled={loading || !formData.from_node_id || !!editingLane}
                  >
                    {filteredToNodes.map((node) => (
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
                  {errors.to_node_id ? (
                    <FormHelperText>{errors.to_node_id}</FormHelperText>
                  ) : !formData.from_node_id ? (
                    <FormHelperText>Select a source node first</FormHelperText>
                  ) : filteredToNodes.length === 0 ? (
                    <FormHelperText>No valid destination nodes available for the selected source</FormHelperText>
                  ) : null}
                </FormControl>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Lead Time (days)"
                  name="lead_time"
                  type="number"
                  value={formData.lead_time}
                  onChange={handleChange}
                  error={!!errors.lead_time}
                  helperText={errors.lead_time}
                  disabled={loading}
                  inputProps={{ min: 0, step: 1 }}
                  size="small"
                />
              </Grid>
              
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Capacity (units)"
                  name="capacity"
                  type="number"
                  value={formData.capacity}
                  onChange={handleChange}
                  error={!!errors.capacity}
                  helperText={errors.capacity}
                  disabled={loading}
                  inputProps={{ min: 1, step: 1 }}
                  size="small"
                />
              </Grid>
              
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Cost per Unit ($)"
                  name="cost_per_unit"
                  type="number"
                  value={formData.cost_per_unit}
                  onChange={handleChange}
                  error={!!errors.cost_per_unit}
                  helperText={errors.cost_per_unit}
                  disabled={loading}
                  inputProps={{ min: 0, step: 0.01 }}
                  size="small"
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
              disabled={loading || !formData.from_node_id || !formData.to_node_id}
              startIcon={editingLane ? <SaveIcon /> : <AddIcon />}
            >
              {editingLane ? 'Update' : 'Add'} Lane
            </Button>
          </DialogActions>
        </form>
      </Dialog>
    </Box>
  );
};

export default LaneForm;
