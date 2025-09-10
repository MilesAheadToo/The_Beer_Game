import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
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
  Chip
} from '@mui/material';
import { 
  Add as AddIcon, 
  Edit as EditIcon, 
  Delete as DeleteIcon,
  Save as SaveIcon,
  Cancel as CancelIcon
} from '@mui/icons-material';

const NODE_TYPES = [
  { value: 'retailer', label: 'Retailer' },
  { value: 'distributor', label: 'Distributor' },
  { value: 'manufacturer', label: 'Manufacturer' },
  { value: 'supplier', label: 'Supplier' },
];

const NODE_TYPE_COLORS = {
  retailer: 'success',
  distributor: 'info',
  manufacturer: 'warning',
  supplier: 'error',
};

const NodeForm = ({ nodes = [], onAdd, onUpdate, onDelete, loading = false }) => {
  const [openDialog, setOpenDialog] = useState(false);
  const [editingNode, setEditingNode] = useState(null);
  const [formData, setFormData] = useState({
    name: '',
    type: 'retailer',
    description: ''
  });
  const [errors, setErrors] = useState({});

  const handleOpenDialog = (node = null) => {
    if (node) {
      setEditingNode(node);
      setFormData({
        name: node.name,
        type: node.type,
        description: node.description || ''
      });
    } else {
      setEditingNode(null);
      setFormData({
        name: '',
        type: 'retailer',
        description: ''
      });
    }
    setErrors({});
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setEditingNode(null);
    setFormData({
      name: '',
      type: 'retailer',
      description: ''
    });
    setErrors({});
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    }
    
    if (!formData.type) {
      newErrors.type = 'Type is required';
    }
    
    // Check for duplicate node names (case-insensitive)
    const isDuplicate = nodes.some(
      node => 
        node.name.toLowerCase() === formData.name.trim().toLowerCase() && 
        (!editingNode || node.id !== editingNode.id)
    );
    
    if (isDuplicate) {
      newErrors.name = 'A node with this name already exists';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    const nodeData = {
      name: formData.name.trim(),
      type: formData.type,
      description: formData.description.trim() || null
    };
    
    if (editingNode) {
      onUpdate(editingNode.id, nodeData);
    } else {
      onAdd(nodeData);
    }
    
    handleCloseDialog();
  };

  const handleDelete = (nodeId) => {
    if (window.confirm('Are you sure you want to delete this node? This will also remove any associated lanes and configurations.')) {
      onDelete(nodeId);
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const getNodeTypeLabel = (type) => {
    return NODE_TYPES.find(t => t.value === type)?.label || type;
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">Nodes</Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={() => handleOpenDialog()}
          disabled={loading}
        >
          Add Node
        </Button>
      </Box>

      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Description</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {nodes.length === 0 ? (
              <TableRow>
                <TableCell colSpan={4} align="center">
                  No nodes added yet. Click "Add Node" to get started.
                </TableCell>
              </TableRow>
            ) : (
              nodes.map((node) => (
                <TableRow key={node.id}>
                  <TableCell>{node.name}</TableCell>
                  <TableCell>
                    <Chip 
                      label={getNodeTypeLabel(node.type)}
                      size="small"
                      color={NODE_TYPE_COLORS[node.type] || 'default'}
                    />
                  </TableCell>
                  <TableCell>
                    {node.description || <Typography color="textSecondary" variant="body2">No description</Typography>}
                  </TableCell>
                  <TableCell align="right">
                    <Tooltip title="Edit">
                      <IconButton 
                        size="small" 
                        onClick={() => handleOpenDialog(node)}
                        disabled={loading}
                      >
                        <EditIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton 
                        size="small" 
                        color="error"
                        onClick={() => handleDelete(node.id)}
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

      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <form onSubmit={handleSubmit}>
          <DialogTitle>
            {editingNode ? 'Edit Node' : 'Add New Node'}
          </DialogTitle>
          <DialogContent>
            <Grid container spacing={2} sx={{ mt: 1 }}>
              <Grid item xs={12} md={8}>
                <TextField
                  fullWidth
                  label="Node Name"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  error={!!errors.name}
                  helperText={errors.name}
                  disabled={loading}
                  size="small"
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth size="small" error={!!errors.type}>
                  <InputLabel id="node-type-label">Node Type</InputLabel>
                  <Select
                    labelId="node-type-label"
                    name="type"
                    value={formData.type}
                    label="Node Type"
                    onChange={handleChange}
                    disabled={loading}
                  >
                    {NODE_TYPES.map((type) => (
                      <MenuItem key={type.value} value={type.value}>
                        {type.label}
                      </MenuItem>
                    ))}
                  </Select>
                  {errors.type && (
                    <Typography color="error" variant="caption">
                      {errors.type}
                    </Typography>
                  )}
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  multiline
                  rows={2}
                  label="Description (Optional)"
                  name="description"
                  value={formData.description}
                  onChange={handleChange}
                  disabled={loading}
                  size="small"
                />
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
              disabled={loading}
              startIcon={editingNode ? <SaveIcon /> : <AddIcon />}
            >
              {editingNode ? 'Update' : 'Add'} Node
            </Button>
          </DialogActions>
        </form>
      </Dialog>
    </Box>
  );
};

export default NodeForm;
