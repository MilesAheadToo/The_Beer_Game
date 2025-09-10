import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Box, 
  Button, 
  Card, 
  CardContent, 
  CardHeader, 
  Divider, 
  Grid, 
  IconButton, 
  List, 
  ListItem, 
  ListItemSecondaryAction, 
  ListItemText, 
  Tooltip,
  Typography,
  Chip,
  CircularProgress,
  Alert,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Menu,
  MenuItem
} from '@mui/material';
import { 
  Add as AddIcon, 
  Edit as EditIcon, 
  Delete as DeleteIcon,
  CheckCircle as ActiveIcon,
  RadioButtonUnchecked as InactiveIcon,
  ContentCopy as CopyIcon,
  MoreVert as MoreVertIcon,
  SportsEsports as GameIcon
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { format } from 'date-fns';
import api from '../../services/api';

const SupplyChainConfigList = () => {
  const [configs, setConfigs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [configToDelete, setConfigToDelete] = useState(null);
  const [activatingConfig, setActivatingConfig] = useState(null);
  
  // Menu state
  const [anchorEl, setAnchorEl] = useState(null);
  const [selectedConfig, setSelectedConfig] = useState(null);
  
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();

  const fetchConfigs = async () => {
    try {
      setLoading(true);
      const response = await api.get('/api/v1/supply-chain-config');
      setConfigs(response.data);
      setError(null);
    } catch (err) {
      console.error('Error fetching supply chain configs:', err);
      setError('Failed to load configurations. Please try again later.');
      enqueueSnackbar('Failed to load configurations', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfigs();
  }, []);

  const handleCreateNew = () => {
    navigate('/supply-chain-config/new');
  };

  const handleEdit = (id) => {
    navigate(`/supply-chain-config/edit/${id}`);
  };

  const handleDeleteClick = (config) => {
    setConfigToDelete(config);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!configToDelete) return;
    
    try {
      await api.delete(`/api/v1/supply-chain-config/${configToDelete.id}`);
      enqueueSnackbar('Configuration deleted successfully', { variant: 'success' });
      fetchConfigs();
    } catch (err) {
      console.error('Error deleting configuration:', err);
      enqueueSnackbar('Failed to delete configuration', { variant: 'error' });
    } finally {
      setDeleteDialogOpen(false);
      setConfigToDelete(null);
    }
  };

  // Menu handlers
  const handleMenuOpen = (event, config) => {
    setAnchorEl(event.currentTarget);
    setSelectedConfig(config);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedConfig(null);
  };

  const handleCreateGame = () => {
    if (selectedConfig) {
      navigate(`/games/new-from-config/${selectedConfig.id}`);
    }
    handleMenuClose();
  };

  const handleActivateConfig = async (configId) => {
    if (activatingConfig === configId) return;
    
    try {
      setActivatingConfig(configId);
      await api.put(`/api/v1/supply-chain-config/${configId}`, { is_active: true });
      enqueueSnackbar('Configuration activated successfully', { variant: 'success' });
      fetchConfigs();
    } catch (err) {
      console.error('Error activating configuration:', err);
      enqueueSnackbar('Failed to activate configuration', { variant: 'error' });
    } finally {
      setActivatingConfig(null);
    }
  };

  const handleDuplicate = async (config) => {
    try {
      const { id, created_at, updated_at, is_active, ...configData } = config;
      const newConfig = {
        ...configData,
        name: `${config.name} (Copy)`,
        is_active: false
      };
      
      await api.post('/api/v1/supply-chain-config', newConfig);
      enqueueSnackbar('Configuration duplicated successfully', { variant: 'success' });
      fetchConfigs();
    } catch (err) {
      console.error('Error duplicating configuration:', err);
      enqueueSnackbar('Failed to duplicate configuration', { variant: 'error' });
    }
  };

  if (loading && !configs.length) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
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
    <>
      <Card>
        <CardHeader 
          title="Supply Chain Configurations"
          action={
            <Button
              variant="contained"
              color="primary"
              startIcon={<AddIcon />}
              onClick={handleCreateNew}
            >
              New Configuration
            </Button>
          }
        />
        <Divider />
        <CardContent>
          {configs.length === 0 ? (
            <Box textAlign="center" py={4}>
              <Typography variant="body1" color="textSecondary">
                No supply chain configurations found. Create your first configuration to get started.
              </Typography>
              <Box mt={2}>
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<AddIcon />}
                  onClick={handleCreateNew}
                >
                  Create Configuration
                </Button>
              </Box>
            </Box>
          ) : (
            <List>
              {configs.map((config) => (
                <React.Fragment key={config.id}>
                  <ListItem>
                    <Box display="flex" alignItems="center" width="100%">
                      <Box flexGrow={1}>
                        <Box display="flex" alignItems="center" mb={1}>
                          <Typography variant="h6" component="div">
                            {config.name}
                          </Typography>
                          {config.is_active && (
                            <Chip
                              label="Active"
                              color="success"
                              size="small"
                              sx={{ ml: 1 }}
                              icon={<ActiveIcon />}
                            />
                          )}
                        </Box>
                        <Typography variant="body2" color="textSecondary">
                          {config.description || 'No description provided'}
                        </Typography>
                        <Box display="flex" mt={1}>
                          <Typography variant="caption" color="textSecondary" sx={{ mr: 2 }}>
                            Created: {format(new Date(config.created_at), 'MMM d, yyyy')}
                          </Typography>
                          <Typography variant="caption" color="textSecondary">
                            Updated: {format(new Date(config.updated_at), 'MMM d, yyyy')}
                          </Typography>
                        </Box>
                      </Box>
                      <Box>
                        <Tooltip title="Actions">
                          <IconButton
                            onClick={(e) => handleMenuOpen(e, config)}
                            color="primary"
                            size="large"
                          >
                            <MoreVertIcon />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Box>
                  </ListItem>
                  <Divider component="li" />
                </React.Fragment>
              ))}
            </List>
          )}
        </CardContent>
      </Card>

      {/* Actions Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => { handleEdit(selectedConfig?.id); handleMenuClose(); }}>
          <EditIcon sx={{ mr: 1 }} /> Edit
        </MenuItem>
        <MenuItem onClick={handleCreateGame}>
          <GameIcon sx={{ mr: 1 }} /> Create Game
        </MenuItem>
        <MenuItem onClick={() => { handleDuplicate(selectedConfig); handleMenuClose(); }}>
          <CopyIcon sx={{ mr: 1 }} /> Duplicate
        </MenuItem>
        {!selectedConfig?.is_active && (
          <MenuItem 
            onClick={() => { 
              handleActivateConfig(selectedConfig?.id); 
              handleMenuClose(); 
            }}
            disabled={activatingConfig === selectedConfig?.id}
          >
            {activatingConfig === selectedConfig?.id ? (
              <CircularProgress size={20} sx={{ mr: 1 }} />
            ) : (
              <InactiveIcon sx={{ mr: 1 }} />
            )}
            Activate
          </MenuItem>
        )}
        <Divider />
        <MenuItem 
          onClick={() => { handleDeleteClick(selectedConfig); handleMenuClose(); }}
          disabled={selectedConfig?.is_active}
          sx={{ color: 'error.main' }}
        >
          <DeleteIcon sx={{ mr: 1 }} /> Delete
        </MenuItem>
      </Menu>

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        aria-labelledby="delete-dialog-title"
      >
        <DialogTitle id="delete-dialog-title">
          Delete Configuration
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the configuration "{configToDelete?.name}"? 
            This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)} color="primary">
            Cancel
          </Button>
          <Button 
            onClick={handleDeleteConfirm} 
            color="error"
            variant="contained"
            autoFocus
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Actions Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => { handleEdit(selectedConfig?.id); handleMenuClose(); }}>
          <EditIcon sx={{ mr: 1 }} /> Edit
        </MenuItem>
        <MenuItem onClick={handleCreateGame}>
          <GameIcon sx={{ mr: 1 }} /> Create Game
        </MenuItem>
        <Divider />
        <MenuItem 
          onClick={() => { handleDeleteClick(selectedConfig); handleMenuClose(); }}
          sx={{ color: 'error.main' }}
        >
          <DeleteIcon sx={{ mr: 1 }} /> Delete
        </MenuItem>
      </Menu>
    </>
  );
};

export default SupplyChainConfigList;
