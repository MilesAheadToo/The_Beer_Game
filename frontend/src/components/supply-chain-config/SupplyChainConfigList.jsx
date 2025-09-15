import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider,
  IconButton,
  List,
  ListItem,
  Menu,
  MenuItem,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  CheckCircle as ActiveIcon,
  RadioButtonUnchecked as InactiveIcon,
  ContentCopy as CopyIcon,
  MoreVert as MoreVertIcon,
  SportsEsports as GameIcon,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { format } from 'date-fns';
import api from '../../services/api';

const SupplyChainConfigList = ({
  title = 'Supply Chain Configurations',
  basePath = '/supply-chain-config',
  restrictToGroupId = null,
} = {}) => {

  const [configs, setConfigs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [configToDelete, setConfigToDelete] = useState(null);
  const [anchorEl, setAnchorEl] = useState(null);
  const [selectedConfig, setSelectedConfig] = useState(null);
  const [activatingConfig, setActivatingConfig] = useState(null);

  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();

  const formatDate = (value) => {
    if (!value) return '—';
    try {
      return format(new Date(value), 'MMM d, yyyy');
    } catch (error) {
      return value;
    }
  };

  const fetchConfigs = useCallback(async () => {
    try {
      setLoading(true);
      const response = await api.get('/api/v1/supply-chain-config');
      const data = response.data || [];
      const targetGroupId =
        restrictToGroupId !== null && restrictToGroupId !== undefined
          ? String(restrictToGroupId)
          : null;

      const filteredConfigs =
        targetGroupId !== null
          ? data.filter(
              (config) =>
                config?.group_id !== undefined &&
                config?.group_id !== null &&
                String(config.group_id) === targetGroupId,
            )
          : data;

      setConfigs(filteredConfigs);
      setError(null);
    } catch (err) {
      console.warn('Supply chain configs endpoint unavailable; showing empty list.', err?.response?.status);
      setConfigs([]);
      setError('Unable to load supply chain configurations right now.');
    } finally {
      setLoading(false);
    }
  }, [restrictToGroupId]);

  useEffect(() => {
    fetchConfigs();
  }, [fetchConfigs]);

  const handleCreateNew = () => {
    navigate(`${basePath}/new`);
  };

  const handleEdit = (id) => {
    navigate(`${basePath}/edit/${id}`);
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
      await fetchConfigs();
    } catch (err) {
      console.error('Error deleting configuration:', err);
      enqueueSnackbar('Failed to delete configuration', { variant: 'error' });
    } finally {
      setDeleteDialogOpen(false);
      setConfigToDelete(null);
    }
  };

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
    if (!configId || activatingConfig === configId) return;

    try {
      setActivatingConfig(configId);
      await api.put(`/api/v1/supply-chain-config/${configId}`, { is_active: true });
      enqueueSnackbar('Configuration activated successfully', { variant: 'success' });
      await fetchConfigs();
    } catch (err) {
      console.error('Error activating configuration:', err);
      enqueueSnackbar('Failed to activate configuration', { variant: 'error' });
    } finally {
      setActivatingConfig(null);
      handleMenuClose();
    }
  };

  const handleDuplicate = async (config) => {
    if (!config) return;

    try {
      const { id, created_at, updated_at, is_active, ...configData } = config;
      await api.post('/api/v1/supply-chain-config', {
        ...configData,
        group_id: config.group_id ?? configData.group_id ?? null,
        name: `${config.name} (Copy)`,
        is_active: false,
      });
      enqueueSnackbar('Configuration duplicated successfully', { variant: 'success' });
      await fetchConfigs();
    } catch (err) {
      console.error('Error duplicating configuration:', err);
      enqueueSnackbar('Failed to duplicate configuration', { variant: 'error' });
    } finally {
      handleMenuClose();
    }
  };

  const renderTableBody = () => {
    if (configs.length === 0) {
      return (
        <TableRow>
          <TableCell colSpan={6} align="center">
            {loading ? (
              <CircularProgress size={24} />
            ) : (
              <Typography
                variant="body2"
                color={error ? 'error.main' : 'text.secondary'}
              >
                {error || 'No supply chain configurations found. Create your first configuration to get started.'}
              </Typography>
            )}
          </TableCell>
        </TableRow>
      );
    }

    return (
      <>
        {configs.map((config) => (
          <TableRow hover key={config.id}>
            <TableCell>
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                {config.name}
              </Typography>
            </TableCell>
            <TableCell>
              <Typography
                variant="body2"
                color="text.secondary"
                noWrap
                title={config.description || 'No description provided'}
              >
                {config.description || 'No description provided'}
              </Typography>
            </TableCell>
            <TableCell>
              {config.is_active ? (
                <Chip
                  label="Active"
                  color="success"
                  size="small"
                  icon={<ActiveIcon />}
                  sx={{ fontWeight: 600 }}
                />
              ) : (
                <Chip
                  label="Inactive"
                  color="default"
                  size="small"
                  icon={<InactiveIcon />}
                />
              )}
            </TableCell>
            <TableCell>{formatDate(config.created_at)}</TableCell>
            <TableCell>{formatDate(config.updated_at)}</TableCell>
            <TableCell align="right">
              <Tooltip title="Edit">
                <IconButton size="small" onClick={() => handleEdit(config.id)}>
                  <EditIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              <Tooltip title="More actions">
                <IconButton size="small" onClick={(event) => handleMenuOpen(event, config)}>
                  <MoreVertIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              <Tooltip title={config.is_active ? 'Deactivate before deleting' : 'Delete'}>
                <span>
                  <IconButton
                    size="small"
                    color="error"
                    onClick={() => handleDeleteClick(config)}
                    disabled={config.is_active}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </span>
              </Tooltip>
            </TableCell>
          </TableRow>
        ))}
        {loading && (
          <TableRow>
            <TableCell colSpan={6} align="center">
              <CircularProgress size={20} />
            </TableCell>
          </TableRow>
        )}
      </>
    );
  };

  return (
    <>
      <Card>
        <CardHeader 
          title={title}
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
              <Typography
                variant="body1"
                color={error ? 'error.main' : 'text.secondary'}
              >
                {error || 'No supply chain configurations found. Create your first configuration to get started.'}
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


      <Paper elevation={0} sx={{ p: 3 }}>
        <Box
          display="flex"
          alignItems="center"
          justifyContent="space-between"
          mb={3}
          flexWrap="wrap"
          gap={2}
        >
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 700 }}>
              Supply Chain Configurations
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Manage the supply chain setups available for your group&apos;s games.
            </Typography>
          </Box>
          <Button
            variant="contained"
            color="primary"
            startIcon={<AddIcon />}
            onClick={handleCreateNew}
          >
            New Configuration
          </Button>
        </Box>

        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Description</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Created</TableCell>
                <TableCell>Updated</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>{renderTableBody()}</TableBody>
          </Table>
        </TableContainer>
      </Paper>

      <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={handleMenuClose}>
        <MenuItem
          onClick={() => {
            if (selectedConfig) {
              handleEdit(selectedConfig.id);
            }
            handleMenuClose();
          }}
        >
          <EditIcon sx={{ mr: 1 }} fontSize="small" /> Edit
        </MenuItem>
        <MenuItem onClick={handleCreateGame}>
          <GameIcon sx={{ mr: 1 }} fontSize="small" /> Create Game
        </MenuItem>
        <MenuItem onClick={() => handleDuplicate(selectedConfig)}>
          <CopyIcon sx={{ mr: 1 }} fontSize="small" /> Duplicate
        </MenuItem>
        {!selectedConfig?.is_active && (
          <MenuItem
            onClick={() => handleActivateConfig(selectedConfig?.id)}
            disabled={activatingConfig === selectedConfig?.id}
          >
            {activatingConfig === selectedConfig?.id ? (
              <CircularProgress size={20} sx={{ mr: 1 }} />
            ) : (
              <InactiveIcon sx={{ mr: 1 }} fontSize="small" />
            )}
            Activate
          </MenuItem>
        )}
        <MenuItem
          onClick={() => {
            if (selectedConfig && !selectedConfig.is_active) {
              handleDeleteClick(selectedConfig);
            }
            handleMenuClose();
          }}
          disabled={selectedConfig?.is_active}
          sx={{ color: 'error.main' }}
        >
          <DeleteIcon sx={{ mr: 1 }} fontSize="small" /> Delete
        </MenuItem>
      </Menu>

      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Configuration</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the configuration &quot;{configToDelete?.name}&quot;? This action cannot be undone.
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
    </>
  );
};

export default SupplyChainConfigList;
