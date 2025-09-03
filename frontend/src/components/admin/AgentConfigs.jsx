import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  CircularProgress,
  Alert
} from '@mui/material';
import { Add as AddIcon, Edit as EditIcon, Delete as DeleteIcon } from '@mui/icons-material';
import api from '../../services/api';
import AgentConfigForm from './AgentConfigForm';

const AgentConfigs = ({ gameId }) => {
  const [configs, setConfigs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingConfig, setEditingConfig] = useState(null);

  const fetchConfigs = async () => {
    try {
      setLoading(true);
      const response = await api.get(`/api/games/${gameId}/agent-configs`);
      setConfigs(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to load agent configurations');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchConfigs();
  }, [gameId]);

  const handleDelete = async (id) => {
    if (window.confirm('Are you sure you want to delete this configuration?')) {
      try {
        await api.delete(`/api/agent-configs/${id}`);
        setConfigs(configs.filter(config => config.id !== id));
      } catch (err) {
        setError('Failed to delete configuration');
        console.error(err);
      }
    }
  };

  const handleSuccess = () => {
    setDialogOpen(false);
    setEditingConfig(null);
    fetchConfigs();
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6">Agent Configurations</Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={() => setDialogOpen(true)}
        >
          New Configuration
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Paper>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Role</TableCell>
                <TableCell>Agent Type</TableCell>
                <TableCell>Configuration</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {configs.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={4} align="center" sx={{ py: 4 }}>
                    <Typography color="textSecondary">
                      No agent configurations found. Create one to get started.
                    </Typography>
                  </TableCell>
                </TableRow>
              ) : (
                configs.map((config) => (
                  <TableRow key={config.id}>
                    <TableCell>
                      <Chip 
                        label={config.role} 
                        color="primary"
                        variant="outlined"
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={config.agent_type.replace('_', ' ')}
                        color="secondary"
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Box sx={{ maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        <Typography variant="body2" noWrap>
                          {JSON.stringify(config.config)}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="right">
                      <IconButton
                        size="small"
                        onClick={() => {
                          setEditingConfig(config);
                          setDialogOpen(true);
                        }}
                      >
                        <EditIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => handleDelete(config.id)}
                        color="error"
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      <Dialog
        open={dialogOpen}
        onClose={() => {
          setDialogOpen(false);
          setEditingConfig(null);
        }}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {editingConfig ? 'Edit Agent Configuration' : 'Create Agent Configuration'}
        </DialogTitle>
        <Divider />
        <DialogContent>
          <AgentConfigForm
            gameId={gameId}
            configId={editingConfig?.id}
            onSuccess={handleSuccess}
          />
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default AgentConfigs;
