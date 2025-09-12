import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  CircularProgress,
  Chip,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Snackbar,
  Alert,
  Divider,
  Stack,
} from '@mui/material';
import { PlayArrow, Edit, Delete, Add, Settings, FileDownloadOutlined } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import gameApi from '../services/gameApi';
import { Alert as ChakraAlert, AlertTitle, AlertDescription, AlertIcon, Box as ChakraBox } from '@chakra-ui/react';

const GamesList = () => {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedGame, setSelectedGame] = useState(null);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [modelStatus, setModelStatus] = useState({ is_trained: false });
  const [loadingModelStatus, setLoadingModelStatus] = useState(true);
  const navigate = useNavigate();
  
  // Form state
  const [formData, setFormData] = useState({
    name: '',
    max_rounds: 20,
    demand_pattern: {
      type: 'classic',
      params: {
        stable_period: 5,
        step_increase: 4,
      },
    },
  });

  // Show snackbar notification
  const showSnackbar = React.useCallback((message, severity = 'info') => {
    setSnackbar({ open: true, message, severity });
  }, []);

  // Close snackbar
  const handleCloseSnackbar = React.useCallback(() => {
    setSnackbar(prev => ({ ...prev, open: false }));
  }, []);

  // Check if Daybreak agent is trained
  const [modelStatus, setModelStatus] = useState(null);
  const [loadingModelStatus, setLoadingModelStatus] = useState(true);

  // Fetch model status
  const fetchModelStatus = async () => {
    try {
      const status = await gameApi.getModelStatus();
      setModelStatus(status);
    } catch (error) {
      console.error('Failed to fetch model status:', error);
    } finally {
      setLoadingModelStatus(false);
    }
  };

  // Load model status on component mount
  useEffect(() => {
    fetchModelStatus();
  }, []);

  // Fetch games from the API
  const fetchGames = React.useCallback(async () => {
    try {
      setLoading(true);
      console.log('Fetching games from API...');
      const response = await gameApi.getGames();
      console.log('API Response:', response);
      
      if (Array.isArray(response)) {
        setGames(response);
      } else if (response && Array.isArray(response.data)) {
        setGames(response.data);
      } else {
        console.error('Unexpected API response format:', response);
        setError('Invalid response format from server');
        showSnackbar('Failed to load games: Invalid response format', 'error');
        setGames([]);
      }
      
      setError(null);
    } catch (err) {
      console.error('Failed to fetch games:', err);
      setError('Failed to load games. Please try again.');
      showSnackbar('Failed to load games', 'error');
      setGames([]);
    } finally {
      setLoading(false);
    }
  }, [showSnackbar]);

  // Load games on component mount
  useEffect(() => {
    fetchGames();
  }, [fetchGames]);

  // Simple export to CSV for the current list
  const handleExport = () => {
    if (!Array.isArray(games) || games.length === 0) return;
    const headers = [
      'id','name','status','current_round','max_rounds','demand_pattern','core_config','created_at'
    ];
    const rows = games.map(g => {
      const dp = g?.demand_pattern ? JSON.stringify(g.demand_pattern) : '';
      const cc = g?.core_config || g?.system_config || g?.config || '';
      const ccStr = typeof cc === 'string' ? cc : (cc ? JSON.stringify(cc) : '');
      return [g.id, g.name, g.status, g.current_round, g.max_rounds, dp, ccStr, g.created_at];
    });
    const csv = [headers.join(','), ...rows.map(r => r.map(val => `"${String(val ?? '').replace(/"/g,'""')}"`).join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'games.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  // Render a compact summary of the core/system config
  const renderCoreConfig = (game) => {
    const cc = game?.core_config || game?.system_config || game?.config;
    if (!cc) {
      return (
        <Tooltip title="This game uses current system defaults. Configure in Core Configuration.">
          <Chip label="Defaults" size="small" variant="outlined" color="default" />
        </Tooltip>
      );
    }
    // Try to show a concise label (name or version), with full JSON in tooltip
    const label = cc.name || cc.version || cc.id || 'Custom';
    const tooltip = typeof cc === 'string' ? cc : JSON.stringify(cc, null, 2);
    return (
      <Tooltip title={<pre style={{ margin: 0 }}>{tooltip}</pre>}>
        <Chip label={String(label)} size="small" color="primary" variant="outlined" />
      </Tooltip>
    );
  };

  // Handle dialog open/close
  const handleOpenDialog = (game = null) => {
    if (game) {
      setSelectedGame(game);
      setFormData({
        name: game.name,
        max_rounds: game.max_rounds,
        demand_pattern: {
          type: game.demand_pattern?.type || 'classic',
          params: {
            stable_period: game.demand_pattern?.params?.stable_period || 5,
            step_increase: game.demand_pattern?.params?.step_increase || 4,
          },
        },
      });
    } else {
      setSelectedGame(null);
      setFormData({
        name: '',
        max_rounds: 20,
        demand_pattern: {
          type: 'classic',
          params: {
            stable_period: 5,
            step_increase: 4,
          },
        },
      });
    }
    setOpenDialog(true);
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
  };

  // Handle form input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleDemandPatternChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      demand_pattern: {
        ...formData.demand_pattern,
        [name]: name === 'type' ? value : { ...formData.demand_pattern.params, [name]: value },
      },
    });
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      if (selectedGame) {
        // Update existing game
        await gameApi.updateGame(selectedGame.id, formData);
        showSnackbar('Game updated successfully', 'success');
      } else {
        // Create new game
        await gameApi.createGame(formData);
        showSnackbar('Game created successfully', 'success');
      }
      fetchGames();
      handleCloseDialog();
    } catch (error) {
      console.error('Error saving game:', error);
      showSnackbar('Failed to save game', 'error');
    }
  };

  // Handle game deletion
  const handleDeleteGame = async (gameId) => {
    if (!window.confirm('Are you sure you want to delete this game? This action cannot be undone.')) return;
    try {
      await gameApi.deleteGame(gameId);
      showSnackbar('Game deleted', 'success');
      fetchGames();
    } catch (err) {
      console.error('Delete failed', err);
      showSnackbar(err?.response?.data?.detail || 'Failed to delete game', 'error');
    }
  };

  // Handle game start
  const handleStartGame = async (gameId) => {
    try {
      await gameApi.startGame(gameId);
      showSnackbar('Game started successfully', 'success');
      fetchGames();
    } catch (error) {
      console.error('Error starting game:', error);
      showSnackbar('Failed to start game', 'error');
    }
  };


  // Format date
  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  // Get status color
  const getStatusColor = (status) => {
    switch (status?.toLowerCase()) {
      case 'created':
        return 'primary';
      case 'in_progress':
        return 'warning';
      case 'completed':
        return 'success';
      default:
        return 'default';
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        {/* Quick Navigation Buttons (error state) */}
        <Box sx={{ display: 'flex', gap: 2, mb: 3, justifyContent: 'flex-end' }}>
          <Button
            variant="outlined"
            startIcon={<Settings />}
            onClick={() => navigate('/system-config')}
            color="primary"
          >
            Core Configuration
          </Button>
          <Button
            variant="outlined"
            onClick={() => navigate('/supply-chain-config')}
          >
            Game Configuration
          </Button>
          <Button
            variant="outlined"
            onClick={() => navigate('/players')}
          >
            Players
          </Button>
          <Button
            variant="outlined"
            startIcon={<Settings />}
            onClick={() => navigate('/admin/training')}
            color="secondary"
          >
            Training
          </Button>
        </Box>

        {/* Daybreak Agent Not Trained Alert */}
        {!loadingModelStatus && modelStatus && !modelStatus.is_trained && (
          <ChakraBox mb={4}>
            <Alert status="error" variant="left-accent" borderRadius="md">
              <AlertIcon />
              <Box>
                <AlertTitle>Daybreak Agent Not Trained</AlertTitle>
                <AlertDescription fontSize="sm">
                  The Daybreak agent has not yet been trained, so it cannot be used until training completes.
                  You may still select Basic (heuristics) or LLM agents.
                </AlertDescription>
              </Box>
            </Alert>
          </ChakraBox>
        )}
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header actions matching Daybreak style */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Stack direction="row" spacing={1} alignItems="center">
          <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: '#24c38b' }} />
          <Typography variant="h6" sx={{ fontWeight: 700 }}>Games</Typography>
        </Stack>
        <Stack direction="row" spacing={1}>
          <Button variant="outlined" startIcon={<Settings />} onClick={() => navigate('/system-config')}>
            Core Configuration
          </Button>
          <Button variant="outlined" onClick={() => navigate('/supply-chain-config')}>
            Game Configuration
          </Button>
          <Button variant="outlined" onClick={() => navigate('/players')}>
            Players
          </Button>
          <Button variant="outlined" onClick={() => navigate('/admin/training')}>
            Training
          </Button>
          <IconButton onClick={handleExport} title="Export">
            <FileDownloadOutlined />
          </IconButton>
        </Stack>
      </Box>
      <Divider sx={{ mb: 2 }} />

      {/* Daybreak Agent Not Trained Alert */}
      {!loadingModelStatus && modelStatus && !modelStatus.is_trained && (
        <ChakraBox mb={4}>
          <ChakraAlert status="error" variant="left-accent" borderRadius="md" bg="red.50">
            <AlertIcon color="red.500" />
            <Box>
              <AlertTitle>Daybreak Agent Not Trained</AlertTitle>
              <AlertDescription fontSize="sm">
                The Daybreak agent has not yet been trained, so it cannot be used until training completes.
                You may still select Basic (heuristics) or LLM agents.
              </AlertDescription>
            </Box>
          </ChakraAlert>
        </ChakraBox>
      )}
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 3 }}>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => navigate('/games/new')}
        >
          New Game
        </Button>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Name</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Current Round</TableCell>
              <TableCell>Max Rounds</TableCell>
              <TableCell>Demand Pattern</TableCell>
              <TableCell>Core Config</TableCell>
              <TableCell>Created At</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {games.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} align="center">
                  No games found. Create a new game to get started.
                </TableCell>
              </TableRow>
            ) : (
              games.map((game) => (
                <TableRow key={game.id}>
                  <TableCell>
                    <Typography noWrap maxWidth={320} title={game.name}>
                      {game.name}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={game.status}
                      color={getStatusColor(game.status)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>{game.current_round}</TableCell>
                  <TableCell>{game.max_rounds}</TableCell>
                  <TableCell>
                    <Typography noWrap maxWidth={260} title={game.demand_pattern?.type || 'classic'}>
                      {(game.demand_pattern?.type || 'classic')}
                      {game.demand_pattern?.params?.stable_period &&
                        ` (Stable: ${game.demand_pattern.params.stable_period} weeks)`}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    {renderCoreConfig(game)}
                  </TableCell>
                  <TableCell>
                    <Typography noWrap maxWidth={220} title={formatDate(game.created_at)}>
                      {formatDate(game.created_at)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Box display="flex" gap={1}>
                      <Button
                        size="small"
                        variant="contained"
                        color="primary"
                        startIcon={<PlayArrow />}
                        onClick={() => handleStartGame(game.id)}
                        disabled={String(game.status).toLowerCase() !== 'created'}
                      >
                        Start
                      </Button>
                      <Button
                        size="small"
                        variant="outlined"
                        color="primary"
                        startIcon={<Edit />}
                        onClick={() => handleOpenDialog(game)}
                      >
                        Edit
                      </Button>
                      <Button
                        size="small"
                        variant="outlined"
                        color="error"
                        startIcon={<Delete />}
                        onClick={() => handleDeleteGame(game.id)}
                      >
                        Delete
                      </Button>
                    </Box>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Add/Edit Game Dialog */}
      <Dialog open={openDialog} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
        <form onSubmit={handleSubmit}>
          <DialogTitle>{selectedGame ? 'Edit Game' : 'Create New Game'}</DialogTitle>
          <DialogContent>
            <Box mt={2} mb={2}>
              <TextField
                fullWidth
                label="Game Name"
                name="name"
                value={formData.name}
                onChange={handleInputChange}
                required
                margin="normal"
              />
              <TextField
                fullWidth
                type="number"
                label="Max Rounds"
                name="max_rounds"
                value={formData.max_rounds}
                onChange={handleInputChange}
                required
                margin="normal"
                inputProps={{ min: 1 }}
              />
              <FormControl fullWidth margin="normal">
                <InputLabel>Demand Pattern</InputLabel>
                <Select
                  name="type"
                  value={formData.demand_pattern.type}
                  onChange={handleDemandPatternChange}
                  label="Demand Pattern"
                >
                  <MenuItem value="classic">Classic</MenuItem>
                  <MenuItem value="random">Random</MenuItem>
                  <MenuItem value="seasonal">Seasonal</MenuItem>
                </Select>
              </FormControl>
              {formData.demand_pattern.type === 'classic' && (
                <Box mt={2}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Stable Period (weeks)"
                    name="stable_period"
                    value={formData.demand_pattern.params?.stable_period || 5}
                    onChange={(e) => {
                      const value = e.target.value;
                      setFormData({
                        ...formData,
                        demand_pattern: {
                          ...formData.demand_pattern,
                          params: {
                            ...formData.demand_pattern.params,
                            stable_period: parseInt(value) || 5,
                          },
                        },
                      });
                    }}
                    margin="normal"
                    inputProps={{ min: 1 }}
                  />
                  <TextField
                    fullWidth
                    type="number"
                    label="Step Increase"
                    name="step_increase"
                    value={formData.demand_pattern.params?.step_increase || 4}
                    onChange={(e) => {
                      const value = e.target.value;
                      setFormData({
                        ...formData,
                        demand_pattern: {
                          ...formData.demand_pattern,
                          params: {
                            ...formData.demand_pattern.params,
                            step_increase: parseInt(value) || 4,
                          },
                        },
                      });
                    }}
                    margin="normal"
                    inputProps={{ min: 1 }}
                  />
                </Box>
              )}
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseDialog}>Cancel</Button>
            <Button type="submit" variant="contained" color="primary">
              {selectedGame ? 'Update' : 'Create'}
            </Button>
          </DialogActions>
        </form>
      </Dialog>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
      >
        <Alert
          onClose={handleCloseSnackbar}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default GamesList;
