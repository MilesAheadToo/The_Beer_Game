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
} from '@mui/material';
import { PlayArrow, Edit, Delete, Add } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import gameApi from '../services/gameApi';

const GamesList = () => {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedGame, setSelectedGame] = useState(null);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
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
    if (window.confirm('Are you sure you want to delete this game?')) {
      // Delete endpoint is not available in the backend yet
      showSnackbar('Game deletion is disabled in this build.', 'info');
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
      <Box p={3}>
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  return (
    <Box p={3}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Games</Typography>
        <Button
          variant="contained"
          color="primary"
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
              <TableCell>Created At</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {games.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} align="center">
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
                    <Typography noWrap maxWidth={220} title={formatDate(game.created_at)}>
                      {formatDate(game.created_at)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Box display="flex" gap={1}>
                      <Tooltip title="Start Game">
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => handleStartGame(game.id)}
                          disabled={game.status !== 'created'}
                        >
                          <PlayArrow />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Edit">
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => handleOpenDialog(game)}
                        >
                          <Edit />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => handleDeleteGame(game.id)}
                        >
                          <Delete />
                        </IconButton>
                      </Tooltip>
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
