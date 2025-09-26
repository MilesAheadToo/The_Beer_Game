import React, { useState, useEffect, useCallback } from 'react';
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
  Snackbar,
  Alert,
  AlertTitle,
  Divider,
  Stack,
} from '@mui/material';
import {
  PlayArrow,
  Edit,
  Delete,
  Add,
  Settings,
  FileDownloadOutlined,
  SportsEsports,
  PersonOutline,
  RestartAlt,
  Autorenew,
  Visibility,
} from '@mui/icons-material';
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { isGroupAdmin as isGroupAdminUser } from '../utils/authUtils';
import { mixedGameApi } from '../services/api';
import { getModelStatus } from '../services/modelService';

const DEFAULT_CLASSIC_PARAMS = {
  initial_demand: 4,
  change_week: 6,
  final_demand: 8,
};

const normalizeClassicSummary = (pattern = {}) => {
  const params = pattern.params || {};
  const safeNumber = (value, fallback) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };

  const initial = safeNumber(params.initial_demand, DEFAULT_CLASSIC_PARAMS.initial_demand);
  const changeWeek = safeNumber(params.change_week, DEFAULT_CLASSIC_PARAMS.change_week);
  const final = safeNumber(params.final_demand, DEFAULT_CLASSIC_PARAMS.final_demand);
  return ` (Initial: ${initial}, Change Week: ${changeWeek}, Final: ${final})`;
};

const GamesList = () => {
  const [games, setGames] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [modelStatus, setModelStatus] = useState({ is_trained: false });
  const [loadingModelStatus, setLoadingModelStatus] = useState(true);
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const { user } = useAuth();
  const isGroupAdmin = isGroupAdminUser(user);
  const restrictLifecycleActions = isGroupAdmin;
  const scConfigBasePath = isGroupAdmin ? '/admin/group/supply-chain-configs' : '/supply-chain-config';
  const supervisionPathBase = '/admin?section=supervision';

  const goToSupervision = useCallback(
    (gameId) => {
      const focusParam = gameId ? `&focusGameId=${gameId}` : '';
      navigate(`${supervisionPathBase}${focusParam}`);
    },
    [navigate, supervisionPathBase],
  );

  const redirectLifecycleAction = useCallback(
    (gameId) => {
      if (!restrictLifecycleActions) {
        return false;
      }
      goToSupervision(gameId);
      showSnackbar('Use the Supervision tab to start, restart, or review this game.', 'info');
      return true;
    },
    [restrictLifecycleActions, goToSupervision, showSnackbar],
  );

  const showSnackbar = useCallback((message, severity = 'info') => {
    setSnackbar({ open: true, message, severity });
  }, []);

  const handleCloseSnackbar = useCallback(() => {
    setSnackbar((prev) => ({ ...prev, open: false }));
  }, []);

  useEffect(() => {
    const loadModelStatus = async () => {
      try {
        const status = await getModelStatus();
        setModelStatus(status);
      } catch (err) {
        console.error('Failed to fetch model status:', err);
      } finally {
        setLoadingModelStatus(false);
      }
    };
    loadModelStatus();
  }, []);

  const fetchGames = useCallback(async () => {
    try {
      setLoading(true);
      const list = await mixedGameApi.getGames();
      setGames(Array.isArray(list) ? list : []);
      setError(null);
    } catch (err) {
      const detail = err?.response?.data?.detail || err?.message || 'Unable to load games right now.';
      setError(detail);
      setGames([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchGames();
  }, [fetchGames]);

  useEffect(() => {
    if (location.state?.refresh) {
      fetchGames();
      navigate(`${location.pathname}${location.search}`, { replace: true, state: null });
    }
  }, [location, fetchGames, navigate]);

  const handleStartGame = async (gameId) => {
    if (redirectLifecycleAction(gameId)) {
      return;
    }
    try {
      await mixedGameApi.startGame(gameId);
      showSnackbar('Game started successfully', 'success');
      fetchGames();
    } catch (error) {
      const detail = error?.response?.data?.detail || error?.message || 'Failed to start game';
      showSnackbar(detail, 'error');
    }
  };

  const handleResetGame = async (gameId) => {
    if (redirectLifecycleAction(gameId)) {
      return;
    }
    try {
      await mixedGameApi.resetGame(gameId);
      showSnackbar('Game reset successfully', 'success');
      fetchGames();
    } catch (error) {
      const detail = error?.response?.data?.detail || error?.message || 'Unable to reset game';
      showSnackbar(detail, 'error');
    }
  };

  const handleRestartGame = async (gameId) => {
    if (redirectLifecycleAction(gameId)) {
      return;
    }
    try {
      await mixedGameApi.resetGame(gameId);
      await mixedGameApi.startGame(gameId);
      showSnackbar('Game restarted', 'success');
      fetchGames();
    } catch (error) {
      const detail = error?.response?.data?.detail || error?.message || 'Unable to restart game';
      showSnackbar(detail, 'error');
    }
  };

  const handleDeleteGame = async (gameId) => {
    if (!window.confirm('Delete this game? This cannot be undone.')) {
      return;
    }
    try {
      await mixedGameApi.deleteGame(gameId);
      showSnackbar('Game deleted', 'success');
      fetchGames();
    } catch (error) {
      const detail = error?.response?.data?.detail || error?.message || 'Failed to delete game';
      showSnackbar(detail, 'error');
    }
  };

  const handleOpenEditor = (game) => {
    if (game) {
      navigate(`/games/${game.id}/edit`);
    } else {
      navigate('/games/new');
    }
  };

  useEffect(() => {
    const editId = searchParams.get('edit');
    if (!editId) {
      return;
    }
    const next = new URLSearchParams(searchParams);
    next.delete('edit');
    setSearchParams(next, { replace: true });
    navigate(`/games/${editId}/edit`);
  }, [searchParams, setSearchParams, navigate]);

  const formatDate = (dateString) => {
    if (!dateString) return '—';
    const date = new Date(dateString);
    if (Number.isNaN(date.getTime())) {
      return String(dateString);
    }
    return date.toLocaleString();
  };

  const getStatusColor = (status) => {
    switch (String(status || '').toLowerCase()) {
      case 'created':
        return 'primary';
      case 'in_progress':
      case 'round_in_progress':
        return 'warning';
      case 'finished':
      case 'completed':
        return 'success';
      default:
        return 'default';
    }
  };

  const DaybreakAlert = () => {
    if (loadingModelStatus || !modelStatus || modelStatus.is_trained) return null;
    return (
      <Box mb={4}>
        <Alert severity="error" variant="outlined">
          <AlertTitle>Daybreak agent not trained</AlertTitle>
          The Daybreak agent has not yet been trained. Train the model before assigning
          Daybreak strategies.
        </Alert>
      </Box>
    );
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
        <Box sx={{ display: 'flex', gap: 2, mb: 3, justifyContent: 'flex-end' }}>
          <Button
            variant="outlined"
            startIcon={<Settings />}
            onClick={() => navigate('/system-config')}
            color="primary"
          >
            SC Configuration
          </Button>
          <Button
            variant="outlined"
            startIcon={<SportsEsports />}
            onClick={() => navigate(scConfigBasePath)}
          >
            Game Configuration
          </Button>
          <Button
            variant="outlined"
            startIcon={<PersonOutline />}
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

        <DaybreakAlert />
        <Alert severity="error" variant="outlined">
          <AlertTitle>Unable to load games</AlertTitle>
          {error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Stack direction="row" spacing={1} alignItems="center">
          <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: '#24c38b' }} />
          <Typography variant="h6" sx={{ fontWeight: 700 }}>Games</Typography>
        </Stack>
        <Stack direction="row" spacing={1}>
          <Button variant="outlined" startIcon={<Settings />} onClick={() => navigate('/system-config')}>
            SC Configuration
          </Button>
          <Button variant="outlined" startIcon={<SportsEsports />} onClick={() => navigate(scConfigBasePath)}>
            Game Configuration
          </Button>
          <Button variant="outlined" startIcon={<PersonOutline />} onClick={() => navigate('/players')}>
            Players
          </Button>
          <Button variant="outlined" onClick={() => navigate('/admin/training')}>
            Training
          </Button>
          <IconButton onClick={() => { /* Export pending */ }} title="Export">
            <FileDownloadOutlined />
          </IconButton>
        </Stack>
      </Box>
      <Divider sx={{ mb: 2 }} />

      {restrictLifecycleActions && (
        <Alert severity="info" sx={{ mb: 3 }}>
          Start, restart, and review controls are available from the Supervision tab.
        </Alert>
      )}

      <DaybreakAlert />

      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 3 }}>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => handleOpenEditor(null)}
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
              <TableCell>Mode</TableCell>
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
                <TableCell colSpan={8} align="center">
                  No games found. Create a new game to get started.
                </TableCell>
              </TableRow>
            ) : (
              games.map((game) => {
                const paramsSummary = game.demand_pattern?.type === 'classic'
                  ? normalizeClassicSummary(game.demand_pattern)
                  : '';
                const modeLabel = String(game.progression_mode || game?.config?.progression_mode || 'supervised');
                const statusLower = String(game.status || '').toLowerCase();

                const canEdit = statusLower === 'created';
                const canStart = statusLower === 'created';

                return (
                  <TableRow key={game.id}>
                    <TableCell>
                      <Typography noWrap maxWidth={320} title={game.name}>
                        {game.name}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={String(game.status || '').replace(/_/g, ' ').toUpperCase()}
                        color={getStatusColor(game.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={modeLabel.replace(/_/g, ' ').replace(/^./, (s) => s.toUpperCase())}
                        size="small"
                        color={modeLabel.toLowerCase() === 'unsupervised' ? 'info' : 'default'}
                      />
                    </TableCell>
                    <TableCell>{game.current_round}</TableCell>
                    <TableCell>{game.max_rounds}</TableCell>
                    <TableCell>
                      <Typography
                        noWrap
                        maxWidth={260}
                        title={game.demand_pattern?.type || 'classic'}
                      >
                        {(game.demand_pattern?.type || 'classic')}{paramsSummary}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography noWrap maxWidth={220} title={formatDate(game.created_at)}>
                        {formatDate(game.created_at)}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Box display="flex" gap={1} flexWrap="wrap">
                        {!restrictLifecycleActions ? (
                          <>
                            <Button
                              size="small"
                              variant="outlined"
                              startIcon={<SportsEsports />}
                              onClick={() => navigate(`/games/${game.id}`)}
                            >
                              Board
                            </Button>
                            {statusLower === 'finished' || statusLower === 'completed' ? (
                              <Button
                                size="small"
                                variant="contained"
                                color="info"
                                onClick={() => navigate(`/games/${game.id}/report`)}
                              >
                                Report
                              </Button>
                            ) : null}
                            <Button
                              size="small"
                              variant="contained"
                              color="primary"
                              startIcon={<PlayArrow />}
                              onClick={() => handleStartGame(game.id)}
                              disabled={!canStart}
                            >
                              Start
                            </Button>
                          </>
                        ) : (
                          <Button
                            size="small"
                            variant="contained"
                            color="primary"
                            startIcon={<Visibility />}
                            onClick={() => goToSupervision(game.id)}
                          >
                            Supervise
                          </Button>
                        )}
                        <Button
                          size="small"
                          variant="outlined"
                          color="primary"
                          startIcon={<Edit />}
                          onClick={() => handleOpenEditor(game)}
                          disabled={!canEdit}
                        >
                          Edit
                        </Button>
                        {!restrictLifecycleActions && (
                          <>
                            <Tooltip title="Reset game back to round 0">
                              <span>
                                <Button
                                  size="small"
                                  variant="outlined"
                                  startIcon={<Autorenew />}
                                  onClick={() => handleResetGame(game.id)}
                                  disabled={statusLower === 'created'}
                                >
                                  Reset
                                </Button>
                              </span>
                            </Tooltip>
                            <Tooltip title="Reset and immediately start the game">
                              <span>
                                <Button
                                  size="small"
                                  variant="outlined"
                                  startIcon={<RestartAlt />}
                                  onClick={() => handleRestartGame(game.id)}
                                >
                                  Restart
                                </Button>
                              </span>
                            </Tooltip>
                          </>
                        )}
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
                );
              })
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={5000}
        onClose={handleCloseSnackbar}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default GamesList;
