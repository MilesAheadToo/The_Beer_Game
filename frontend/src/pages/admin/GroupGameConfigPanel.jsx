import React, { useMemo } from 'react';
import {
  Box,
  Button,
  Chip,
  CircularProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Stack,
  Alert,
  IconButton,
  Tooltip,
} from '@mui/material';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { useNavigate } from 'react-router-dom';
import { format } from 'date-fns';
import { useSnackbar } from 'notistack';
import { mixedGameApi } from '../../services/api';

const formatDate = (value) => {
  if (!value) return '—';
  try {
    return format(new Date(value), 'MMM d, yyyy HH:mm');
  } catch (error) {
    return String(value);
  }
};

const statusColor = (status = '') => {
  const normalized = String(status).toLowerCase();
  if (normalized === 'created') return 'default';
  if (normalized === 'in_progress') return 'info';
  if (normalized === 'completed') return 'success';
  if (normalized === 'paused') return 'warning';
  return 'default';
};

const GroupGameConfigPanel = ({
  games = [],
  loading = false,
  error = null,
  onRefresh,
  groupId = null,
  currentUserId = null,
}) => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();

  const filteredGames = useMemo(() => {
    if (!Array.isArray(games) || games.length === 0) {
      return [];
    }

    return games.filter((game) => {
      if (!game) return false;
      const targetGroup = game.group_id ?? game?.config?.group_id ?? null;
      if (groupId != null) {
        if (targetGroup != null) {
          return Number(targetGroup) === Number(groupId);
        }
        if (game.created_by != null) {
          return Number(game.created_by) === Number(currentUserId);
        }
      }
      return true;
    });
  }, [games, groupId, currentUserId]);

  const handleCreateGame = () => {
    navigate('/games/new');
  };

  const handleViewGame = (gameId, status) => {
    if (String(status || '').toLowerCase() === 'completed') {
      navigate(`/games/${gameId}/report`);
    } else {
      navigate(`/games/${gameId}`);
    }
  };

  const runAction = async (gameId, action, apiCall, successMessage) => {
    try {
      await apiCall(gameId);
      enqueueSnackbar(successMessage, { variant: 'success' });
      if (onRefresh) {
        await onRefresh();
      }
    } catch (err) {
      const detail = err?.response?.data?.detail || err?.message || 'Action failed';
      enqueueSnackbar(detail, { variant: 'error' });
    }
  };

  const handleRestart = async (gameId) => {
    try {
      await mixedGameApi.resetGame(gameId);
      await mixedGameApi.startGame(gameId);
      enqueueSnackbar('Game restarted', { variant: 'success' });
      if (onRefresh) {
        await onRefresh();
      }
    } catch (err) {
      const detail = err?.response?.data?.detail || err?.message || 'Unable to restart game';
      enqueueSnackbar(detail, { variant: 'error' });
    }
  };
  const handleDelete = async (gameId, name) => {
    if (!window.confirm(`Delete game "${name}"? This cannot be undone.`)) return;
    await runAction(gameId, 'delete', mixedGameApi.deleteGame, 'Game deleted');
  };
  const handleEdit = (game) => {
    onRefresh?.();
    navigate(`/games?edit=${game.id}`);
  };

  return (
    <Paper elevation={0} sx={{ p: 3 }}>
      <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" alignItems={{ xs: 'stretch', md: 'center' }} spacing={2} mb={3}>
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            Game Configuration
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Review recent mixed game setups and create new sessions for your group.
          </Typography>
        </Box>
        <Stack direction="row" spacing={1}>
          {onRefresh && (
            <Button variant="outlined" onClick={onRefresh} disabled={loading}>
              Refresh
            </Button>
          )}
          <Button variant="contained" color="primary" onClick={handleCreateGame}>
            New Mixed Game
          </Button>
        </Stack>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {loading ? (
        <Box display="flex" justifyContent="center" alignItems="center" minHeight={240}>
          <CircularProgress />
        </Box>
      ) : filteredGames.length === 0 ? (
        <Box textAlign="center" py={6}>
          <Typography variant="body1" color="text.secondary" gutterBottom>
            No games found for your group yet.
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Configure a new mixed game to get your players started.
          </Typography>
        </Box>
      ) : (
        <TableContainer sx={{ overflowX: 'auto' }}>
          <Table size="small" sx={{ tableLayout: 'fixed' }}>
            <TableHead>
              <TableRow>
                <TableCell rowSpan={2} sx={{ width: { xs: '60vw', md: 'auto' } }}>Name</TableCell>
                <TableCell rowSpan={2}>Mode</TableCell>
                <TableCell rowSpan={2}>Status</TableCell>
                <TableCell align="center" colSpan={2}>Rounds</TableCell>
                <TableCell rowSpan={2} sx={{ width: { xs: 90, md: 120 } }}>Last Updated</TableCell>
                <TableCell rowSpan={2} align="right" sx={{ width: { xs: 140, md: 210 } }}>Actions</TableCell>
              </TableRow>
              <TableRow>
                <TableCell sx={{ width: 80 }}>Current</TableCell>
                <TableCell sx={{ width: 80 }}>Max</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredGames.map((game) => (
                <TableRow hover key={game.id}>
                  <TableCell sx={{ pr: 2 }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600, whiteSpace: 'normal', wordBreak: 'break-word' }}>
                      {game.name}
                    </Typography>
                    {game.description && (
                      <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'normal', wordBreak: 'break-word' }}>
                        {game.description}
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    <Chip
                      size="small"
                      variant="outlined"
                      color={String((game.progression_mode || game?.config?.progression_mode || 'supervised')).toLowerCase() === 'unsupervised' ? 'info' : 'default'}
                      label={String(game.progression_mode || game?.config?.progression_mode || 'supervised').replace(/_/g, ' ').replace(/^./, (s) => s.toUpperCase())}
                    />
                  </TableCell>
                  <TableCell>
                    <Chip
                      size="small"
                      color={statusColor(game.status)}
                      label={(game.status || '').replace(/_/g, ' ') || 'Unknown'}
                    />
                  </TableCell>
                  <TableCell sx={{ width: 80 }}>{game.current_round ?? 0}</TableCell>
                  <TableCell sx={{ width: 80 }}>{game.max_rounds ?? '—'}</TableCell>
                  <TableCell sx={{ width: { xs: 90, md: 120 } }}>{formatDate(game.updated_at)}</TableCell>
                  <TableCell align="right">
                    <Stack direction="row" spacing={1} justifyContent="flex-end">
                      <Tooltip title="View">
                        <span>
                          <IconButton size="small" onClick={() => handleViewGame(game.id, game.status)}>
                            <VisibilityIcon fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                      <Tooltip title="Restart">
                        <span>
                          <IconButton size="small" onClick={() => handleRestart(game.id)}>
                            <RestartAltIcon fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                      <Tooltip title="Edit">
                        <span>
                          <IconButton size="small" onClick={() => handleEdit(game)}>
                            <EditIcon fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <span>
                          <IconButton size="small" onClick={() => handleDelete(game.id, game.name)}>
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </span>
                      </Tooltip>
                    </Stack>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Paper>
  );
};

export default GroupGameConfigPanel;
