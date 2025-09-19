import React, { useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  IconButton,
  Paper,
  Stack,
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
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Flag as FinishIcon,
  SkipNext as NextRoundIcon,
  Visibility as ViewIcon,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import { useNavigate } from 'react-router-dom';
import { format } from 'date-fns';
import { mixedGameApi } from '../../services/api';

const formatDate = (value) => {
  if (!value) return '—';
  try {
    return format(new Date(value), 'MMM d, yyyy HH:mm');
  } catch (error) {
    return String(value);
  }
};

const statusLabel = (status = '') => (status || 'unknown').replace(/_/g, ' ');

const statusColor = (status = '') => {
  const normalized = String(status).toLowerCase();
  if (normalized === 'created') return 'default';
  if (normalized === 'in_progress') return 'info';
  if (normalized === 'completed') return 'success';
  if (normalized === 'paused') return 'warning';
  if (normalized === 'failed' || normalized === 'error') return 'error';
  return 'default';
};

const GroupGameSupervisionPanel = ({
  games = [],
  loading = false,
  error = null,
  onRefresh,
  groupId = null,
  currentUserId = null,
}) => {
  const navigate = useNavigate();
  const { enqueueSnackbar } = useSnackbar();
  const [actionState, setActionState] = useState({});

  const supervisedGames = useMemo(() => {
    if (!Array.isArray(games)) {
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

  const setGameActionState = (gameId, state) => {
    setActionState((prev) => ({ ...prev, [gameId]: state }));
  };

  const runAction = async (gameId, action, apiCall, successMessage) => {
    setGameActionState(gameId, action);
    try {
      await apiCall(gameId);
      enqueueSnackbar(successMessage, { variant: 'success' });
      if (onRefresh) {
        await onRefresh();
      }
    } catch (err) {
      const detail = err?.response?.data?.detail || err?.message || 'Action failed';
      enqueueSnackbar(detail, { variant: 'error' });
    } finally {
      setGameActionState(gameId, null);
    }
  };

  const handleStart = (gameId) =>
    runAction(gameId, 'start', mixedGameApi.startGame, 'Game started');
  const handleStop = (gameId) => runAction(gameId, 'stop', mixedGameApi.stopGame, 'Game stopped');
  const handleNextRound = (gameId) =>
    runAction(gameId, 'next_round', mixedGameApi.nextRound, 'Advanced to next round');
  const handleFinish = (gameId) =>
    runAction(gameId, 'finish', mixedGameApi.finishGame, 'Game marked as finished');

  const renderActions = (game) => {
    const status = String(game.status || '').toLowerCase();
    const mode = String(game.progression_mode || game?.config?.progression_mode || 'supervised').toLowerCase();
    const busy = Boolean(actionState[game.id]);
    const viewTarget = status === 'completed' ? `/games/${game.id}/report` : `/games/${game.id}`;

    if (status === 'completed') {
      return (
        <Tooltip title="View game">
          <span>
            <IconButton color="primary" onClick={() => navigate(viewTarget)}>
              <ViewIcon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
      );
    }

    if (status === 'created' || status === 'paused') {
      return (
        <Stack direction="row" spacing={1}>
          <Tooltip title="View game">
            <span>
              <IconButton color="primary" onClick={() => navigate(viewTarget)}>
                <ViewIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Button
            size="small"
            variant="contained"
            color="success"
            startIcon={<StartIcon fontSize="small" />}
            onClick={() => handleStart(game.id)}
            disabled={busy}
          >
            {busy ? 'Starting…' : 'Start'}
          </Button>
        </Stack>
      );
    }

    if (status === 'in_progress') {
      const autoChip = (
        <Chip
          label="Auto"
          size="small"
          color="info"
          variant="outlined"
        />
      );
      return (
        <Stack direction="row" spacing={1} alignItems="center">
          <Tooltip title="View game">
            <span>
              <IconButton color="primary" onClick={() => navigate(viewTarget)}>
                <ViewIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          {mode === 'unsupervised' ? (
            autoChip
          ) : (
            <Button
              size="small"
              variant="outlined"
              startIcon={<NextRoundIcon fontSize="small" />}
              onClick={() => handleNextRound(game.id)}
              disabled={busy}
            >
              {busy && actionState[game.id] === 'next_round' ? 'Advancing…' : 'Next Round'}
            </Button>
          )}
          <Button
            size="small"
            variant="contained"
            color="warning"
            startIcon={<FinishIcon fontSize="small" />}
            onClick={() => handleFinish(game.id)}
            disabled={busy}
          >
            {busy && actionState[game.id] === 'finish' ? 'Finishing…' : 'Finish'}
          </Button>
          <Button
            size="small"
            variant="outlined"
            color="error"
            startIcon={<StopIcon fontSize="small" />}
            onClick={() => handleStop(game.id)}
            disabled={busy}
          >
            {busy && actionState[game.id] === 'stop' ? 'Stopping…' : 'Stop'}
          </Button>
        </Stack>
      );
    }

    return (
          <Tooltip title="View game">
            <span>
              <IconButton color="primary" onClick={() => navigate(viewTarget)}>
                <ViewIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
    );
  };

  return (
    <Paper elevation={0} sx={{ p: 3 }}>
      <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" alignItems={{ xs: 'stretch', md: 'center' }} spacing={2} mb={3}>
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            Game Supervision
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Monitor live sessions and orchestrate progress across your group's games.
          </Typography>
        </Box>
        {onRefresh && (
          <Button variant="outlined" onClick={onRefresh} disabled={loading}>
            Refresh
          </Button>
        )}
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
      ) : supervisedGames.length === 0 ? (
        <Box textAlign="center" py={6}>
          <Typography variant="body1" color="text.secondary" gutterBottom>
            There are no games to supervise right now.
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Create a new mixed game or wait for a session to begin.
          </Typography>
        </Box>
      ) : (
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Round</TableCell>
                <TableCell>Last Updated</TableCell>
                <TableCell align="right">Controls</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {supervisedGames.map((game) => (
                <TableRow hover key={game.id}>
                  <TableCell>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                      {game.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {game.description || 'No description provided'}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip size="small" color={statusColor(game.status)} label={statusLabel(game.status)} />
                  </TableCell>
                  <TableCell>
                    {game.current_round ?? 0} / {game.max_rounds ?? '—'}
                  </TableCell>
                  <TableCell>{formatDate(game.updated_at)}</TableCell>
                  <TableCell align="right">{renderActions(game)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Paper>
  );
};

export default GroupGameSupervisionPanel;
