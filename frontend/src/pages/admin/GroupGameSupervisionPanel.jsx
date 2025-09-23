import React, { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
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
  LinearProgress,
  Divider,
} from '@mui/material';
import {
  PlayArrow as StartIcon,
  Stop as StopIcon,
  Flag as FinishIcon,
  SkipNext as NextRoundIcon,
  Visibility as ViewIcon,
  CheckCircleOutline as CompleteIcon,
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
  const [autoProgress, setAutoProgress] = useState(null);

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
      const result = await apiCall(gameId);
      enqueueSnackbar(successMessage, { variant: 'success' });
      if (onRefresh) {
        await onRefresh();
      }
      return result;
    } catch (err) {
      const detail = err?.response?.data?.detail || err?.message || 'Action failed';
      enqueueSnackbar(detail, { variant: 'error' });
      return undefined;
    } finally {
      setGameActionState(gameId, null);
    }
  };

  const handleStart = async (game) => {
    if (!game) return;
    const result = await runAction(game.id, 'start', mixedGameApi.startGame, 'Game started');
    if (!result) {
      return;
    }

    const mode = String(
      result?.progression_mode ||
        game.progression_mode ||
        game?.config?.progression_mode ||
        'supervised',
    ).toLowerCase();

    if (mode === 'unsupervised') {
      setAutoProgress({
        gameId: game.id,
        name: game.name,
        currentRound: result?.current_round ?? game.current_round ?? 0,
        maxRounds: result?.max_rounds ?? game.max_rounds ?? 0,
        status: result?.status ?? game.status,
        lastUpdated: new Date().toISOString(),
        done: false,
        error: null,
        history: Array.isArray(result?.config?.history) ? result.config.history : [],
      });
    }
  };
  const handleStop = (gameId) => runAction(gameId, 'stop', mixedGameApi.stopGame, 'Game stopped');
  const handleNextRound = (gameId) =>
    runAction(gameId, 'next_round', mixedGameApi.nextRound, 'Advanced to next round');
  const handleFinish = (gameId) =>
    runAction(gameId, 'finish', mixedGameApi.finishGame, 'Game marked as finished');

  const monitoringGameId = autoProgress?.gameId;
  const monitoringDone = autoProgress?.done;

  useEffect(() => {
    if (!monitoringGameId || monitoringDone) {
      return undefined;
    }

    let cancelled = false;

    const poll = async () => {
      try {
        const state = await mixedGameApi.getGameState(monitoringGameId);
        const gameData = state?.game || {};
        if (cancelled) return;

        const statusRaw = String(gameData?.status || '').toLowerCase();
        const currentRound = state?.round ?? gameData?.current_round ?? 0;
        const maxRoundsFromData = gameData?.max_rounds ?? 0;
        const history = Array.isArray(state?.history) ? state.history : [];

        let finished = false;
        setAutoProgress((prev) => {
          if (!prev || prev.gameId !== monitoringGameId) {
            return prev;
          }

          const nextMaxRounds = maxRoundsFromData || prev.maxRounds || 0;
          const done =
            statusRaw === 'completed' ||
            statusRaw === 'finished' ||
            (nextMaxRounds > 0 && currentRound >= nextMaxRounds);
          if (done && !prev.done) {
            finished = true;
          }

          return {
            ...prev,
            currentRound,
            maxRounds: nextMaxRounds,
            status: gameData?.status ?? prev.status,
            lastUpdated: new Date().toISOString(),
            error: null,
            done,
            history,
          };
        });

        if (finished && onRefresh) {
          await onRefresh();
        }
      } catch (err) {
        if (cancelled) return;
        const detail = err?.response?.data?.detail || err?.message || 'Unable to update progress right now';
        setAutoProgress((prev) => {
          if (!prev || prev.gameId !== monitoringGameId) {
            return prev;
          }
          return {
            ...prev,
            error: detail,
            lastUpdated: new Date().toISOString(),
          };
        });
      }
    };

    const interval = setInterval(poll, 1500);
    poll();

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [monitoringGameId, monitoringDone, onRefresh]);

  useEffect(() => {
    if (!autoProgress?.done) {
      return undefined;
    }

    const timeout = setTimeout(() => {
      setAutoProgress(null);
    }, 1200);

    return () => clearTimeout(timeout);
  }, [autoProgress?.done]);

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
        <Stack
          direction="row"
          spacing={1}
          useFlexGap
          alignItems="center"
          sx={{ flexWrap: 'wrap', justifyContent: { xs: 'flex-start', md: 'flex-end' } }}
        >
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
            onClick={() => handleStart(game)}
            disabled={busy}
          >
            {busy ? 'Starting…' : 'Start'}
          </Button>
        </Stack>
      );
    }

    if (status === 'in_progress') {
      const autoChip = (
        <Chip label="Auto" size="small" color="info" variant="outlined" />
      );
      return (
        <Stack
          direction="row"
          spacing={1}
          useFlexGap
          alignItems="center"
          sx={{ flexWrap: 'wrap', justifyContent: { xs: 'flex-start', md: 'flex-end' } }}
        >
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
    <>
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
          <TableContainer sx={{ overflowX: 'visible' }}>
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
                    <TableCell sx={{ minWidth: 140 }}>
                      <Stack spacing={0.5}>
                        <Typography variant="body2">
                          {game.current_round ?? 0} / {game.max_rounds ?? '—'}
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={(() => {
                            const current = Number(game.current_round ?? 0);
                            const total = Number(game.max_rounds ?? 0);
                            if (!total || Number.isNaN(total) || total <= 0) {
                              return 0;
                            }
                            const pct = (current / total) * 100;
                            return Math.max(0, Math.min(100, pct));
                          })()}
                          sx={{ height: 6, borderRadius: 3 }}
                        />
                      </Stack>
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

    <Dialog
      open={Boolean(autoProgress)}
      fullWidth
      maxWidth="xs"
      onClose={() => setAutoProgress(null)}
    >
      <DialogTitle>Running unsupervised game</DialogTitle>
      <DialogContent>
        {autoProgress && (
          <Stack spacing={2} alignItems="center" sx={{ py: 1 }}>
            {autoProgress.done ? (
              <CompleteIcon color="success" fontSize="large" />
            ) : (
              <CircularProgress color="primary" />
            )}
            <Box textAlign="center">
              <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                {autoProgress.name}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Round {autoProgress.currentRound ?? 0} / {autoProgress.maxRounds || '—'}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Status: {statusLabel(autoProgress.status)}
              </Typography>
            </Box>
            {autoProgress.maxRounds > 0 && (
              <Box sx={{ width: '100%' }}>
                <LinearProgress
                  variant="determinate"
                  sx={{ height: 6, borderRadius: 3 }}
                  value={(() => {
                    const total = Number(autoProgress.maxRounds);
                    const current = Number(autoProgress.currentRound ?? 0);
                    if (!total || Number.isNaN(total) || total <= 0) {
                      return 0;
                    }
                    const pct = (current / total) * 100;
                    return Math.max(0, Math.min(100, pct));
                  })()}
                />
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                  {(() => {
                    const total = Number(autoProgress.maxRounds);
                    const current = Number(autoProgress.currentRound ?? 0);
                    if (!total || Number.isNaN(total) || total <= 0) {
                      return 'Progress unavailable';
                    }
                    const pct = Math.max(0, Math.min(100, Math.round((current / total) * 100)));
                    return `${pct}% complete`;
                  })()}
                </Typography>
              </Box>
            )}
            {autoProgress.error && (
              <Typography variant="body2" color="error" align="center">
                {autoProgress.error}
              </Typography>
            )}
            {!autoProgress.done && (
              <Typography variant="body2" color="text.secondary" align="center">
                We&apos;re advancing each round automatically. You can close this dialog at any time.
              </Typography>
            )}
            {autoProgress.done && (
              <Typography variant="body2" align="center" sx={{ color: 'success.main' }}>
                All rounds complete. Preparing summary…
              </Typography>
            )}
            {(() => {
              const history = Array.isArray(autoProgress.history) ? autoProgress.history : [];
              if (!history.length) {
                return (
                  <Typography variant="caption" color="text.secondary" align="center">
                    Waiting for first round results…
                  </Typography>
                );
              }
              const latest = history[history.length - 1];
              const orders = latest?.orders && typeof latest.orders === 'object' ? latest.orders : {};
              const entries = Object.entries(orders);
              if (!entries.length) {
                return null;
              }
              return (
                <Box sx={{ width: '100%' }}>
                  <Divider sx={{ my: 1 }} />
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                    Latest orders (Round {latest?.round ?? autoProgress.currentRound ?? '—'})
                  </Typography>
                  <Stack spacing={0.75} sx={{ width: '100%' }}>
                    {entries.map(([role, details]) => {
                      const quantity = details?.quantity ?? details?.order ?? details?.amount ?? '—';
                      const comment = details?.comment;
                      return (
                        <Box
                          key={role}
                          sx={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'flex-start',
                            gap: 1,
                          }}
                        >
                          <Typography variant="body2" sx={{ fontWeight: 600, textTransform: 'capitalize' }}>
                            {role.replace(/_/g, ' ')}
                          </Typography>
                          <Box textAlign="right">
                            <Typography variant="body2">{`${quantity} units`}</Typography>
                            {comment && (
                              <Typography variant="caption" color="text.secondary">
                                {comment}
                              </Typography>
                            )}
                          </Box>
                        </Box>
                      );
                    })}
                  </Stack>
                </Box>
              );
            })()}
          </Stack>
        )}
      </DialogContent>
    </Dialog>
  </>
  );
};

export default GroupGameSupervisionPanel;
