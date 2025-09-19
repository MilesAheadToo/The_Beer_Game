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
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { format } from 'date-fns';

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

  const handleViewGame = (gameId) => {
    navigate(`/games/${gameId}`);
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
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Mode</TableCell>
                <TableCell>Current Round</TableCell>
                <TableCell>Max Rounds</TableCell>
                <TableCell>Last Updated</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredGames.map((game) => (
                <TableRow hover key={game.id}>
                  <TableCell>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                      {game.name}
                    </Typography>
                    {game.description && (
                      <Typography variant="body2" color="text.secondary" noWrap>
                        {game.description}
                      </Typography>
                    )}
                  </TableCell>
                  <TableCell>
                    <Chip
                      size="small"
                      color={statusColor(game.status)}
                      label={(game.status || '').replace(/_/g, ' ') || 'Unknown'}
                    />
                  </TableCell>
                  <TableCell>
                    <Chip
                      size="small"
                      variant="outlined"
                      color={String((game.progression_mode || 'supervised')).toLowerCase() === 'unsupervised' ? 'info' : 'default'}
                      label={String(game.progression_mode || game?.config?.progression_mode || 'supervised').replace(/_/g, ' ').replace(/^./, (s) => s.toUpperCase())}
                    />
                  </TableCell>
                  <TableCell>{game.current_round ?? 0}</TableCell>
                  <TableCell>{game.max_rounds ?? '—'}</TableCell>
                  <TableCell>{formatDate(game.updated_at)}</TableCell>
                  <TableCell align="right">
                    <Button size="small" variant="outlined" onClick={() => handleViewGame(game.id)}>
                      View
                    </Button>
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
