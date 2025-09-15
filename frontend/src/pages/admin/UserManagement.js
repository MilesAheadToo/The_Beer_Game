import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Chip,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material';
import { Add as AddIcon, Delete as DeleteIcon, Edit as EditIcon } from '@mui/icons-material';
import { toast } from 'react-toastify';
import { api } from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';
import { isSystemAdmin as isSystemAdminUser, normalizeRoles } from '../../utils/authUtils';

const BASE_FORM = {
  username: '',
  email: '',
  password: '',
};

const getUserType = (user) => {
  if (!user) return 'player';
  const roles = normalizeRoles(user.roles || []);
  if (user.is_superuser || roles.includes('systemadmin')) {
    return 'system_admin';
  }
  if (roles.includes('groupadmin') || roles.includes('admin')) {
    return 'group_admin';
  }
  return 'player';
};

const parseErrorMessage = (error, fallback) => {
  const detail = error?.response?.data?.detail;
  if (!detail) return fallback;
  if (typeof detail === 'string') return detail;
  if (typeof detail === 'object') {
    return detail.message || fallback;
  }
  return fallback;
};

function GroupPlayerManagement() {
  const navigate = useNavigate();
  const { isGroupAdmin, user } = useAuth();
  const systemAdmin = isSystemAdminUser(user);
  const rawGroupId = user?.group_id;
  const parsedGroupId = typeof rawGroupId === 'number' ? rawGroupId : Number(rawGroupId);
  const groupId = Number.isFinite(parsedGroupId) ? parsedGroupId : null;

  const [players, setPlayers] = useState([]);
  const [groups, setGroups] = useState([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingUser, setEditingUser] = useState(null);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState({ ...BASE_FORM });

  useEffect(() => {
    if (!isGroupAdmin) {
      navigate('/unauthorized');
      return;
    }
    if (systemAdmin) {
      navigate('/system/users', { replace: true });
    }
  }, [isGroupAdmin, navigate, systemAdmin]);

  const loadGroups = useCallback(async () => {
    try {
      const response = await api.get('/api/v1/groups');
      const data = Array.isArray(response.data) ? response.data : [];
      setGroups(data);
      return data;
    } catch (error) {
      console.error('Error loading groups:', error);
      setGroups([]);
      throw error;
    }
  }, []);

  const loadPlayers = useCallback(async () => {
    if (!groupId) {
      setPlayers([]);
      return [];
    }

    try {
      const response = await api.get('/api/v1/users', {
        params: { limit: 250, user_type: 'player' },
      });
      const data = Array.isArray(response.data) ? response.data : [];
      const filtered = data.filter(
        (item) => getUserType(item) === 'player' && item.group_id === groupId,
      );
      setPlayers(filtered);
      return filtered;
    } catch (error) {
      console.error('Error loading players:', error);
      setPlayers([]);
      throw error;
    }
  }, [groupId]);

  useEffect(() => {
    if (!isGroupAdmin || systemAdmin) {
      return;
    }

    if (!groupId) {
      return;
    }

    const fetchAll = async () => {
      setLoading(true);
      try {
        await Promise.all([loadGroups(), loadPlayers()]);
      } catch (error) {
        toast.error('Failed to load player information');
      } finally {
        setLoading(false);
      }
    };

    fetchAll();
  }, [groupId, isGroupAdmin, systemAdmin, loadGroups, loadPlayers]);

  const groupMap = useMemo(() => {
    const map = {};
    (groups || []).forEach((group) => {
      map[group.id] = group.name;
    });
    return map;
  }, [groups]);

  const handleOpenDialog = () => {
    setEditingUser(null);
    setForm({ ...BASE_FORM });
    setDialogOpen(true);
  };

  const handleCloseDialog = () => {
    if (saving) return;
    setDialogOpen(false);
    setEditingUser(null);
    setForm({ ...BASE_FORM });
  };

  const handleEditUser = (player) => {
    setEditingUser(player);
    setForm({
      username: player.username || '',
      email: player.email || '',
      password: '',
    });
    setDialogOpen(true);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const trimmedUsername = form.username.trim();
    const trimmedEmail = form.email.trim();

    if (!trimmedUsername || !trimmedEmail) {
      toast.error('Username and email are required.');
      return;
    }

    if (!groupId) {
      toast.error('Your account is not linked to a group. Please contact your system administrator.');
      return;
    }

    const payload = {
      username: trimmedUsername,
      email: trimmedEmail,
    };

    if (!editingUser) {
      if (!form.password.trim()) {
        toast.error('Password is required for new players.');
        return;
      }
      payload.password = form.password;
      payload.group_id = groupId;
      payload.user_type = 'player';
    } else if (form.password.trim()) {
      payload.password = form.password.trim();
    }

    setSaving(true);
    try {
      if (editingUser) {
        await api.put(`/api/v1/users/${editingUser.id}`, payload);
        toast.success('Player updated successfully');
      } else {
        await api.post('/api/v1/users', payload);
        toast.success('Player created successfully');
      }

      handleCloseDialog();
      await loadPlayers();
    } catch (error) {
      const message = parseErrorMessage(error, 'Failed to save player');
      toast.error(message);
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteUser = async (player) => {
    if (!player) return;
    const confirmMessage = `Are you sure you want to delete ${player.username || 'this player'}?`;
    if (!window.confirm(confirmMessage)) return;

    try {
      await api.delete(`/api/v1/users/${player.id}`);
      toast.success('Player deleted');
      await loadPlayers();
    } catch (error) {
      const message = parseErrorMessage(error, 'Failed to delete player');
      toast.error(message);
    }
  };

  if (!isGroupAdmin || systemAdmin) {
    return null;
  }

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ maxWidth: '1100px', mx: 'auto', my: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" component="h1" sx={{ fontWeight: 600, color: 'text.primary' }}>
            Player Management
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Manage the players within your group.
          </Typography>
        </Box>
        <Button variant="contained" startIcon={<AddIcon />} onClick={handleOpenDialog}>
          Add Player
        </Button>
      </Box>

      <Box sx={{ backgroundColor: 'background.paper', borderRadius: 2, boxShadow: 3, overflow: 'hidden' }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell sx={{ fontWeight: 600 }}>Username</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Email</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Group</TableCell>
              <TableCell sx={{ fontWeight: 600 }}>Type</TableCell>
              <TableCell align="right" sx={{ fontWeight: 600 }}>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {players.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} align="center">
                  <Typography color="text.secondary">No players found for your group yet.</Typography>
                </TableCell>
              </TableRow>
            ) : (
              players.map((player) => {
                const type = getUserType(player);
                return (
                  <TableRow key={player.id} hover>
                    <TableCell>{player.username}</TableCell>
                    <TableCell>{player.email}</TableCell>
                    <TableCell>{groupMap[player.group_id] || '—'}</TableCell>
                    <TableCell>
                      <Chip label={type === 'player' ? 'Player' : 'User'} color={type === 'player' ? 'success' : 'default'} size="small" />
                    </TableCell>
                    <TableCell align="right">
                      <IconButton color="primary" onClick={() => handleEditUser(player)}>
                        <EditIcon fontSize="small" />
                      </IconButton>
                      <IconButton color="error" onClick={() => handleDeleteUser(player)}>
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </Box>

      <Dialog open={dialogOpen} onClose={handleCloseDialog} fullWidth maxWidth="sm">
        <DialogTitle>{editingUser ? 'Edit Player' : 'Add Player'}</DialogTitle>
        <form onSubmit={handleSubmit}>
          <DialogContent dividers>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <TextField
                label="Username"
                value={form.username}
                onChange={(event) => setForm((prev) => ({ ...prev, username: event.target.value }))}
                required
                fullWidth
              />
              <TextField
                label="Email"
                type="email"
                value={form.email}
                onChange={(event) => setForm((prev) => ({ ...prev, email: event.target.value }))}
                required
                fullWidth
              />
              <TextField
                label={editingUser ? 'Password (leave blank to keep current)' : 'Password'}
                type="password"
                value={form.password}
                onChange={(event) => setForm((prev) => ({ ...prev, password: event.target.value }))}
                required={!editingUser}
                fullWidth
              />
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseDialog} disabled={saving}>
              Cancel
            </Button>
            <Button type="submit" variant="contained" disabled={saving}>
              {saving ? 'Saving…' : editingUser ? 'Save Changes' : 'Add Player'}
            </Button>
          </DialogActions>
        </form>
      </Dialog>
    </Box>
  );
}

export default GroupPlayerManagement;
