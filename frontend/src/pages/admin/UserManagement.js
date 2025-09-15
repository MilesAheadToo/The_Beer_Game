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
  FormControl,
  FormHelperText,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material';
import { Add, Delete, Edit } from '@mui/icons-material';
import { toast } from 'react-toastify';
import { useAuth } from '../../contexts/AuthContext';
import { api, mixedGameApi } from '../../services/api';
import { normalizeRoles } from '../../utils/authUtils';

const USER_TYPE_OPTIONS = [
  { value: 'player', label: 'Player' },
  { value: 'group_admin', label: 'Group Admin' },
  { value: 'system_admin', label: 'System Admin' },
];

const USER_TYPE_LABELS = {
  player: 'Player',
  group_admin: 'Group Admin',
  system_admin: 'System Admin',
};

const DEFAULT_FORM = {
  username: '',
  email: '',
  password: '',
  userType: 'player',
  groupId: '',
};

const DEFAULT_REPLACEMENT_PROMPT = {
  open: false,
  user: null,
  options: [],
  selected: '',
  message: '',
};

const getUserType = (user) => {
  if (!user) return 'player';
  const normalizedRoles = normalizeRoles(user.roles || []);
  if (user.is_superuser || normalizedRoles.includes('systemadmin')) {
    return 'system_admin';
  }
  if (normalizedRoles.includes('groupadmin') || normalizedRoles.includes('admin')) {
    return 'group_admin';
  }
  return 'player';
};

const getTypeChipColor = (type) => {
  switch (type) {
    case 'system_admin':
      return 'secondary';
    case 'group_admin':
      return 'primary';
    default:
      return 'success';
  }
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

function UserManagement() {
  const [users, setUsers] = useState([]);
  const [groups, setGroups] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingUser, setEditingUser] = useState(null);
  const [form, setForm] = useState({ ...DEFAULT_FORM });
  const [saving, setSaving] = useState(false);
  const [replacementPrompt, setReplacementPrompt] = useState({ ...DEFAULT_REPLACEMENT_PROMPT });

  const navigate = useNavigate();
  const { isGroupAdmin } = useAuth();

  const groupMap = useMemo(() => {
    const map = {};
    (groups || []).forEach((group) => {
      map[group.id] = group.name;
    });
    return map;
  }, [groups]);

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

  const loadUsers = useCallback(async () => {
    try {
      const response = await api.get('/api/v1/users');
      const data = Array.isArray(response.data) ? response.data : [];
      setUsers(data);
      return data;
    } catch (error) {
      console.error('Error loading users:', error);
      setUsers([]);
      throw error;
    }
  }, []);

  useEffect(() => {
    if (!isGroupAdmin) {
      navigate('/unauthorized');
      return;
    }

    const fetchAll = async () => {
      setIsLoading(true);
      try {
        await mixedGameApi.health();
        await Promise.all([loadGroups(), loadUsers()]);
      } catch (error) {
        console.error('Error loading user management data:', error);
        toast.error('Failed to load user information');
      } finally {
        setIsLoading(false);
      }
    };

    fetchAll();
  }, [isGroupAdmin, navigate, loadGroups, loadUsers]);

  const handleOpenModal = (user = null) => {
    if (user) {
      setEditingUser(user);
      setForm({
        username: user.username || '',
        email: user.email || '',
        password: '',
        userType: getUserType(user),
        groupId: user.group_id ? String(user.group_id) : '',
      });
    } else {
      setEditingUser(null);
      setForm({ ...DEFAULT_FORM });
    }
    setDialogOpen(true);
  };

  const handleCloseModal = () => {
    setDialogOpen(false);
    setEditingUser(null);
    setForm({ ...DEFAULT_FORM });
  };

  const handleTypeChange = (event) => {
    const value = event.target.value;
    setForm((prev) => ({
      ...prev,
      userType: value,
      groupId: value === 'system_admin' ? '' : prev.groupId,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    const trimmedUsername = form.username.trim();
    const trimmedEmail = form.email.trim();
    const requiresGroup = form.userType !== 'system_admin';

    if (!trimmedUsername || !trimmedEmail) {
      toast.error('Username and email are required.');
      return;
    }

    if (requiresGroup && !form.groupId) {
      toast.error('Please select a group for this user.');
      return;
    }

    const payload = {
      username: trimmedUsername,
      email: trimmedEmail,
      user_type: form.userType,
      group_id: requiresGroup ? Number(form.groupId) : null,
    };

    if (!editingUser || form.password) {
      if (!editingUser && !form.password) {
        toast.error('Password is required for new users.');
        return;
      }
      payload.password = form.password;
    }

    setSaving(true);
    try {
      if (editingUser) {
        await api.put(`/api/v1/users/${editingUser.id}`, payload);
        toast.success('User updated successfully');
      } else {
        await api.post('/api/v1/users', payload);
        toast.success('User created successfully');
      }

      handleCloseModal();
      await Promise.all([loadUsers(), loadGroups()]);
    } catch (error) {
      const message = parseErrorMessage(error, 'Failed to save user');
      toast.error(message);
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteUser = async (user) => {
    if (!user) return;
    const confirmMessage = `Are you sure you want to delete ${user.username || 'this user'}?`;
    if (!window.confirm(confirmMessage)) return;

    try {
      await api.delete(`/api/v1/users/${user.id}`);
      toast.success('User deleted');
      await Promise.all([loadUsers(), loadGroups()]);
    } catch (error) {
      const detail = error?.response?.data?.detail;
      if (detail && typeof detail === 'object' && detail.code === 'replacement_required') {
        setReplacementPrompt({
          open: true,
          user,
          options: Array.isArray(detail.candidates) ? detail.candidates : [],
          selected: '',
          message: detail.message || 'Select a group admin to promote before deleting this system admin.',
        });
        return;
      }

      if (detail && typeof detail === 'object' && detail.code === 'no_group_admin_available') {
        toast.error(detail.message || 'Cannot delete the last system admin without another admin.');
        return;
      }

      const message = parseErrorMessage(error, 'Failed to delete user');
      toast.error(message);
    }
  };

  const closeReplacementPrompt = () => {
    setReplacementPrompt({ ...DEFAULT_REPLACEMENT_PROMPT });
  };

  const handleConfirmReplacement = async () => {
    if (!replacementPrompt.user) return;
    if (!replacementPrompt.selected) {
      toast.error('Please select a replacement system admin.');
      return;
    }

    try {
      await api.delete(`/api/v1/users/${replacementPrompt.user.id}`, {
        params: { replacement_admin_id: replacementPrompt.selected },
      });
      toast.success('User deleted and replacement promoted to system admin');
      closeReplacementPrompt();
      await Promise.all([loadUsers(), loadGroups()]);
    } catch (error) {
      const message = parseErrorMessage(error, 'Failed to delete user');
      toast.error(message);
    }
  };

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box p={2}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h4">User Management</Typography>
        <Button variant="contained" startIcon={<Add />} onClick={() => handleOpenModal(null)}>
          Add User
        </Button>
      </Box>

      <Table>
        <TableHead>
          <TableRow>
            <TableCell>User</TableCell>
            <TableCell>Email</TableCell>
            <TableCell>Group</TableCell>
            <TableCell>Type</TableCell>
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {users.length === 0 ? (
            <TableRow>
              <TableCell colSpan={5} align="center">
                <Typography variant="body2" color="text.secondary">
                  No users found.
                </Typography>
              </TableCell>
            </TableRow>
          ) : (
            users.map((user) => {
              const type = getUserType(user);
              return (
                <TableRow key={user.id} hover>
                  <TableCell>{user.username}</TableCell>
                  <TableCell>{user.email}</TableCell>
                  <TableCell>{groupMap[user.group_id] || '—'}</TableCell>
                  <TableCell>
                    <Chip
                      size="small"
                      label={USER_TYPE_LABELS[type] || 'User'}
                      color={getTypeChipColor(type)}
                    />
                  </TableCell>
                  <TableCell align="right">
                    <IconButton onClick={() => handleOpenModal(user)} size="small" color="primary">
                      <Edit fontSize="small" />
                    </IconButton>
                    <IconButton onClick={() => handleDeleteUser(user)} size="small" color="error">
                      <Delete fontSize="small" />
                    </IconButton>
                  </TableCell>
                </TableRow>
              );
            })
          )}
        </TableBody>
      </Table>

      <Dialog open={dialogOpen} onClose={handleCloseModal} maxWidth="sm" fullWidth>
        <DialogTitle>{editingUser ? 'Edit User' : 'Create User'}</DialogTitle>
        <DialogContent dividers>
          <Box
            component="form"
            id="user-management-form"
            onSubmit={handleSubmit}
            sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}
          >
            <TextField
              label="Username"
              value={form.username}
              onChange={(event) => setForm((prev) => ({ ...prev, username: event.target.value }))}
              fullWidth
              required
            />
            <TextField
              label="Email"
              type="email"
              value={form.email}
              onChange={(event) => setForm((prev) => ({ ...prev, email: event.target.value }))}
              fullWidth
              required
            />
            <TextField
              label={editingUser ? 'Password (leave blank to keep current)' : 'Password'}
              type="password"
              value={form.password}
              onChange={(event) => setForm((prev) => ({ ...prev, password: event.target.value }))}
              fullWidth
              required={!editingUser}
            />
            <FormControl fullWidth>
              <InputLabel id="user-type-label">User Type</InputLabel>
              <Select
                labelId="user-type-label"
                label="User Type"
                value={form.userType}
                onChange={handleTypeChange}
              >
                {USER_TYPE_OPTIONS.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            {form.userType !== 'system_admin' && (
              <FormControl fullWidth>
                <InputLabel id="group-select-label">Group</InputLabel>
                <Select
                  labelId="group-select-label"
                  label="Group"
                  value={form.groupId}
                  onChange={(event) => setForm((prev) => ({ ...prev, groupId: event.target.value }))}
                  required
                >
                  <MenuItem value="">
                    <em>Select a group</em>
                  </MenuItem>
                  {groups.map((group) => (
                    <MenuItem key={group.id} value={String(group.id)}>
                      {group.name}
                    </MenuItem>
                  ))}
                </Select>
                {groups.length === 0 && (
                  <FormHelperText error>
                    No groups available. Create a group before adding players or group admins.
                  </FormHelperText>
                )}
              </FormControl>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseModal} disabled={saving}>
            Cancel
          </Button>
          <Button type="submit" form="user-management-form" variant="contained" disabled={saving}>
            {saving ? 'Saving…' : 'Save'}
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={replacementPrompt.open} onClose={closeReplacementPrompt} maxWidth="xs" fullWidth>
        <DialogTitle>Promote a Group Admin</DialogTitle>
        <DialogContent dividers>
          <Typography variant="body2" color="text.secondary" mb={2}>
            {replacementPrompt.message || 'Select a group admin to promote to system admin before deleting this user.'}
          </Typography>
          <FormControl fullWidth>
            <InputLabel id="replacement-admin-label">Replacement System Admin</InputLabel>
            <Select
              labelId="replacement-admin-label"
              label="Replacement System Admin"
              value={replacementPrompt.selected}
              onChange={(event) =>
                setReplacementPrompt((prev) => ({ ...prev, selected: event.target.value }))
              }
            >
              <MenuItem value="">
                <em>Select a group admin</em>
              </MenuItem>
              {replacementPrompt.options.map((option) => (
                <MenuItem key={option.id} value={String(option.id)}>
                  {option.username || option.email}
                  {option.group_name ? ` — ${option.group_name}` : ''}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={closeReplacementPrompt}>Cancel</Button>
          <Button onClick={handleConfirmReplacement} variant="contained">
            Promote & Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default UserManagement;
