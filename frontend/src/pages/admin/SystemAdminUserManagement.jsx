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
import { Add as AddIcon, Delete as DeleteIcon, Edit as EditIcon } from '@mui/icons-material';
import { toast } from 'react-toastify';
import { api } from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';
import { isSystemAdmin as isSystemAdminUser, getUserType as resolveUserType } from '../../utils/authUtils';

const BASE_FORM = {
  username: '',
  email: '',
  password: '',
  groupId: '',
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

function SystemAdminUserManagement() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const systemAdmin = isSystemAdminUser(user);

  const [admins, setAdmins] = useState([]);
  const [groups, setGroups] = useState([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingUser, setEditingUser] = useState(null);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState({ ...BASE_FORM });

  useEffect(() => {
    if (!systemAdmin) {
      navigate('/unauthorized');
    }
  }, [systemAdmin, navigate]);

  const loadGroups = useCallback(async () => {
    try {
      const response = await api.get('/groups');
      const data = Array.isArray(response.data) ? response.data : [];
      setGroups(data);
      return data;
    } catch (error) {
      console.error('Error loading groups:', error);
      setGroups([]);
      throw error;
    }
  }, []);

  const loadAdmins = useCallback(async () => {
    try {
      const response = await api.get('/users', { params: { user_type: 'GroupAdmin', limit: 250 } });
      const data = Array.isArray(response.data) ? response.data : [];
      const filtered = data.filter((item) => resolveUserType(item) === 'groupadmin');
      setAdmins(filtered);
      return filtered;
    } catch (error) {
      console.error('Error loading group administrators:', error);
      setAdmins([]);
      throw error;
    }
  }, []);

  useEffect(() => {
    if (!systemAdmin) {
      return;
    }

    const fetchAll = async () => {
      setLoading(true);
      try {
        await Promise.all([loadGroups(), loadAdmins()]);
      } catch (error) {
        toast.error('Failed to load user information');
      } finally {
        setLoading(false);
      }
    };

    fetchAll();
  }, [systemAdmin, loadGroups, loadAdmins]);

  const groupMap = useMemo(() => {
    const map = {};
    (groups || []).forEach((group) => {
      map[group.id] = group.name;
    });
    return map;
  }, [groups]);

  const handleOpenDialog = () => {
    const defaultGroupId = groups.length === 1 ? String(groups[0].id) : '';
    setEditingUser(null);
    setForm({ ...BASE_FORM, groupId: defaultGroupId });
    setDialogOpen(true);
  };

  const handleCloseDialog = () => {
    if (saving) return;
    setDialogOpen(false);
    setEditingUser(null);
    setForm({ ...BASE_FORM });
  };

  const handleEditUser = (admin) => {
    setEditingUser(admin);
    setForm({
      username: admin.username || '',
      email: admin.email || '',
      password: '',
      groupId: admin.group_id ? String(admin.group_id) : '',
    });
    setDialogOpen(true);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    const trimmedUsername = form.username.trim();
    const trimmedEmail = form.email.trim();
    const trimmedPassword = form.password.trim();
    const trimmedGroup = form.groupId.trim();

    if (!trimmedUsername || !trimmedEmail) {
      toast.error('Username and email are required.');
      return;
    }

    if (!trimmedGroup) {
      toast.error('Please select a group for this administrator.');
      return;
    }

    const payload = {
      username: trimmedUsername,
      email: trimmedEmail,
      group_id: Number(trimmedGroup),
      user_type: 'GroupAdmin',
    };

    if (!editingUser) {
      if (!trimmedPassword) {
        toast.error('Password is required for new administrators.');
        return;
      }
      payload.password = trimmedPassword;
    } else if (trimmedPassword) {
      payload.password = trimmedPassword;
    }

    setSaving(true);
    try {
      if (editingUser) {
        await api.put(`/users/${editingUser.id}`, payload);
        toast.success('Group administrator updated successfully');
      } else {
        await api.post('/users', payload);
        toast.success('Group administrator created successfully');
      }

      handleCloseDialog();
      await loadAdmins();
    } catch (error) {
      const message = parseErrorMessage(error, 'Failed to save administrator');
      toast.error(message);
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteUser = async (admin) => {
    if (!admin) return;
    const confirmMessage = `Are you sure you want to delete ${admin.username || 'this group administrator'}?`;
    if (!window.confirm(confirmMessage)) return;

    try {
      await api.delete(`/users/${admin.id}`);
      toast.success('Group administrator deleted');
      await loadAdmins();
    } catch (error) {
      const message = parseErrorMessage(error, 'Failed to delete administrator');
      toast.error(message);
    }
  };

  if (!systemAdmin) {
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
            Group Administrator Management
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Create and manage group administrators across the platform.
          </Typography>
        </Box>
        <Button variant="contained" startIcon={<AddIcon />} onClick={handleOpenDialog}>
          Add Group Admin
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
            {admins.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} align="center">
                  <Typography color="text.secondary">No group administrators found yet.</Typography>
                </TableCell>
              </TableRow>
            ) : (
              admins.map((admin) => (
                <TableRow key={admin.id} hover>
                  <TableCell>{admin.username}</TableCell>
                  <TableCell>{admin.email}</TableCell>
                  <TableCell>{groupMap[admin.group_id] || '—'}</TableCell>
                  <TableCell>
                    <Chip label="Group Admin" color="primary" size="small" />
                  </TableCell>
                  <TableCell align="right">
                    <IconButton color="primary" onClick={() => handleEditUser(admin)}>
                      <EditIcon fontSize="small" />
                    </IconButton>
                    <IconButton color="error" onClick={() => handleDeleteUser(admin)}>
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Box>

      <Dialog open={dialogOpen} onClose={handleCloseDialog} fullWidth maxWidth="sm">
        <DialogTitle>{editingUser ? 'Edit Group Admin' : 'Add Group Admin'}</DialogTitle>
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
              <FormControl fullWidth required error={!form.groupId}>
                <InputLabel id="group-select-label">Group</InputLabel>
                <Select
                  labelId="group-select-label"
                  label="Group"
                  value={form.groupId}
                  onChange={(event) => setForm((prev) => ({ ...prev, groupId: event.target.value }))}
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
                  <FormHelperText>No groups available. Create a group before adding administrators.</FormHelperText>
                )}
              </FormControl>
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseDialog} disabled={saving}>
              Cancel
            </Button>
            <Button type="submit" variant="contained" disabled={saving}>
              {saving ? 'Saving…' : editingUser ? 'Save Changes' : 'Add Group Admin'}
            </Button>
          </DialogActions>
        </form>
      </Dialog>
    </Box>
  );
}

export default SystemAdminUserManagement;
