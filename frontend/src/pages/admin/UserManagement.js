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
import { normalizeRoles, isSystemAdmin as isSystemAdminUser } from '../../utils/authUtils';

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

const BASE_FORM = {
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

  const [form, setForm] = useState({ ...BASE_FORM });

  const [replacementPrompt, setReplacementPrompt] = useState({ ...DEFAULT_REPLACEMENT_PROMPT });

  const navigate = useNavigate();
  const { isGroupAdmin, user } = useAuth();
  const systemAdmin = useMemo(() => isSystemAdminUser(user), [user]);
  const defaultGroupId = useMemo(() => (user?.group_id ? String(user.group_id) : ''), [user]);

  const resetForm = useCallback(() => {
    setForm({
      ...BASE_FORM,
      userType: 'player',
      groupId: systemAdmin ? '' : defaultGroupId,
    });
  }, [defaultGroupId, systemAdmin]);

  useEffect(() => {
    if (!showAddUser) {
      resetForm();
    }
  }, [resetForm, showAddUser]);

  const groupMap = useMemo(() => {
    const map = {};
    (groups || []).forEach((group) => {
      map[group.id] = group.name;
    });
    return map;
  }, [groups]);

  const pageTitle = systemAdmin ? 'User Management' : 'Player Management';
  const addButtonLabel = systemAdmin ? 'Add User' : 'Add Player';
  const modalTitle = editingUser ? (systemAdmin ? 'Edit User' : 'Edit Player') : (systemAdmin ? 'Add New User' : 'Add New Player');
  const submitButtonLabel = editingUser ? 'Save Changes' : addButtonLabel;

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


  const handleOpenModal = () => {
    setEditingUser(null);
    resetForm();
    setShowAddUser(true);

  };

  const handleCloseModal = () => {
    setDialogOpen(false);
    setEditingUser(null);
    resetForm();
  };


  const handleEditUser = (userToEdit) => {
    setEditingUser(userToEdit);
    setForm({
      username: userToEdit.username || '',
      email: userToEdit.email || '',
      password: '',
      userType: systemAdmin ? getUserType(userToEdit) : 'player',
      groupId: systemAdmin
        ? userToEdit.group_id ? String(userToEdit.group_id) : ''
        : userToEdit.group_id ? String(userToEdit.group_id) : defaultGroupId,
    });
    setShowAddUser(true);
  };

  const handleTypeChange = (value) => {

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

    if (!trimmedUsername || !trimmedEmail) {
      toast.error('Username and email are required.');
      return;
    }

    const payload = {
      username: trimmedUsername,
      email: trimmedEmail,
    };

    if (systemAdmin) {
      const requiresGroup = form.userType !== 'system_admin';
      if (requiresGroup && !form.groupId) {
        toast.error('Please select a group for this user.');
        return;
      }

      payload.user_type = form.userType;
      payload.group_id = requiresGroup ? Number(form.groupId) : null;
    }

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

    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800">{pageTitle}</h1>
        <button
          onClick={handleOpenModal}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center"
        >
          <FaPlus className="mr-2" /> {addButtonLabel}
        </button>
      </div>

      {showAddUser && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-lg">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">{modalTitle}</h2>
              <button onClick={handleCloseModal} className="text-gray-500 hover:text-gray-700">
                ✕
              </button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
                <input
                  type="text"
                  value={form.username}
                  onChange={(event) => setForm((prev) => ({ ...prev, username: event.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
                <input
                  type="email"
                  value={form.email}
                  onChange={(event) => setForm((prev) => ({ ...prev, email: event.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Password {editingUser ? '(leave blank to keep current)' : ''}
                </label>
                <input
                  type="password"
                  value={form.password}
                  onChange={(event) => setForm((prev) => ({ ...prev, password: event.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                  required={!editingUser}
                />
              </div>

              {systemAdmin && (
                <div>
                  <span className="block text-sm font-medium text-gray-700 mb-2">User Type</span>
                  <div className="flex flex-wrap gap-3">
                    {USER_TYPE_OPTIONS.map((option) => (
                      <label
                        key={option.value}
                        className={`flex items-center gap-2 px-3 py-2 border rounded-md cursor-pointer ${
                          form.userType === option.value ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
                        }`}
                      >
                        <input
                          type="radio"
                          name="userType"
                          value={option.value}
                          checked={form.userType === option.value}
                          onChange={() => handleTypeChange(option.value)}
                          className="h-4 w-4"
                        />
                        <span className="text-sm font-medium text-gray-700">{option.label}</span>
                      </label>
                    ))}
                  </div>
                </div>
              )}

              {!systemAdmin && (
                <p className="text-sm text-gray-500">Players will automatically be added to your group.</p>
              )}

              {systemAdmin && form.userType !== 'system_admin' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Group</label>
                  <select
                    value={form.groupId}
                    onChange={(event) => setForm((prev) => ({ ...prev, groupId: event.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md"
                    required
                  >
                    <option value="">Select a group</option>
                    {groups.map((group) => (
                      <option key={group.id} value={group.id}>
                        {group.name}
                      </option>
                    ))}
                  </select>
                  {groups.length === 0 && (
                    <p className="text-xs text-red-600 mt-1">
                      No groups available. Create a group before adding players or group admins.
                    </p>
                  )}
                </div>
              )}

              <div className="flex justify-end space-x-3 pt-4">
                <button
                  type="button"
                  onClick={handleCloseModal}
                  className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700"
                >
                  {submitButtonLabel}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <div className="table-surface overflow-hidden sm:rounded-lg">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  User
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Email
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Group
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {users.map((user) => {
                const type = getUserType(user);
                return (
                  <tr key={user.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="flex-shrink-0 h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center">
                          {type === 'system_admin' ? (
                            <FaUserShield className="h-5 w-5 text-blue-600" />
                          ) : (
                            <FaUser className="h-5 w-5 text-gray-400" />
                          )}
                        </div>
                        <div className="ml-4">
                          <div className="text-sm font-medium text-gray-900">{user.username}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{user.email}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {groupMap[user.group_id] || '—'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getTypeBadgeClass(type)}`}>
                        {USER_TYPE_LABELS[type] || 'User'}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <div className="flex items-center justify-end space-x-3">
                        <button
                          onClick={() => handleEditUser(user)}
                          className="text-blue-600 hover:text-blue-900"
                          title="Edit User"
                        >
                          <FaEdit />
                        </button>
                        <button
                          onClick={() => handleDeleteUser(user)}
                          className="text-red-600 hover:text-red-900"
                          title="Delete User"
                        >
                          <FaTrash />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {replacementPrompt.open && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Promote a Group Admin</h3>
            <p className="text-sm text-gray-700 mb-3">
              {replacementPrompt.message || 'Select a group admin to promote to system admin before deleting this user.'}
            </p>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">Replacement System Admin</label>
              <select
                value={replacementPrompt.selected}
                onChange={(event) =>
                  setReplacementPrompt((prev) => ({ ...prev, selected: event.target.value }))
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-md"

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
