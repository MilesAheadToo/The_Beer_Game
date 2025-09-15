import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaTrash, FaPlus, FaUserShield, FaEdit } from 'react-icons/fa';
import { toast } from 'react-toastify';
import { useAuth } from '../../contexts/AuthContext';
import { api, mixedGameApi } from '../../services/api';
import { normalizeRoles, isSystemAdmin as isSystemAdminUser } from '../../utils/authUtils';

const DEFAULT_FORM = {
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

const isGroupAdminAccount = (user) => {
  if (!user || user.is_superuser) return false;
  const normalizedRoles = normalizeRoles(user.roles || []);
  return normalizedRoles.includes('groupadmin') || normalizedRoles.includes('admin');
};

const getTypeBadgeClass = () => 'bg-blue-100 text-blue-800';

function SystemUserManagement() {
  const [users, setUsers] = useState([]);
  const [groups, setGroups] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showAddUser, setShowAddUser] = useState(false);
  const [editingUser, setEditingUser] = useState(null);
  const [form, setForm] = useState({ ...DEFAULT_FORM });

  const navigate = useNavigate();
  const { user, loading } = useAuth();
  const systemAdmin = isSystemAdminUser(user);

  const groupMap = useMemo(() => {
    const entries = (groups || []).map((group) => [group.id, group.name]);
    return Object.fromEntries(entries);
  }, [groups]);

  const loadGroups = useCallback(async () => {
    const response = await api.get('/groups');
    setGroups(Array.isArray(response.data) ? response.data : []);
  }, []);

  const loadUsers = useCallback(async () => {
    const response = await api.get('/auth/users/');
    const data = Array.isArray(response.data) ? response.data : [];
    setUsers(data.filter(isGroupAdminAccount));
  }, []);

  useEffect(() => {
    if (loading) {
      return;
    }

    if (!systemAdmin) {
      navigate('/unauthorized');
      return;
    }

    const fetchAll = async () => {
      setIsLoading(true);
      try {
        await mixedGameApi.health();
        await Promise.all([loadGroups(), loadUsers()]);
      } catch (error) {
        console.error('Error loading system user management data:', error);
        toast.error('Failed to load group admin information');
      } finally {
        setIsLoading(false);
      }
    };

    fetchAll();
  }, [loading, systemAdmin, navigate, loadGroups, loadUsers]);

  const handleOpenModal = () => {
    setEditingUser(null);
    setForm({ ...DEFAULT_FORM });
    setShowAddUser(true);
  };

  const handleCloseModal = () => {
    setShowAddUser(false);
    setEditingUser(null);
    setForm({ ...DEFAULT_FORM });
  };

  const handleEditUser = (userToEdit) => {
    setEditingUser(userToEdit);
    setForm({
      username: userToEdit.username || '',
      email: userToEdit.email || '',
      password: '',
      groupId: userToEdit.group_id ? String(userToEdit.group_id) : '',
    });
    setShowAddUser(true);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    const trimmedUsername = form.username.trim();
    const trimmedEmail = form.email.trim();

    if (!trimmedUsername || !trimmedEmail) {
      toast.error('Username and email are required.');
      return;
    }

    if (!form.groupId) {
      toast.error('Please select a group for this admin.');
      return;
    }

    const payload = {
      username: trimmedUsername,
      email: trimmedEmail,
      user_type: 'group_admin',
      group_id: Number(form.groupId),
    };

    if (!editingUser || form.password) {
      if (!editingUser && !form.password) {
        toast.error('Password is required for new admins.');
        return;
      }
      payload.password = form.password;
    }

    try {
      if (editingUser) {
        await api.put(`/users/${editingUser.id}`, payload);
        toast.success('Group admin updated');
      } else {
        await api.post('/users/', payload);
        toast.success('Group admin created');
      }

      handleCloseModal();
      try {
        await Promise.all([loadUsers(), loadGroups()]);
      } catch (refreshError) {
        console.error('Error refreshing group admin list:', refreshError);
        toast.error('Changes saved, but the list could not be refreshed.');
      }
    } catch (error) {
      const message = parseErrorMessage(error, 'Failed to save group admin');
      toast.error(message);
    }
  };

  const handleDeleteUser = async (userToDelete) => {
    if (!userToDelete) return;
    const confirmMessage = `Are you sure you want to delete ${userToDelete.username || 'this admin'}?`;
    if (!window.confirm(confirmMessage)) return;

    try {
      await api.delete(`/users/${userToDelete.id}`);
      toast.success('Group admin deleted');
      try {
        await Promise.all([loadUsers(), loadGroups()]);
      } catch (refreshError) {
        console.error('Error refreshing group admin list:', refreshError);
        toast.error('Admin deleted, but the list could not be refreshed.');
      }
    } catch (error) {
      const message = parseErrorMessage(error, 'Failed to delete group admin');
      toast.error(message);
    }
  };

  if (loading || isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500" />
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800">Group Admin Management</h1>
        <button
          onClick={handleOpenModal}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center"
        >
          <FaPlus className="mr-2" /> Add Group Admin
        </button>
      </div>

      {showAddUser && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-lg">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">{editingUser ? 'Edit Group Admin' : 'Add Group Admin'}</h2>
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
                    No groups available. Create a group before adding group admins.
                  </p>
                )}
              </div>

              <div className="bg-blue-50 border border-blue-200 rounded-md p-3 text-sm text-blue-800">
                Group admin accounts have access to manage games and users within their assigned group.
              </div>

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
                  {editingUser ? 'Save Changes' : 'Add Group Admin'}
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
                  Admin
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
              {users.map((admin) => (
                <tr key={admin.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="flex-shrink-0 h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center">
                        <FaUserShield className="h-5 w-5 text-blue-600" />
                      </div>
                      <div className="ml-4">
                        <div className="text-sm font-medium text-gray-900">{admin.username}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{admin.email}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {groupMap[admin.group_id] || '—'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getTypeBadgeClass()}`}>
                      Group Admin
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <div className="flex items-center justify-end space-x-3">
                      <button
                        onClick={() => handleEditUser(admin)}
                        className="text-blue-600 hover:text-blue-900"
                        title="Edit Group Admin"
                      >
                        <FaEdit />
                      </button>
                      <button
                        onClick={() => handleDeleteUser(admin)}
                        className="text-red-600 hover:text-red-900"
                        title="Delete Group Admin"
                      >
                        <FaTrash />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default SystemUserManagement;
