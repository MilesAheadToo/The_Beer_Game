import React, { useEffect, useState } from 'react';
import { Box, Button, TextField, Dialog, DialogActions, DialogContent, DialogTitle, Table, TableHead, TableRow, TableCell, TableBody, IconButton } from '@mui/material';
import { Edit, Delete } from '@mui/icons-material';
import api from '../../services/api';

const GroupManagement = () => {
  const [groups, setGroups] = useState([]);
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState(null);
  const [form, setForm] = useState({ name: '', description: '', logo: '', admin: { username: '', email: '', password: '', full_name: '' } });

  const fetchGroups = async () => {
    try {
      const res = await api.get('/api/v1/groups');
      setGroups(res.data);
    } catch (err) {
      setGroups([]);
    }
  };

  useEffect(() => { fetchGroups(); }, []);

  const handleOpen = (group) => {
    if (group) {
      setEditing(group.id);
      setForm({ name: group.name, description: group.description || '', logo: group.logo || '', admin: { username: '', email: '', password: '', full_name: '' } });
    } else {
      setEditing(null);
      setForm({ name: '', description: '', logo: '', admin: { username: '', email: '', password: '', full_name: '' } });
    }
    setOpen(true);
  };

  const handleClose = () => setOpen(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    if (name.startsWith('admin.')) {
      const key = name.split('.')[1];
      setForm({ ...form, admin: { ...form.admin, [key]: value } });
    } else {
      setForm({ ...form, [name]: value });
    }
  };

  const handleSubmit = async () => {
    try {
      if (editing) {
        await api.put(`/api/v1/groups/${editing}`, { name: form.name, description: form.description, logo: form.logo });
      } else {
        await api.post('/api/v1/groups', form);
      }
      handleClose();
      fetchGroups();
    } catch (err) {
      console.error(err);
    }
  };

  const handleDelete = async (id) => {
    try {
      await api.delete(`/api/v1/groups/${id}`);
      fetchGroups();
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <Box p={2}>
      <Button variant="contained" onClick={() => handleOpen(null)}>Add Group</Button>
      <Table sx={{ mt: 2 }}>
        <TableHead>
          <TableRow>
            <TableCell>Name</TableCell>
            <TableCell>Description</TableCell>
            <TableCell>Admin</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {groups.map(g => (
            <TableRow key={g.id}>
              <TableCell>{g.name}</TableCell>
              <TableCell>{g.description}</TableCell>
              <TableCell>{g.admin?.email}</TableCell>
              <TableCell>
                <IconButton onClick={() => handleOpen(g)}><Edit /></IconButton>
                <IconButton onClick={() => handleDelete(g.id)}><Delete /></IconButton>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
        <DialogTitle>{editing ? 'Edit Group' : 'Create Group'}</DialogTitle>
        <DialogContent>
          <TextField margin="dense" label="Name" name="name" fullWidth value={form.name} onChange={handleChange} />
          <TextField margin="dense" label="Description" name="description" fullWidth value={form.description} onChange={handleChange} />
          <TextField margin="dense" label="Logo" name="logo" fullWidth value={form.logo} onChange={handleChange} />
          {!editing && (
            <>
              <TextField margin="dense" label="Admin Username" name="admin.username" fullWidth value={form.admin.username} onChange={handleChange} />
              <TextField margin="dense" label="Admin Email" name="admin.email" fullWidth value={form.admin.email} onChange={handleChange} />
              <TextField margin="dense" label="Admin Full Name" name="admin.full_name" fullWidth value={form.admin.full_name} onChange={handleChange} />
              <TextField margin="dense" label="Admin Password" type="password" name="admin.password" fullWidth value={form.admin.password} onChange={handleChange} />
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose}>Cancel</Button>
          <Button onClick={handleSubmit} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default GroupManagement;
