import React, { useEffect, useState } from 'react';
import {
  Box,
  Button,
  TextField,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  IconButton,
  Typography,
} from '@mui/material';
import { Edit, Delete } from '@mui/icons-material';
import api from '../../services/api';

const defaultForm = {
  name: 'Daybreak',
  description: '',
  logo: '/daybreak_logo.png',
  admin: {
    username: 'groupadmin',
    email: 'groupadmin@daybreak.ai',
    password: 'Daybreak@2025',
    full_name: 'Group Administrator'
  }
};

const GroupManagement = () => {
  const [groups, setGroups] = useState([]);
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState(null);
  const [form, setForm] = useState(defaultForm);
  const [logoPreview, setLogoPreview] = useState(defaultForm.logo || '');
  const [logoFileName, setLogoFileName] = useState('');

  const fetchGroups = async () => {
    try {
      const res = await api.get('/api/v1/groups');
      setGroups(res.data);
      if (res.data.length === 0) {
        handleOpen(null);
      }
    } catch (err) {
      setGroups([]);
    }
  };

  useEffect(() => { fetchGroups(); }, []);

  useEffect(() => {
    setLogoPreview(form.logo || '');
  }, [form.logo]);

  const handleOpen = (group) => {
    if (group) {
      setEditing(group.id);
      setForm({ name: group.name, description: group.description || '', logo: group.logo || '', admin: { username: '', email: '', password: '', full_name: '' } });
      setLogoPreview(group.logo || '');
    } else {
      setEditing(null);
      setForm(defaultForm);
      setLogoPreview(defaultForm.logo || '');
    }
    setLogoFileName('');
    setOpen(true);
  };

  const handleClose = () => setOpen(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    if (name.startsWith('admin.')) {
      const key = name.split('.')[1];
      setForm({ ...form, admin: { ...form.admin, [key]: value } });
    } else if (name === 'logo') {
      setLogoFileName('');
      setForm({ ...form, logo: value });
      setLogoPreview(value);
    } else {
      setForm({ ...form, [name]: value });
    }
  };

  const handleLogoFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result || '';
      setForm((prev) => ({ ...prev, logo: result }));
      setLogoPreview(result);
      setLogoFileName(file.name);
    };
    reader.readAsDataURL(file);
    event.target.value = '';
  };

  const handleRemoveLogo = () => {
    setLogoFileName('');
    setForm((prev) => ({ ...prev, logo: '' }));
    setLogoPreview('');
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
      {groups.length > 0 && (
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
      )}
      <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
        <DialogTitle>{editing ? 'Edit Group' : 'Create Group'}</DialogTitle>
        <DialogContent>
          <TextField margin="dense" label="Name" name="name" fullWidth value={form.name} onChange={handleChange} />
          <TextField margin="dense" label="Description" name="description" fullWidth value={form.description} onChange={handleChange} />
          <Box mt={2} mb={1}>
            <Typography variant="subtitle2" gutterBottom>
              Group Logo
            </Typography>
            <TextField
              margin="dense"
              label="Logo URL or data"
              name="logo"
              fullWidth
              value={form.logo || ''}
              onChange={handleChange}
              placeholder="Paste a logo URL or upload a file below"
            />
            <Box display="flex" alignItems="center" mt={1} gap={1} flexWrap="wrap">
              <Button variant="outlined" component="label" size="small">
                Upload Logo
                <input type="file" accept="image/*" hidden onChange={handleLogoFileChange} />
              </Button>
              <Typography variant="body2" color="text.secondary">
                {logoFileName ? `Selected file: ${logoFileName}` : 'Upload an image (PNG, JPG, SVG) or paste a URL above.'}
              </Typography>
              {logoPreview && (
                <Button size="small" onClick={handleRemoveLogo}>
                  Remove
                </Button>
              )}
            </Box>
            {logoPreview && (
              <Box mt={2} display="flex" alignItems="center" gap={2}>
                <Box
                  component="img"
                  src={logoPreview}
                  alt="Logo preview"
                  sx={{
                    width: 80,
                    height: 80,
                    objectFit: 'contain',
                    borderRadius: 1,
                    border: '1px solid',
                    borderColor: 'divider',
                    backgroundColor: 'background.default',
                    p: 1,
                  }}
                />
                <Typography variant="body2" color="text.secondary">
                  Preview of the logo that will be saved for this group.
                </Typography>
              </Box>
            )}
          </Box>
          <TextField margin="dense" label="SC Config" name="sc_config" fullWidth value="Default TBG" disabled />
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
