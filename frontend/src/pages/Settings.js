import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Tabs,
  Tab,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Switch,
  Divider,
  Grid,
  Avatar,
  IconButton,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  ListSubheader,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  Person as PersonIcon,
  Notifications as NotificationsIcon,
  Security as SecurityIcon,
  Api as ApiIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Add as AddIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
} from '@mui/icons-material';

const Settings = () => {
  const [tabValue, setTabValue] = useState(0);
  const [notificationSettings, setNotificationSettings] = useState({
    email: true,
    push: true,
    weeklyReport: true,
    criticalAlerts: true,
  });
  const [apiKeys, setApiKeys] = useState([
    { id: 'key1', name: 'Production', key: 'sk_test_1234567890', created: '2023-01-15' },
    { id: 'key2', name: 'Development', key: 'sk_test_0987654321', created: '2023-02-20' },
  ]);
  const [newApiKey, setNewApiKey] = useState({ name: '', key: '' });
  const [editingKey, setEditingKey] = useState(null);
  const [showKeyAlert, setShowKeyAlert] = useState(false);
  const [passwordForm, setPasswordForm] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });
  const [profileForm, setProfileForm] = useState({
    firstName: 'John',
    lastName: 'Doe',
    email: 'john.doe@example.com',
    company: 'Supply Chain Co.',
    timezone: 'UTC+01:00',
  });

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleNotificationChange = (setting) => (event) => {
    setNotificationSettings({
      ...notificationSettings,
      [setting]: event.target.checked,
    });
  };

  const handleAddApiKey = () => {
    if (newApiKey.name.trim() === '') return;
    
    const key = {
      id: `key${Date.now()}`,
      name: newApiKey.name,
      key: `sk_test_${Math.random().toString(36).substring(2, 15)}`,
      created: new Date().toISOString().split('T')[0],
    };
    
    setApiKeys([...apiKeys, key]);
    setNewApiKey({ name: '', key: '' });
    setShowKeyAlert(true);
  };

  const handleDeleteApiKey = (id) => {
    setApiKeys(apiKeys.filter(key => key.id !== id));
  };

  const handleEditApiKey = (key) => {
    setEditingKey(key.id);
    setNewApiKey({ name: key.name, key: key.key });
  };

  const handleUpdateApiKey = () => {
    if (!editingKey || newApiKey.name.trim() === '') return;
    
    setApiKeys(apiKeys.map(key => 
      key.id === editingKey 
        ? { ...key, name: newApiKey.name }
        : key
    ));
    
    setEditingKey(null);
    setNewApiKey({ name: '', key: '' });
  };

  const handleCancelEdit = () => {
    setEditingKey(null);
    setNewApiKey({ name: '', key: '' });
  };

  const handlePasswordChange = (field) => (event) => {
    setPasswordForm({
      ...passwordForm,
      [field]: event.target.value,
    });
  };

  const handleProfileChange = (field) => (event) => {
    setProfileForm({
      ...profileForm,
      [field]: event.target.value,
    });
  };

  const handleSaveProfile = () => {
    // In a real app, this would call an API to update the profile
    console.log('Profile updated:', profileForm);
  };

  const handleChangePassword = () => {
    // In a real app, this would validate and call an API to change the password
    if (passwordForm.newPassword !== passwordForm.confirmPassword) {
      alert('New passwords do not match');
      return;
    }
    console.log('Password changed');
    setPasswordForm({
      currentPassword: '',
      newPassword: '',
      confirmPassword: '',
    });
  };

  const renderTabContent = () => {
    switch (tabValue) {
      case 0: // Profile
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Profile Information
            </Typography>
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} md={4}>
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <Avatar
                    sx={{ width: 120, height: 120, mb: 2 }}
                    src="/path-to-avatar.jpg"
                  />
                  <Button variant="outlined" size="small" sx={{ mt: 1 }}>
                    Change Photo
                  </Button>
                </Box>
              </Grid>
              <Grid item xs={12} md={8}>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="First Name"
                      value={profileForm.firstName}
                      onChange={handleProfileChange('firstName')}
                      margin="normal"
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Last Name"
                      value={profileForm.lastName}
                      onChange={handleProfileChange('lastName')}
                      margin="normal"
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Email"
                      type="email"
                      value={profileForm.email}
                      onChange={handleProfileChange('email')}
                      margin="normal"
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Company"
                      value={profileForm.company}
                      onChange={handleProfileChange('company')}
                      margin="normal"
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <FormControl fullWidth margin="normal">
                      <InputLabel>Timezone</InputLabel>
                      <Select
                        value={profileForm.timezone}
                        onChange={handleProfileChange('timezone')}
                        label="Timezone"
                      >
                        <MenuItem value="UTC+00:00">UTC±00:00 (GMT)</MenuItem>
                        <MenuItem value="UTC+01:00">UTC+01:00 (CET)</MenuItem>
                        <MenuItem value="UTC-05:00">UTC-05:00 (EST)</MenuItem>
                        <MenuItem value="UTC-08:00">UTC-08:00 (PST)</MenuItem>
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12} sx={{ mt: 2 }}>
                    <Button
                      variant="contained"
                      color="primary"
                      onClick={handleSaveProfile}
                    >
                      Save Changes
                    </Button>
                  </Grid>
                </Grid>
              </Grid>
            </Grid>

            <Divider sx={{ my: 4 }} />

            <Typography variant="h6" gutterBottom>
              Change Password
            </Typography>
            <Grid container spacing={2} sx={{ maxWidth: 600, mb: 4 }}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Current Password"
                  type="password"
                  value={passwordForm.currentPassword}
                  onChange={handlePasswordChange('currentPassword')}
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="New Password"
                  type="password"
                  value={passwordForm.newPassword}
                  onChange={handlePasswordChange('newPassword')}
                  margin="normal"
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Confirm New Password"
                  type="password"
                  value={passwordForm.confirmPassword}
                  onChange={handlePasswordChange('confirmPassword')}
                  margin="normal"
                  error={
                    passwordForm.newPassword !== '' &&
                    passwordForm.confirmPassword !== '' &&
                    passwordForm.newPassword !== passwordForm.confirmPassword
                  }
                  helperText={
                    passwordForm.newPassword !== '' &&
                    passwordForm.confirmPassword !== '' &&
                    passwordForm.newPassword !== passwordForm.confirmPassword
                      ? 'Passwords do not match'
                      : ''
                  }
                />
              </Grid>
              <Grid item xs={12}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleChangePassword}
                  disabled={
                    !passwordForm.currentPassword ||
                    !passwordForm.newPassword ||
                    passwordForm.newPassword !== passwordForm.confirmPassword
                  }
                >
                  Update Password
                </Button>
              </Grid>
            </Grid>
          </Box>
        );

      case 1: // Notifications
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Notification Preferences
            </Typography>
            <Typography variant="body2" color="textSecondary" paragraph>
              Choose how you receive notifications. You can change these settings at any time.
            </Typography>

            <Paper sx={{ p: 3, mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                Email Notifications
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.email}
                    onChange={handleNotificationChange('email')}
                    color="primary"
                  />
                }
                label="Enable email notifications"
                sx={{ mb: 1, display: 'block' }}
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.weeklyReport}
                    onChange={handleNotificationChange('weeklyReport')}
                    color="primary"
                    disabled={!notificationSettings.email}
                  />
                }
                label="Weekly summary report"
                sx={{ mb: 1, display: 'block', ml: 4 }}
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.criticalAlerts}
                    onChange={handleNotificationChange('criticalAlerts')}
                    color="primary"
                    disabled={!notificationSettings.email}
                  />
                }
                label="Critical alerts"
                sx={{ display: 'block', ml: 4 }}
              />
            </Paper>

            <Paper sx={{ p: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                Push Notifications
              </Typography>
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.push}
                    onChange={handleNotificationChange('push')}
                    color="primary"
                  />
                }
                label="Enable push notifications"
              />
            </Paper>
          </Box>
        );

      case 2: // API Keys
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              API Keys
            </Typography>
            <Typography variant="body2" color="textSecondary" paragraph>
              Manage your API keys for programmatic access to the Supply Chain API.
            </Typography>

            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <TextField
                label="Key Name"
                value={newApiKey.name}
                onChange={(e) => setNewApiKey({ ...newApiKey, name: e.target.value })}
                size="small"
                sx={{ flex: 1 }}
                placeholder="e.g., Production, Development"
              />
              {editingKey ? (
                <>
                  <Button
                    variant="contained"
                    color="primary"
                    startIcon={<SaveIcon />}
                    onClick={handleUpdateApiKey}
                    disabled={!newApiKey.name.trim()}
                  >
                    Update
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<CancelIcon />}
                    onClick={handleCancelEdit}
                  >
                    Cancel
                  </Button>
                </>
              ) : (
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<AddIcon />}
                  onClick={handleAddApiKey}
                  disabled={!newApiKey.name.trim()}
                >
                  Add Key
                </Button>
              )}
            </Box>

            <Paper>
              <List>
                <ListSubheader>Your API Keys</ListSubheader>
                {apiKeys.length === 0 ? (
                  <ListItem>
                    <ListItemText primary="No API keys found" />
                  </ListItem>
                ) : (
                  apiKeys.map((key) => (
                    <ListItem key={key.id} divider>
                      <ListItemText
                        primary={key.name}
                        secondary={`Created: ${key.created} • ${key.key}`}
                        sx={{ wordBreak: 'break-all' }}
                      />
                      <ListItemSecondaryAction>
                        <IconButton
                          edge="end"
                          onClick={() => handleEditApiKey(key)}
                          sx={{ mr: 1 }}
                        >
                          <EditIcon />
                        </IconButton>
                        <IconButton
                          edge="end"
                          onClick={() => handleDeleteApiKey(key.id)}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))
                )}
              </List>
            </Paper>

            <Alert severity="info" sx={{ mt: 3 }}>
              <strong>Keep your API keys secure.</strong> Do not share them in publicly accessible
              areas such as GitHub, client-side code, and so forth.
            </Alert>
          </Box>
        );

      case 3: // Security
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Security Settings
            </Typography>
            <Typography variant="body2" color="textSecondary" paragraph>
              Manage your account security settings and active sessions.
            </Typography>

            <Paper sx={{ p: 3, mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                Two-Factor Authentication
              </Typography>
              <Typography variant="body2" paragraph>
                Add an extra layer of security to your account by enabling two-factor authentication.
              </Typography>
              <Button variant="outlined" color="primary">
                Set Up Two-Factor Authentication
              </Button>
            </Paper>

            <Paper sx={{ p: 3, mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                Active Sessions
              </Typography>
              <Typography variant="body2" paragraph>
                This is a list of devices that have logged into your account. Revoke any sessions that you do not recognize.
              </Typography>
              <List>
                <ListItem divider>
                  <ListItemText
                    primary="Chrome on Windows 10"
                    secondary={`Current session • Last active: ${new Date().toLocaleString()}`}
                  />
                  <ListItemSecondaryAction>
                    <Button color="error" size="small">
                      Revoke
                    </Button>
                  </ListItemSecondaryAction>
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Firefox on macOS"
                    secondary={`Last active: 2 days ago`}
                  />
                  <ListItemSecondaryAction>
                    <Button color="error" size="small">
                      Revoke
                    </Button>
                  </ListItemSecondaryAction>
                </ListItem>
              </List>
            </Paper>

            <Paper sx={{ p: 3 }}>
              <Typography variant="subtitle1" gutterBottom color="error">
                Danger Zone
              </Typography>
              <Typography variant="body2" paragraph>
                Permanently delete your account and all associated data. This action cannot be undone.
              </Typography>
              <Button variant="outlined" color="error" startIcon={<DeleteIcon />}>
                Delete My Account
              </Button>
            </Paper>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      
      <Tabs
        value={tabValue}
        onChange={handleTabChange}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ mb: 3 }}
      >
        <Tab icon={<PersonIcon />} label="Profile" />
        <Tab icon={<NotificationsIcon />} label="Notifications" />
        <Tab icon={<ApiIcon />} label="API Keys" />
        <Tab icon={<SecurityIcon />} label="Security" />
      </Tabs>

      {renderTabContent()}

      <Snackbar
        open={showKeyAlert}
        autoHideDuration={6000}
        onClose={() => setShowKeyAlert(false)}
        message="New API key created. Make sure to copy it now as you won't be able to see it again!"
        action={
          <Button color="secondary" size="small" onClick={() => setShowKeyAlert(false)}>
            OK
          </Button>
        }
      />
    </Box>
  );
};

export default Settings;
