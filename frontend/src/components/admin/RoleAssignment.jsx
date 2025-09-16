import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Paper, 
  Grid,
  FormControlLabel,
  Switch,
  CircularProgress,
  Alert,
  Button
} from '@mui/material';
import { api } from '../../services/api';

const RoleAssignment = ({ gameId }) => {
  const [roles, setRoles] = useState([]);
  const [assignments, setAssignments] = useState({});
  const [agentConfigs, setAgentConfigs] = useState([]);
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        // Fetch available roles
        const rolesRes = await api.get(`/games/${gameId}/available-roles`);
        setRoles(rolesRes.data);

        // Fetch current assignments
        const assignmentsRes = await api.get(`/games/${gameId}/roles`);
        setAssignments(assignmentsRes.data);

        // Fetch agent configs
        const configsRes = await api.get(`/games/${gameId}/agent-configs`);
        setAgentConfigs(configsRes.data);

        // Fetch users (you'll need to implement this endpoint)
        const usersRes = await api.get(`/games/${gameId}/users`);
        setUsers(usersRes.data);

        setLoading(false);
      } catch (err) {
        setError('Failed to load role assignments');
        console.error(err);
        setLoading(false);
      }
    };

    fetchData();
  }, [gameId]);

  const handleRoleChange = (role, field, value) => {
    setAssignments(prev => ({
      ...prev,
      [role]: {
        ...(prev[role] || {}),
        [field]: value
      }
    }));
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      const updates = [];
      
      // Update each role assignment
      for (const [role, assignment] of Object.entries(assignments)) {
        updates.push(
          api.put(`/games/${gameId}/roles/${role}`, assignment)
        );
      }
      
      await Promise.all(updates);
      // Show success message or update UI
    } catch (err) {
      setError('Failed to save role assignments');
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Role Assignments
      </Typography>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Grid container spacing={3}>
          {roles.map((role) => (
            <Grid item xs={12} md={6} key={role}>
              <Box mb={3}>
                <Typography variant="subtitle1" gutterBottom>
                  {role.charAt(0).toUpperCase() + role.slice(1)}
                </Typography>
                
                <FormControlLabel
                  control={
                    <Switch
                      checked={assignments[role]?.is_ai || false}
                      onChange={(e) => 
                        handleRoleChange(role, 'is_ai', e.target.checked)
                      }
                      color="primary"
                    />
                  }
                  label={assignments[role]?.is_ai ? 'AI Controlled' : 'Human Controlled'}
                  sx={{ mb: 2 }}
                />
                
                {assignments[role]?.is_ai ? (
                  <FormControl fullWidth variant="outlined" size="small">
                    <InputLabel>Agent Configuration</InputLabel>
                    <Select
                      value={assignments[role]?.agent_config_id || ''}
                      onChange={(e) => 
                        handleRoleChange(role, 'agent_config_id', e.target.value)
                      }
                      label="Agent Configuration"
                    >
                      {agentConfigs.map((config) => (
                        <MenuItem key={config.id} value={config.id}>
                          {config.agent_type} - {config.role}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                ) : (
                  <FormControl fullWidth variant="outlined" size="small">
                    <InputLabel>Assign User</InputLabel>
                    <Select
                      value={assignments[role]?.user_id || ''}
                      onChange={(e) => 
                        handleRoleChange(role, 'user_id', e.target.value)
                      }
                      label="Assign User"
                    >
                      {users.map((user) => (
                        <MenuItem key={user.id} value={user.id}>
                          {user.name || user.email}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                )}
              </Box>
            </Grid>
          ))}
        </Grid>
        
        <Box mt={3} display="flex" justifyContent="flex-end">
          <Button 
            variant="contained" 
            color="primary"
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? 'Saving...' : 'Save Changes'}
          </Button>
        </Box>
      </Paper>
    </Box>
  );
};

export default RoleAssignment;
