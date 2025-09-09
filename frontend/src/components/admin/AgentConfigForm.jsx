import React, { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  FormHelperText,
  CircularProgress,
  Alert,
  Divider
} from '@mui/material';
import { Save, Delete } from '@mui/icons-material';
import { useFormik } from 'formik';
import * as Yup from 'yup';
import api from '../../services/api';

const agentTypes = [
  { value: 'base', label: 'Base Agent' },
  { value: 'rule_based', label: 'Rule Based' },
  { value: 'reinforcement_learning', label: 'Reinforcement Learning' },
];

const validationSchema = Yup.object({
  role: Yup.string().required('Required'),
  agent_type: Yup.string().required('Required'),
  config: Yup.object().test(
    'config-validation',
    'Invalid configuration',
    function(value) {
      // Add specific validation based on agent_type if needed
      return true;
    }
  )
});

const AgentConfigForm = ({ gameId, configId, onSuccess }) => {
  const [loading, setLoading] = useState(!!configId);
  const [error, setError] = useState(null);
  const [availableRoles, setAvailableRoles] = useState([]);

  const formik = useFormik({
    initialValues: {
      role: '',
      agent_type: 'base',
      config: {}
    },
    validationSchema,
    onSubmit: async (values) => {
      try {
        setError(null);
        const data = {
          ...values,
          game_id: gameId
        };
        
        if (configId) {
          await api.put(`/api/agent-configs/${configId}`, data);
        } else {
          await api.post('/api/agent-configs', data);
        }
        
        if (onSuccess) onSuccess();
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to save configuration');
        console.error(err);
      }
    },
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch available roles
        const rolesRes = await api.get(`/api/games/${gameId}/available-roles`);
        setAvailableRoles(rolesRes.data);

        // If editing, load the config
        if (configId) {
          const configRes = await api.get(`/api/agent-configs/${configId}`);
          formik.setValues(configRes.data);
        }
      } catch (err) {
        setError('Failed to load data');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [gameId, configId, formik]);

  const renderConfigFields = () => {
    const { agent_type } = formik.values;
    
    switch (agent_type) {
      case 'rule_based':
        return (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Aggressiveness (0-1)"
                type="number"
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                value={formik.values.config?.aggressiveness || 0.5}
                onChange={(e) =>
                  formik.setFieldValue('config', {
                    ...formik.values.config,
                    aggressiveness: parseFloat(e.target.value)
                  })
                }
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Smoothing Factor (0-1)"
                type="number"
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                value={formik.values.config?.smoothing_factor || 0.7}
                onChange={(e) =>
                  formik.setFieldValue('config', {
                    ...formik.values.config,
                    smoothing_factor: parseFloat(e.target.value)
                  })
                }
              />
            </Grid>
          </Grid>
        );
        
      case 'reinforcement_learning':
        return (
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Learning Rate (0-1)"
                type="number"
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                value={formik.values.config?.learning_rate || 0.01}
                onChange={(e) =>
                  formik.setFieldValue('config', {
                    ...formik.values.config,
                    learning_rate: parseFloat(e.target.value)
                  })
                }
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Discount Factor (0-1)"
                type="number"
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                value={formik.values.config?.discount_factor || 0.9}
                onChange={(e) =>
                  formik.setFieldValue('config', {
                    ...formik.values.config,
                    discount_factor: parseFloat(e.target.value)
                  })
                }
              />
            </Grid>
          </Grid>
        );
        
      default:
        return (
          <Typography color="textSecondary">
            No additional configuration required for this agent type.
          </Typography>
        );
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        {configId ? 'Edit' : 'Create'} Agent Configuration
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <form onSubmit={formik.handleSubmit}>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth error={formik.touched.role && Boolean(formik.errors.role)}>
              <InputLabel>Role</InputLabel>
              <Select
                name="role"
                value={formik.values.role}
                onChange={formik.handleChange}
                label="Role"
                disabled={!!configId}
              >
                {availableRoles.map((role) => (
                  <MenuItem key={role} value={role}>
                    {role.charAt(0).toUpperCase() + role.slice(1)}
                  </MenuItem>
                ))}
              </Select>
              {formik.touched.role && formik.errors.role && (
                <FormHelperText>{formik.errors.role}</FormHelperText>
              )}
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormControl fullWidth error={formik.touched.agent_type && Boolean(formik.errors.agent_type)}>
              <InputLabel>Agent Type</InputLabel>
              <Select
                name="agent_type"
                value={formik.values.agent_type}
                onChange={formik.handleChange}
                label="Agent Type"
              >
                {agentTypes.map((type) => (
                  <MenuItem key={type.value} value={type.value}>
                    {type.label}
                  </MenuItem>
                ))}
              </Select>
              {formik.touched.agent_type && formik.errors.agent_type && (
                <FormHelperText>{formik.errors.agent_type}</FormHelperText>
              )}
            </FormControl>
          </Grid>
          
          <Grid item xs={12}>
            <Typography variant="subtitle2" gutterBottom>
              Configuration
            </Typography>
            <Divider sx={{ mb: 2 }} />
            {renderConfigFields()}
          </Grid>
          
          <Grid item xs={12}>
            <Box display="flex" justifyContent="space-between">
              <div>
                {configId && (
                  <Button
                    color="error"
                    startIcon={<Delete />}
                    onClick={async () => {
                      if (window.confirm('Are you sure you want to delete this configuration?')) {
                        try {
                          await api.delete(`/api/agent-configs/${configId}`);
                          if (onSuccess) onSuccess();
                        } catch (err) {
                          setError('Failed to delete configuration');
                          console.error(err);
                        }
                      }
                    }}
                  >
                    Delete
                  </Button>
                )}
              </div>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                startIcon={<Save />}
                disabled={formik.isSubmitting}
              >
                {formik.isSubmitting ? 'Saving...' : 'Save Configuration'}
              </Button>
            </Box>
          </Grid>
        </Grid>
      </form>
    </Paper>
  );
};

export default AgentConfigForm;
