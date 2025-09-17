import React, { useState, useEffect } from 'react';
import { 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  TextField, 
  Grid,
  Typography,
  Divider,
  Paper
} from '@mui/material';

const DEMAND_PATTERN_TYPES = [
  { value: 'constant', label: 'Constant' },
  { value: 'random', label: 'Random' },
  { value: 'seasonal', label: 'Seasonal' },
  { value: 'trending', label: 'Trending' },
];

const DemandPatternInput = ({ 
  value = {
    type: 'constant',
    value: 10,
    min: 5,
    max: 15,
    seasonality: 1.0,
    trend: 0
  }, 
  onChange,
  disabled = false
}) => {
  const [pattern, setPattern] = useState(value);

  useEffect(() => {
    if (value) {
      setPattern(value);
    }
  }, [value]);

  const handleChange = (field, fieldValue) => {
    const updatedPattern = { ...pattern, [field]: fieldValue };
    setPattern(updatedPattern);
    if (onChange) {
      onChange(updatedPattern);
    }
  };

  const renderPatternFields = () => {
    switch (pattern.type) {
      case 'constant':
        return (
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Value"
              type="number"
              value={pattern.value || 0}
              onChange={(e) => handleChange('value', parseFloat(e.target.value) || 0)}
              disabled={disabled}
              size="small"
            />
          </Grid>
        );
      
      case 'random':
        return (
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Min Value"
                type="number"
                value={pattern.min || 0}
                onChange={(e) => handleChange('min', parseFloat(e.target.value) || 0)}
                disabled={disabled}
                size="small"
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Max Value"
                type="number"
                value={pattern.max || 0}
                onChange={(e) => handleChange('max', parseFloat(e.target.value) || 0)}
                disabled={disabled}
                size="small"
              />
            </Grid>
          </Grid>
        );
      
      case 'seasonal':
        return (
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Base Value"
                type="number"
                value={pattern.value || 0}
                onChange={(e) => handleChange('value', parseFloat(e.target.value) || 0)}
                disabled={disabled}
                size="small"
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Seasonality (0-2)"
                type="number"
                value={pattern.seasonality || 1.0}
                onChange={(e) => handleChange('seasonality', parseFloat(e.target.value) || 1.0)}
                inputProps={{ min: 0, max: 2, step: 0.1 }}
                disabled={disabled}
                size="small"
              />
            </Grid>
          </Grid>
        );
      
      case 'trending':
        return (
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Base Value"
                type="number"
                value={pattern.value || 0}
                onChange={(e) => handleChange('value', parseFloat(e.target.value) || 0)}
                disabled={disabled}
                size="small"
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                fullWidth
                label="Trend per period"
                type="number"
                value={pattern.trend || 0}
                onChange={(e) => handleChange('trend', parseFloat(e.target.value) || 0)}
                disabled={disabled}
                size="small"
              />
            </Grid>
          </Grid>
        );
      
      default:
        return null;
    }
  };

  return (
    <Paper variant="outlined" sx={{ p: 2, mt: 2 }}>
      <Typography variant="subtitle2" gutterBottom>
        Demand Pattern
      </Typography>
      <Divider sx={{ mb: 2 }} />
      
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <FormControl fullWidth size="small" disabled={disabled}>
            <InputLabel id="demand-pattern-type-label">Pattern Type</InputLabel>
            <Select
              labelId="demand-pattern-type-label"
              value={pattern.type || 'constant'}
              label="Pattern Type"
              onChange={(e) => handleChange('type', e.target.value)}
              disabled={disabled}
            >
              {DEMAND_PATTERN_TYPES.map((type) => (
                <MenuItem key={type.value} value={type.value}>
                  {type.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12}>
          {renderPatternFields()}
        </Grid>
      </Grid>
    </Paper>
  );
};

export default DemandPatternInput;
