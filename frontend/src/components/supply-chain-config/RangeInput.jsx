import React from 'react';
import { TextField, Grid, Typography, Box } from '@mui/material';

const RangeInput = ({ 
  label, 
  value = { min: 0, max: 100 }, 
  onChange, 
  min = 0, 
  max = 10000,
  step = 1,
  disabled = false
}) => {
  const handleMinChange = (e) => {
    const newMin = parseFloat(e.target.value) || 0;
    onChange({ ...value, min: Math.min(newMin, value.max) });
  };

  const handleMaxChange = (e) => {
    const newMax = parseFloat(e.target.value) || 0;
    onChange({ ...value, max: Math.max(newMax, value.min) });
  };

  return (
    <Box>
      {label && (
        <Typography variant="subtitle2" gutterBottom>
          {label}
        </Typography>
      )}
      <Grid container spacing={2}>
        <Grid item xs={6}>
          <TextField
            fullWidth
            label="Min"
            type="number"
            value={value?.min || ''}
            onChange={handleMinChange}
            inputProps={{ min, max: value?.max, step }}
            disabled={disabled}
            size="small"
          />
        </Grid>
        <Grid item xs={6}>
          <TextField
            fullWidth
            label="Max"
            type="number"
            value={value?.max || ''}
            onChange={handleMaxChange}
            inputProps={{ min: value?.min, max, step }}
            disabled={disabled}
            size="small"
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export default RangeInput;
