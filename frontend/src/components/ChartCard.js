import React from 'react';
import { Paper, Typography, Box } from '@mui/material';

const ChartCard = ({ title, subtitle, height = 320, children, footer }) => {
  return (
    <Paper sx={{ p: 2.5 }}>
      {title && (
        <Typography variant="h6" gutterBottom>{title}</Typography>
      )}
      {subtitle && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>{subtitle}</Typography>
      )}
      <Box sx={{ height }}>
        {children}
      </Box>
      {footer}
    </Paper>
  );
};

export default ChartCard;
