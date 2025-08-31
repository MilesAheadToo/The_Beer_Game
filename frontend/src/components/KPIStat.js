import React from 'react';
import { Paper, Box, Typography, Stack } from '@mui/material';

const KPIStat = ({ title, value, subtitle, delta, deltaPositive }) => {
  return (
    <Paper sx={{ p: 2.5 }}>
      <Typography variant="overline" color="text.secondary">{title}</Typography>
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mt: 0.5 }}>
        <Typography variant="h4">{value}</Typography>
        {delta && (
          <Box component="span" sx={{ color: deltaPositive ? 'success.main' : 'error.main', fontWeight: 600 }}>
            {delta}
          </Box>
        )}
      </Stack>
      {subtitle && (
        <Typography variant="caption" color="text.secondary">{subtitle}</Typography>
      )}
    </Paper>
  );
};

export default KPIStat;
