import React from 'react';
import { Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip, IconButton, Typography } from '@mui/material';
import EditIcon from '@mui/icons-material/Edit';

const rows = [
  { sku: 'SKU-001', product: 'Premium Widget A', category: 'Electronics', current: 2810, forecast: 3200, safety: 500, lead: '14d', trend: 'up', risk: 'low', accuracy: '92.4%' },
  { sku: 'SKU-012', product: 'Industrial Tool I', category: 'Tools', current: 2340, forecast: 2650, safety: 250, lead: '25d', trend: 'down', risk: 'medium', accuracy: '87.6%' },
];

const riskColor = (risk) => ({
  low: 'success',
  medium: 'warning',
  high: 'error',
}[risk] || 'default');

const TrendCell = ({ value }) => (
  <Typography sx={{ color: value === 'up' ? 'success.main' : value === 'down' ? 'error.main' : 'text.secondary' }}>
    {value === 'up' ? '↗' : value === 'down' ? '↘' : '→'}
  </Typography>
);

const SkuTable = () => (
  <Paper sx={{ p: 0 }}>
    <TableContainer>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>SKU</TableCell>
            <TableCell>Product Name</TableCell>
            <TableCell>Category</TableCell>
            <TableCell align="right">Current Stock</TableCell>
            <TableCell align="right">Forecast Demand</TableCell>
            <TableCell align="right">Safety Stock</TableCell>
            <TableCell>Lead Time</TableCell>
            <TableCell>Trend</TableCell>
            <TableCell>Risk Level</TableCell>
            <TableCell>Accuracy</TableCell>
            <TableCell align="center">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((r) => (
            <TableRow key={r.sku} hover>
              <TableCell>{r.sku}</TableCell>
              <TableCell>{r.product}</TableCell>
              <TableCell>{r.category}</TableCell>
              <TableCell align="right">{r.current.toLocaleString()}</TableCell>
              <TableCell align="right">{r.forecast.toLocaleString()}</TableCell>
              <TableCell align="right">{r.safety.toLocaleString()}</TableCell>
              <TableCell>{r.lead}</TableCell>
              <TableCell><TrendCell value={r.trend} /></TableCell>
              <TableCell>
                <Chip size="small" label={r.risk} color={riskColor(r.risk)} variant="outlined" sx={{ textTransform: 'capitalize' }} />
              </TableCell>
              <TableCell>{r.accuracy}</TableCell>
              <TableCell align="center">
                <IconButton size="small"><EditIcon fontSize="small" /></IconButton>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  </Paper>
);

export default SkuTable;
