import React from 'react';
import { Box, Grid, FormControl, InputLabel, Select, MenuItem, Button, Stack } from '@mui/material';
import FileDownloadIcon from '@mui/icons-material/FileDownload';

const FilterBar = () => {
  const [quarter, setQuarter] = React.useState('Q1');
  const [year, setYear] = React.useState('2025');
  const [scope, setScope] = React.useState('All SKU');
  const [granularity, setGranularity] = React.useState('Weekly');

  return (
    <Box sx={{ mb: 2 }}>
      <Grid container spacing={2} alignItems="center">
        <Grid item xs={12} md={8}>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
            <FormControl size="small" sx={{ minWidth: 110 }}>
              <InputLabel>Quarter</InputLabel>
              <Select label="Quarter" value={quarter} onChange={(e) => setQuarter(e.target.value)}>
                {['Q1','Q2','Q3','Q4'].map(q => (
                  <MenuItem key={q} value={q}>{q}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 110 }}>
              <InputLabel>Year</InputLabel>
              <Select label="Year" value={year} onChange={(e) => setYear(e.target.value)}>
                {['2024','2025','2026'].map(y => (
                  <MenuItem key={y} value={y}>{y}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 140 }}>
              <InputLabel>Scope</InputLabel>
              <Select label="Scope" value={scope} onChange={(e) => setScope(e.target.value)}>
                {['All SKU','Top 20','Category A'].map(s => (
                  <MenuItem key={s} value={s}>{s}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 140 }}>
              <InputLabel>Granularity</InputLabel>
              <Select label="Granularity" value={granularity} onChange={(e) => setGranularity(e.target.value)}>
                {['Daily','Weekly','Monthly'].map(g => (
                  <MenuItem key={g} value={g}>{g}</MenuItem>
                ))}
              </Select>
            </FormControl>
          </Stack>
        </Grid>
        <Grid item xs={12} md={4}>
          <Stack direction="row" spacing={1} justifyContent={{ xs: 'flex-start', md: 'flex-end' }}>
            <Button variant="outlined" startIcon={<FileDownloadIcon />}>Export</Button>
          </Stack>
        </Grid>
      </Grid>
    </Box>
  );
};

export default FilterBar;
