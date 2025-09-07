import React, { useState } from 'react';
import PageLayout from '../components/PageLayout';
import {
  Box,
  Typography,
  Grid,
  Tabs,
  Tab,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  IconButton,
} from '@mui/material';
import {
  Timeline as TimelineIcon,
  ShowChart as ChartIcon,
  TableChart as TableIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  FilterList as FilterIcon,
} from '@mui/icons-material';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

// Sample data for analysis
const sampleTimeSeriesData = [
  { week: 'W1', inventory: 4000, demand: 4200, orders: 5000 },
  { week: 'W2', inventory: 4800, demand: 3800, orders: 4500 },
  { week: 'W3', inventory: 5500, demand: 5200, orders: 5000 },
  { week: 'W4', inventory: 4300, demand: 4500, orders: 4800 },
  { week: 'W5', inventory: 3700, demand: 4000, orders: 4500 },
  { week: 'W6', inventory: 4200, demand: 4100, orders: 5000 },
];

const samplePerformanceData = [
  { metric: 'Total Cost', value: 24567, unit: '$', change: 5.2, trend: 'up' },
  { metric: 'Average Inventory', value: 2345, unit: 'units', change: -2.1, trend: 'down' },
  { metric: 'Service Level', value: 94.5, unit: '%', change: 1.2, trend: 'up' },
  { metric: 'Order Fulfillment', value: 98.2, unit: '%', change: 0.8, trend: 'up' },
  { metric: 'Backorders', value: 45, unit: 'units', change: -12.3, trend: 'down' },
  { metric: 'Lead Time', value: 2.3, unit: 'days', change: -0.5, trend: 'down' },
];

const sampleBullwhipData = [
  { name: 'Retailer', value: 1.2 },
  { name: 'Distributor', value: 1.8 },
  { name: 'Wholesaler', value: 2.3 },
  { name: 'Factory', value: 2.9 },
];

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

const Analysis = () => {
  const [tabValue, setTabValue] = useState(0);
  const [timeRange, setTimeRange] = useState('last6');
  const [nodeFilter, setNodeFilter] = useState('all');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(5);

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleTimeRangeChange = (event) => {
    setTimeRange(event.target.value);
  };

  const handleNodeFilterChange = (event) => {
    setNodeFilter(event.target.value);
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const renderTabContent = () => {
    switch (tabValue) {
      case 0: // Overview
        return (
          <Grid container spacing={3}>
            {samplePerformanceData.map((item, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      {item.metric}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'flex-end', mb: 1 }}>
                      <Typography variant="h4" component="div">
                        {item.value}
                      </Typography>
                      <Typography color="textSecondary" sx={{ ml: 1, mb: 0.5 }}>
                        {item.unit}
                      </Typography>
                    </Box>
                    <Typography
                      variant="body2"
                      sx={{
                        color: item.trend === 'up' ? 'success.main' : 'error.main',
                        display: 'flex',
                        alignItems: 'center',
                      }}
                    >
                      {item.trend === 'up' ? '↑' : '↓'} {Math.abs(item.change)}% from last period
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Inventory & Demand Over Time
                  </Typography>
                  <Box sx={{ height: 400 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={sampleTimeSeriesData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="week" />
                        <YAxis yAxisId="left" />
                        <YAxis yAxisId="right" orientation="right" />
                        <Tooltip />
                        <Legend />
                        <Line yAxisId="left" type="monotone" dataKey="inventory" stroke="#8884d8" name="Inventory Level" />
                        <Line yAxisId="right" type="monotone" dataKey="demand" stroke="#82ca9d" name="Demand" />
                        <Line yAxisId="right" type="monotone" dataKey="orders" stroke="#ffc658" name="Orders" />
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        );
      case 1: // Bullwhip Effect
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Bullwhip Effect Analysis
                  </Typography>
                  <Typography variant="body2" color="textSecondary" paragraph>
                    The bullwhip effect shows how demand variability increases as we move up the supply chain.
                  </Typography>
                  <Box sx={{ height: 300, mt: 3 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={sampleBullwhipData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis label={{ value: 'Variability', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="value" name="Demand Variability" fill="#8884d8">
                          {sampleBullwhipData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Bullwhip Effect by Node
                  </Typography>
                  <Box sx={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={sampleBullwhipData}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          outerRadius={100}
                          fill="#8884d8"
                          dataKey="value"
                          nameKey="name"
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                        >
                          {sampleBullwhipData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Bullwhip Effect Mitigation Strategies
                  </Typography>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle1" gutterBottom>
                        Causes of Bullwhip Effect:
                      </Typography>
                      <ul>
                        <li>Demand forecast updating</li>
                        <li>Order batching</li>
                        <li>Price fluctuations</li>
                        <li>Rationing and shortage gaming</li>
                      </ul>
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Typography variant="subtitle1" gutterBottom>
                        Mitigation Strategies:
                      </Typography>
                      <ul>
                        <li>Implement Vendor Managed Inventory (VMI)</li>
                        <li>Improve information sharing</li>
                        <li>Reduce lead times</li>
                        <li>Use smaller order quantities</li>
                      </ul>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        );
      case 2: // Detailed Analysis
        return (
          <Box>
            <Grid container spacing={3} sx={{ mb: 3 }}>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Time Range</InputLabel>
                  <Select value={timeRange} onChange={handleTimeRangeChange} label="Time Range">
                    <MenuItem value="last6">Last 6 Weeks</MenuItem>
                    <MenuItem value="last12">Last 12 Weeks</MenuItem>
                    <MenuItem value="ytd">Year to Date</MenuItem>
                    <MenuItem value="custom">Custom Range</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Node Filter</InputLabel>
                  <Select value={nodeFilter} onChange={handleNodeFilterChange} label="Node Filter">
                    <MenuItem value="all">All Nodes</MenuItem>
                    <MenuItem value="retailer">Retailer</MenuItem>
                    <MenuItem value="distributor">Distributor</MenuItem>
                    <MenuItem value="warehouse">Warehouse</MenuItem>
                    <MenuItem value="factory">Factory</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={4} sx={{ display: 'flex', gap: 1 }}>
                <Button variant="outlined" startIcon={<FilterIcon />} fullWidth>
                  Filters
                </Button>
                <Button variant="outlined" startIcon={<DownloadIcon />}>
                  Export
                </Button>
              </Grid>
            </Grid>

            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Performance Metrics</Typography>
                  <IconButton size="small">
                    <RefreshIcon fontSize="small" />
                  </IconButton>
                </Box>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Metric</TableCell>
                        <TableCell align="right">Current</TableCell>
                        <TableCell align="right">Min</TableCell>
                        <TableCell align="right">Max</TableCell>
                        <TableCell align="right">Avg</TableCell>
                        <TableCell align="right">Target</TableCell>
                        <TableCell align="right">Variance</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {samplePerformanceData
                        .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                        .map((row, index) => (
                          <TableRow key={index} hover>
                            <TableCell>{row.metric}</TableCell>
                            <TableCell align="right">
                              {row.value} {row.unit}
                            </TableCell>
                            <TableCell align="right">
                              {(row.value * 0.8).toFixed(1)} {row.unit}
                            </TableCell>
                            <TableCell align="right">
                              {(row.value * 1.3).toFixed(1)} {row.unit}
                            </TableCell>
                            <TableCell align="right">
                              {(row.value * 1.05).toFixed(1)} {row.unit}
                            </TableCell>
                            <TableCell align="right">
                              {(row.value * 0.95).toFixed(1)} {row.unit}
                            </TableCell>
                            <TableCell
                              align="right"
                              sx={{
                                color: row.trend === 'up' ? 'success.main' : 'error.main',
                                fontWeight: 'medium',
                              }}
                            >
                              {row.trend === 'up' ? '+' : ''}
                              {row.change}%
                            </TableCell>
                          </TableRow>
                        ))}
                    </TableBody>
                  </Table>
                </TableContainer>
                <TablePagination
                  rowsPerPageOptions={[5, 10, 25]}
                  component="div"
                  count={samplePerformanceData.length}
                  rowsPerPage={rowsPerPage}
                  page={page}
                  onPageChange={handleChangePage}
                  onRowsPerPageChange={handleChangeRowsPerPage}
                />
              </CardContent>
            </Card>
          </Box>
        );
      default:
        return null;
    }
  };

  return (
    <PageLayout title="Supply Chain Analysis">
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 4 }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          aria-label="analysis tabs"
          sx={{
            '& .MuiTabs-flexContainer': {
              gap: 2
            },
            '& .MuiTab-root': {
              minHeight: 48,
              fontWeight: 500,
              '&.Mui-selected': {
                color: 'primary.main',
                fontWeight: 600
              }
            }
          }}
        >
          <Tab icon={<TimelineIcon />} label="Overview" />
          <Tab icon={<ChartIcon />} label="Bullwhip Effect" />
          <Tab icon={<TableIcon />} label="Detailed Analysis" />
        </Tabs>
      </Box>

      {renderTabContent()}
    </PageLayout>
  );
}

export default Analysis;
