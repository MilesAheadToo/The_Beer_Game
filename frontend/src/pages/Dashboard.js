import React from 'react';
import { Box, Grid, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import FilterBar from '../components/FilterBar';
import KPIStat from '../components/KPIStat';
import ChartCard from '../components/ChartCard';
import SkuTable from '../components/SkuTable';

// Sample data for the template-like dashboard
const demandSeries = [
  { name: 'W1', actual: 2100, forecast: 2200, target: 2000 },
  { name: 'W2', actual: 2250, forecast: 2300, target: 2050 },
  { name: 'W3', actual: 2150, forecast: 2350, target: 2100 },
  { name: 'W4', actual: 2300, forecast: 2400, target: 2100 },
  { name: 'W5', actual: 2400, forecast: 2380, target: 2150 },
  { name: 'W6', actual: 2350, forecast: 2450, target: 2150 },
  { name: 'W7', actual: 2420, forecast: 2500, target: 2200 },
  { name: 'W8', actual: 2380, forecast: 2480, target: 2200 },
  { name: 'W9', actual: 2450, forecast: 2550, target: 2250 },
  { name: 'W10', actual: 2480, forecast: 2580, target: 2250 },
  { name: 'W11', actual: 2460, forecast: 2600, target: 2300 },
  { name: 'W12', actual: 2520, forecast: 2650, target: 2300 },
];

const stockVsSafety = [
  { name: 'Widget A', stock: 1800, safety: 400 },
  { name: 'Widget B', stock: 900, safety: 300 },
  { name: 'Component C', stock: 7500, safety: 800 },
  { name: 'Assembly D', stock: 1200, safety: 350 },
  { name: 'Module E', stock: 320, safety: 100 },
  { name: 'Part F', stock: 6780, safety: 500 },
];

const stockVsForecast = [
  { name: 'Widget A', stock: 1800, forecast: 2200 },
  { name: 'Widget B', stock: 900, forecast: 1200 },
  { name: 'Component C', stock: 7500, forecast: 3900 },
  { name: 'Assembly D', stock: 1200, forecast: 1500 },
  { name: 'Module E', stock: 320, forecast: 500 },
  { name: 'Part F', stock: 6780, forecast: 5200 },
];

const Dashboard = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Demand Planning System
      </Typography>

      <FilterBar />

      {/* KPI cards */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} md={3}>
          <KPIStat title="Total Demand Forecast" value="248,500" subtitle="units" delta="+1.2% from last period" deltaPositive />
        </Grid>
        <Grid item xs={12} md={3}>
          <KPIStat title="Current Inventory" value="186,240" subtitle="units" delta="-3.5% from last period" />
        </Grid>
        <Grid item xs={12} md={3}>
          <KPIStat title="Forecast Accuracy" value="87.4%" subtitle="+2.7% from last period" delta="+2.7%" deltaPositive />
        </Grid>
        <Grid item xs={12} md={3}>
          <KPIStat title="Stockout Risk" value="3" subtitle="SKUs at risk" />
        </Grid>
      </Grid>

      {/* Demand Forecast Overview */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12}>
          <ChartCard title="Demand Forecast Overview" subtitle="Historical actuals vs forecasted demand for the next 12 weeks">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={demandSeries}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#1e88e5" name="Actual Demand" />
                <Line type="monotone" dataKey="forecast" stroke="#e53935" name="Forecasted Demand" strokeDasharray="4 2" />
                <Line type="monotone" dataKey="target" stroke="#43a047" name="Target Demand" />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        </Grid>
      </Grid>

      {/* Bar charts row */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} md={6}>
          <ChartCard title="Current Inventory vs Safety Stock" subtitle="Monitor stock levels against safety thresholds" height={300}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stockVsSafety}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" interval={0} angle={-20} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="stock" fill="#1e88e5" name="Current Stock" />
                <Bar dataKey="safety" fill="#e53935" name="Safety Stock" />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </Grid>
        <Grid item xs={12} md={6}>
          <ChartCard title="Stock vs Forecast Demand" subtitle="Compare current inventory with projected demand" height={300}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stockVsForecast}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" interval={0} angle={-20} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="stock" fill="#1e88e5" name="Current Stock" />
                <Bar dataKey="forecast" fill="#43a047" name="Forecast Demand" />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </Grid>
      </Grid>

      {/* SKU Overview Table */}
      <Typography variant="h6" gutterBottom>SKU Overview</Typography>
      <SkuTable />
    </Box>
  );
};

export default Dashboard;
