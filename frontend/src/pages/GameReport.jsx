import React, { useEffect, useMemo, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Button,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Chip,
} from '@mui/material';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';
import PageLayout from '../components/PageLayout';
import { mixedGameApi } from '../services/api';

const roles = ['retailer', 'wholesaler', 'distributor', 'manufacturer'];
const roleColors = {
  retailer: '#2563eb',
  wholesaler: '#ec4899',
  distributor: '#f97316',
  manufacturer: '#22c55e',
};

const GameReport = () => {
  const { gameId } = useParams();
  const navigate = useNavigate();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchReport = async () => {
      try {
        setLoading(true);
        const data = await mixedGameApi.getReport(gameId);
        setReport(data);
        setError(null);
      } catch (err) {
        console.error('Failed to load game report', err);
        setError(err?.response?.data?.detail || 'Failed to load report');
      } finally {
        setLoading(false);
      }
    };
    fetchReport();
  }, [gameId]);

  const ordersChartData = useMemo(() => {
    if (!report) return [];
    const roundSet = new Set();
    (report.demand_series || []).forEach((p) => roundSet.add(p.round));
    Object.values(report.order_series || {}).forEach((series) => series.forEach((p) => roundSet.add(p.round)));
    const rounds = Array.from(roundSet).sort((a, b) => a - b);

    const demandMap = new Map((report.demand_series || []).map((p) => [p.round, p.demand]));
    const roleSeries = report.order_series || {};

    return rounds.map((round) => {
      const dataPoint = { round, demand: demandMap.get(round) ?? 0 };
      roles.forEach((role) => {
        const series = roleSeries[role] || [];
        const match = series.find((p) => p.round === round);
        dataPoint[role] = match ? match.quantity : 0;
      });
      return dataPoint;
    });
  }, [report]);

  const inventoryChartData = useMemo(() => {
    if (!report?.inventory_series) return [];
    return [...report.inventory_series].sort((a, b) => a.round - b.round);
  }, [report]);

  const roundsTable = useMemo(() => {
    if (!report?.history) return [];
    return [...report.history].sort((a, b) => b.round - a.round);
  }, [report]);

  const totals = report?.totals || {};

  return (
    <PageLayout title={report ? `Game Report: ${report.name}` : 'Game Report'}>
      <Box p={3}>
        <Button variant="outlined" onClick={() => navigate(`/games/${gameId}`)} sx={{ mb: 3 }}>
          Back to Game Board
        </Button>

        {loading ? (
          <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
            <CircularProgress />
          </Box>
        ) : error ? (
          <Paper sx={{ p: 4 }}>
            <Typography color="error">{error}</Typography>
          </Paper>
        ) : (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Summary
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Progression Mode: <Chip size="small" label={String(report.progression_mode || 'supervised').replace(/^./, (s) => s.toUpperCase())} sx={{ ml: 1 }} />
                </Typography>
                <Typography variant="h5" sx={{ fontWeight: 700 }}>
                  Total Supply Chain Cost: ${report.total_cost?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </Typography>
                <Table size="small" sx={{ mt: 2 }}>
                  <TableHead>
                    <TableRow>
                      <TableCell>Role</TableCell>
                      <TableCell align="right">Orders</TableCell>
                      <TableCell align="right">Holding Cost</TableCell>
                      <TableCell align="right">Backlog Cost</TableCell>
                      <TableCell align="right">Total Cost</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {roles.map((role) => {
                      const metrics = totals[role] || {};
                      return (
                        <TableRow key={role}>
                          <TableCell sx={{ textTransform: 'capitalize' }}>{role}</TableCell>
                          <TableCell align="right">{(metrics.orders ?? 0).toLocaleString()}</TableCell>
                          <TableCell align="right">${(metrics.holding_cost ?? 0).toFixed(2)}</TableCell>
                          <TableCell align="right">${(metrics.backlog_cost ?? 0).toFixed(2)}</TableCell>
                          <TableCell align="right">${(metrics.total_cost ?? 0).toFixed(2)}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </Paper>
            </Grid>

            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Demand vs Orders
                </Typography>
                <Box sx={{ width: '100%', height: 280 }}>
                  <ResponsiveContainer>
                    <LineChart data={ordersChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="demand" stroke="#111827" strokeWidth={2} name="Demand" />
                      {roles.map((role) => (
                        <Line
                          key={role}
                          type="monotone"
                          dataKey={role}
                          stroke={roleColors[role]}
                          name={role.charAt(0).toUpperCase() + role.slice(1)}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </Paper>
            </Grid>

            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Inventory Position by Facility
                </Typography>
                <Box sx={{ width: '100%', height: 320 }}>
                  <ResponsiveContainer>
                    <LineChart data={inventoryChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="round" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      {roles.map((role) => (
                        <Line
                          key={role}
                          type="monotone"
                          dataKey={role}
                          stroke={roleColors[role]}
                          name={role.charAt(0).toUpperCase() + role.slice(1)}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </Paper>
            </Grid>

            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Round Details
                </Typography>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Round</TableCell>
                      <TableCell>Demand</TableCell>
                      {roles.map((role) => (
                        <TableCell key={role} align="right" sx={{ textTransform: 'capitalize' }}>
                          {role}
                        </TableCell>
                      ))}
                      <TableCell align="right">Total Cost</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {roundsTable.map((entry) => (
                      <TableRow key={entry.round}>
                        <TableCell>{entry.round}</TableCell>
                        <TableCell>{entry.demand}</TableCell>
                        {roles.map((role) => (
                          <TableCell key={role} align="right">
                            {entry.orders?.[role]?.quantity ?? 0}
                          </TableCell>
                        ))}
                        <TableCell align="right">${(entry.total_cost ?? 0).toFixed(2)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Paper>
            </Grid>
          </Grid>
        )}
      </Box>
    </PageLayout>
  );
};

export default GameReport;
