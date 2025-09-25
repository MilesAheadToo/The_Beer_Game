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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ToggleButton,
  ToggleButtonGroup,
  Divider,
  Stack,
  Tooltip as MuiTooltip,
} from '@mui/material';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts';
import PageLayout from '../components/PageLayout';
import { mixedGameApi } from '../services/api';

const roles = ['retailer', 'wholesaler', 'distributor', 'manufacturer'];
const roleColors = {
  retailer: '#4f46e5',
  wholesaler: '#ec4899',
  distributor: '#f97316',
  manufacturer: '#10b981',
};

const GameReport = () => {
  const { gameId } = useParams();
  const navigate = useNavigate();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [roundSortAsc, setRoundSortAsc] = useState(false);
  const [commentDialog, setCommentDialog] = useState({ open: false, entry: null });
  const [roundViewMode, setRoundViewMode] = useState('compact');

  const roundCount = useMemo(() => {
    if (!report) return 0;
    if (typeof report.rounds_completed === 'number') return report.rounds_completed;
    return Array.isArray(report.history) ? report.history.length : 0;
  }, [report]);

  const demandStats = useMemo(() => {
    const series = report?.demand_series || [];
    if (!series.length) {
      return {
        initial: null,
        final: null,
        peak: null,
      };
    }

    const values = series.map((point) => Number(point.demand ?? 0));
    return {
      initial: values[0] ?? null,
      final: values[values.length - 1] ?? null,
      peak: Math.max(...values),
    };
  }, [report]);

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
    if (!report?.history) return [];
    return [...report.history]
      .sort((a, b) => a.round - b.round)
      .map((entry) => {
        const inventory = entry.inventory_positions || {};
        const backlog = entry.backlogs || {};
        const dataPoint = { round: entry.round };
        roles.forEach((role) => {
          const inventoryValue = Number(inventory?.[role] ?? 0);
          const backlogValue = Number(backlog?.[role] ?? 0);
          dataPoint[role] = inventoryValue - backlogValue;
        });
        return dataPoint;
      });
  }, [report]);

  const roundsTable = useMemo(() => {
    if (!report?.history) return [];
    const sorted = [...report.history].sort((a, b) => a.round - b.round);
    return roundSortAsc ? sorted : sorted.reverse();
  }, [report, roundSortAsc]);

  const totalOrdersPlaced = useMemo(() => {
    if (!report?.history) return 0;
    return report.history.reduce((acc, entry) => {
      const roundOrders = roles.reduce((sum, role) => sum + Number(entry.orders?.[role]?.quantity ?? 0), 0);
      return acc + roundOrders;
    }, 0);
  }, [report]);

  const averageOrderPerRound = roundCount > 0 ? totalOrdersPlaced / roundCount : 0;

  const openComments = (entry) => {
    setCommentDialog({ open: true, entry });
  };

  const closeComments = () => {
    setCommentDialog({ open: false, entry: null });
  };

  const totals = report?.totals || {};

  return (
    <PageLayout title={report ? `Game Report: ${report.name}` : 'Game Report'}>
      <Box
        sx={{
          p: { xs: 2, md: 3 },
          background: 'linear-gradient(135deg, #eef2ff 0%, #e0e7ff 45%, #f9fafb 100%)',
          minHeight: '100%',
        }}
      >
        <Button
          variant="outlined"
          onClick={() => navigate(`/games/${gameId}`)}
          sx={{ mb: 3, borderRadius: 2, textTransform: 'none', borderColor: '#4f46e5', color: '#4338ca' }}
        >
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
          <Box>
            <Paper
              elevation={0}
              sx={{
                p: { xs: 3, md: 4 },
                mb: 3,
                borderRadius: 4,
                background: 'linear-gradient(135deg, rgba(79,70,229,0.12) 0%, rgba(79,70,229,0.05) 60%, rgba(255,255,255,0.9) 100%)',
                border: '1px solid rgba(79,70,229,0.25)',
              }}
            >
              <Grid container spacing={3} alignItems="center">
                <Grid item xs={12} md={8}>
                  <Stack spacing={2}>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Chip
                        size="small"
                        color="primary"
                        label={String(report.status || 'completed').replace(/_/g, ' ')}
                        sx={{ textTransform: 'capitalize' }}
                      />
                      <Chip
                        size="small"
                        label={`Progression: ${String(report.progression_mode || 'supervised')
                          .replace(/_/g, ' ')
                          .replace(/^./, (s) => s.toUpperCase())}`}
                        sx={{ backgroundColor: 'rgba(99,102,241,0.14)', color: '#312e81', fontWeight: 600 }}
                      />
                    </Stack>
                    <Box>
                      <Typography variant="h4" sx={{ fontWeight: 700, color: '#1e1b4b' }}>
                        {report?.name || 'Supply Chain Simulation'}
                      </Typography>
                      <Typography variant="subtitle1" sx={{ color: '#4338ca' }}>
                        Total Supply Chain Cost
                      </Typography>
                      <Typography variant="h3" sx={{ fontWeight: 800, color: '#312e81' }}>
                        ${report.total_cost?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </Typography>
                    </Box>
                    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
                      <Box
                        sx={{
                          px: 2,
                          py: 1,
                          borderRadius: 3,
                          backgroundColor: 'rgba(79,70,229,0.08)',
                          display: 'flex',
                          flexDirection: 'column',
                          gap: 0.5,
                          minWidth: 160,
                        }}
                      >
                        <Typography variant="caption" sx={{ color: '#4338ca', fontWeight: 600 }}>
                          Rounds Completed
                        </Typography>
                        <Typography variant="h6" sx={{ fontWeight: 700, color: '#1e1b4b' }}>
                          {roundCount}
                        </Typography>
                      </Box>
                      <Box
                        sx={{
                          px: 2,
                          py: 1,
                          borderRadius: 3,
                          backgroundColor: 'rgba(14,165,233,0.08)',
                          display: 'flex',
                          flexDirection: 'column',
                          gap: 0.5,
                          minWidth: 160,
                        }}
                      >
                        <Typography variant="caption" sx={{ color: '#0ea5e9', fontWeight: 600 }}>
                          Avg Orders / Round
                        </Typography>
                        <Typography variant="h6" sx={{ fontWeight: 700, color: '#0f172a' }}>
                          {averageOrderPerRound.toFixed(1)}
                        </Typography>
                      </Box>
                      <Box
                        sx={{
                          px: 2,
                          py: 1,
                          borderRadius: 3,
                          backgroundColor: 'rgba(34,197,94,0.08)',
                          display: 'flex',
                          flexDirection: 'column',
                          gap: 0.5,
                          minWidth: 160,
                        }}
                      >
                        <Typography variant="caption" sx={{ color: '#15803d', fontWeight: 600 }}>
                          Demand Trend
                        </Typography>
                        <Typography variant="body2" sx={{ color: '#166534' }}>
                          {demandStats.initial !== null && demandStats.final !== null
                            ? `${demandStats.initial} → ${demandStats.final}`
                            : 'Not available'}
                        </Typography>
                        {demandStats.peak !== null && (
                          <Typography variant="caption" sx={{ color: '#16a34a' }}>
                            Peak {demandStats.peak}
                          </Typography>
                        )}
                      </Box>
                    </Stack>
                  </Stack>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Paper
                    elevation={0}
                    sx={{
                      p: 2,
                      borderRadius: 3,
                      backgroundColor: '#ffffff',
                      border: '1px solid rgba(79,70,229,0.16)',
                    }}
                  >
                    <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#4338ca', mb: 1 }}>
                      Simulation Snapshot
                    </Typography>
                    <Stack spacing={1.5}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body2" color="text.secondary">
                          Game ID
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          #{report.game_id}
                        </Typography>
                      </Box>
                      <Divider light sx={{ borderStyle: 'dashed' }} />
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body2" color="text.secondary">
                          Peak Demand
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {demandStats.peak ?? '—'}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body2" color="text.secondary">
                          Total Orders
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {totalOrdersPlaced.toLocaleString()}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body2" color="text.secondary">
                          Last Updated
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>
                          {roundsTable.length
                            ? (() => {
                                const latest = roundsTable[0];
                                if (!latest?.timestamp) return 'N/A';
                                try {
                                  return new Date(latest.timestamp).toLocaleString();
                                } catch (e) {
                                  return 'N/A';
                                }
                              })()
                            : 'N/A'}
                        </Typography>
                      </Box>
                    </Stack>
                  </Paper>
                </Grid>
              </Grid>
            </Paper>

            <Grid container spacing={3}>
              <Grid item xs={12} lg={5}>
                <Paper
                  elevation={0}
                  sx={{
                    p: 3,
                    borderRadius: 4,
                    height: '100%',
                    border: '1px solid rgba(148,163,184,0.3)',
                    backgroundColor: '#ffffff',
                  }}
                >
                  <Typography variant="subtitle1" sx={{ fontWeight: 700, color: '#1e293b', mb: 2 }}>
                    Cost Breakdown by Role
                  </Typography>
                  <Stack spacing={2}>
                    {roles.map((role) => {
                      const metrics = totals[role] || {};
                      const title = role.charAt(0).toUpperCase() + role.slice(1);
                      return (
                        <Paper
                          key={role}
                          variant="outlined"
                          sx={{
                            p: 2,
                            borderRadius: 3,
                            borderColor: 'rgba(148,163,184,0.4)',
                            background: 'linear-gradient(135deg, rgba(79,70,229,0.04), rgba(255,255,255,0.9))',
                          }}
                        >
                          <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                            <Stack direction="row" spacing={1} alignItems="center">
                              <Box
                                sx={{
                                  width: 10,
                                  height: 10,
                                  borderRadius: '50%',
                                  backgroundColor: roleColors[role],
                                }}
                              />
                              <Typography variant="subtitle2" sx={{ fontWeight: 700, textTransform: 'capitalize' }}>
                                {title}
                              </Typography>
                            </Stack>
                            <Typography variant="subtitle1" sx={{ fontWeight: 700, color: '#1e1b4b' }}>
                              ${Number(metrics.total_cost ?? 0).toFixed(2)}
                            </Typography>
                          </Stack>
                          <Grid container spacing={1}>
                            <Grid item xs={4}>
                              <MuiTooltip title="Total orders placed by this role">
                                <Typography variant="caption" color="text.secondary">
                                  Orders
                                </Typography>
                              </MuiTooltip>
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                {(metrics.orders ?? 0).toLocaleString()}
                              </Typography>
                            </Grid>
                            <Grid item xs={4}>
                              <MuiTooltip title="Holding cost accumulated">
                                <Typography variant="caption" color="text.secondary">
                                  Holding
                                </Typography>
                              </MuiTooltip>
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                ${Number(metrics.holding_cost ?? 0).toFixed(2)}
                              </Typography>
                            </Grid>
                            <Grid item xs={4}>
                              <MuiTooltip title="Backlog cost accumulated">
                                <Typography variant="caption" color="text.secondary">
                                  Backlog
                                </Typography>
                              </MuiTooltip>
                              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                ${Number(metrics.backlog_cost ?? 0).toFixed(2)}
                              </Typography>
                            </Grid>
                          </Grid>
                        </Paper>
                      );
                    })}
                  </Stack>
                </Paper>
              </Grid>
              <Grid item xs={12} lg={7}>
                <Paper
                  elevation={0}
                  sx={{
                    p: 3,
                    borderRadius: 4,
                    height: '100%',
                    border: '1px solid rgba(148,163,184,0.3)',
                    backgroundColor: '#ffffff',
                  }}
                >
                  <Typography variant="subtitle1" sx={{ fontWeight: 700, color: '#1e293b', mb: 2 }}>
                    Demand vs Orders
                  </Typography>
                  <Box sx={{ width: '100%', height: { xs: 260, md: 320 } }}>
                    <ResponsiveContainer>
                      <LineChart data={ordersChartData}>
                        <CartesianGrid strokeDasharray="4 4" stroke="rgba(148,163,184,0.4)" />
                        <XAxis dataKey="round" stroke="#475569" />
                        <YAxis stroke="#475569" />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Legend />
                        <Line type="monotone" dataKey="demand" stroke="#111827" strokeWidth={2} name="Demand" dot={false} />
                        {roles.map((role) => (
                          <Line
                            key={role}
                            type="monotone"
                            dataKey={role}
                            stroke={roleColors[role]}
                            name={role.charAt(0).toUpperCase() + role.slice(1)}
                            strokeWidth={2}
                            dot={false}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                </Paper>
              </Grid>

              <Grid item xs={12}>
                <Paper
                  elevation={0}
                  sx={{
                    p: 3,
                    borderRadius: 4,
                    border: '1px solid rgba(148,163,184,0.3)',
                    backgroundColor: '#ffffff',
                  }}
                >
                  <Typography variant="subtitle1" sx={{ fontWeight: 700, color: '#1e293b', mb: 2 }}>
                    Inventory Position by Facility
                  </Typography>
                  <Box sx={{ width: '100%', height: { xs: 280, md: 340 } }}>
                    <ResponsiveContainer>
                      <LineChart data={inventoryChartData}>
                        <CartesianGrid strokeDasharray="4 4" stroke="rgba(148,163,184,0.35)" />
                        <XAxis dataKey="round" stroke="#475569" />
                        <YAxis stroke="#475569" />
                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                        <Legend />
                        {roles.map((role) => (
                          <Line
                            key={role}
                            type="monotone"
                            dataKey={role}
                            stroke={roleColors[role]}
                            name={role.charAt(0).toUpperCase() + role.slice(1)}
                            strokeWidth={2}
                            dot={false}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </Box>
                </Paper>
              </Grid>

              <Grid item xs={12}>
                <Paper
                  elevation={0}
                  sx={{
                    p: 3,
                    borderRadius: 4,
                    border: '1px solid rgba(148,163,184,0.3)',
                    backgroundColor: '#ffffff',
                  }}
                >
                  <Box display="flex" flexDirection={{ xs: 'column', md: 'row' }} gap={2} alignItems={{ md: 'center' }} mb={2}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 700, color: '#1e293b' }}>
                      Round Details
                    </Typography>
                    <Box flexGrow={1} />
                    <ToggleButtonGroup
                      value={roundViewMode}
                      exclusive
                      size="small"
                      onChange={(_, value) => value && setRoundViewMode(value)}
                      sx={{
                        backgroundColor: 'rgba(226,232,240,0.6)',
                        borderRadius: 999,
                        '& .MuiToggleButton-root': {
                          textTransform: 'none',
                          border: 'none',
                          px: 2,
                          borderRadius: 999,
                        },
                        '& .Mui-selected': {
                          backgroundColor: '#4338ca',
                          color: '#fff',
                          '&:hover': {
                            backgroundColor: '#3730a3',
                          },
                        },
                      }}
                    >
                      <ToggleButton value="compact">Compact</ToggleButton>
                      <ToggleButton value="detailed">Detailed</ToggleButton>
                    </ToggleButtonGroup>
                  </Box>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell
                          onClick={() => setRoundSortAsc((prev) => !prev)}
                          sx={{ cursor: 'pointer', fontWeight: 700, color: '#4338ca' }}
                        >
                          Round {roundSortAsc ? '▲' : '▼'}
                        </TableCell>
                        <TableCell sx={{ fontWeight: 600 }}>Demand</TableCell>
                        {roles.map((role) => (
                          <TableCell key={role} align="right" sx={{ textTransform: 'capitalize', fontWeight: 600 }}>
                            {role}
                          </TableCell>
                        ))}
                        <TableCell align="right" sx={{ fontWeight: 600 }}>
                          Total Cost
                        </TableCell>
                        <TableCell align={roundViewMode === 'compact' ? 'center' : 'left'} sx={{ fontWeight: 600 }}>
                          Comments
                        </TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {roundsTable.map((entry) => (
                        <TableRow key={entry.round} hover sx={{ '&:nth-of-type(odd)': { backgroundColor: 'rgba(248,250,252,0.8)' } }}>
                          <TableCell>{entry.round}</TableCell>
                          <TableCell>{entry.demand}</TableCell>
                          {roles.map((role) => (
                            <TableCell key={role} align="right">
                              {entry.orders?.[role]?.quantity ?? 0}
                            </TableCell>
                          ))}
                          <TableCell align="right">${(entry.total_cost ?? 0).toFixed(2)}</TableCell>
                          <TableCell align={roundViewMode === 'compact' ? 'center' : 'left'}>
                            {roundViewMode === 'compact' ? (
                              <Button size="small" variant="outlined" onClick={() => openComments(entry)} sx={{ textTransform: 'none', borderRadius: 2 }}>
                                View
                              </Button>
                            ) : (
                              <Box>
                                {(() => {
                                  const nodes = roles
                                    .map((role) => {
                                      const comment = entry.orders?.[role]?.comment;
                                      if (!comment) return null;
                                      return (
                                        <Typography key={role} variant="body2" sx={{ mb: 0.5 }}>
                                          <strong>{role.charAt(0).toUpperCase() + role.slice(1)}:</strong> {comment}
                                        </Typography>
                                      );
                                    })
                                    .filter(Boolean);
                                  return nodes.length ? (
                                    nodes
                                  ) : (
                                    <Typography variant="body2" color="text.secondary">
                                      No comments recorded
                                    </Typography>
                                  );
                                })()}
                              </Box>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </Paper>
              </Grid>
            </Grid>
          </Box>
        )}
        <Dialog open={commentDialog.open} onClose={closeComments} maxWidth="sm" fullWidth>
          <DialogTitle>Round {commentDialog.entry?.round} Comments</DialogTitle>
          <DialogContent dividers>
            {roles.map((role) => {
              const comment = commentDialog.entry?.orders?.[role]?.comment;
              const submittedAt = commentDialog.entry?.orders?.[role]?.submitted_at;
              return (
                <List dense disablePadding key={role}>
                  <ListItem disableGutters>
                    <ListItemText
                      primary={role.charAt(0).toUpperCase() + role.slice(1)}
                      secondary={comment ? `${comment}${submittedAt ? ` (submitted ${new Date(submittedAt).toLocaleString()})` : ''}` : 'No comment'}
                    />
                  </ListItem>
                </List>
              );
            })}
          </DialogContent>
          <DialogActions>
            <Button onClick={closeComments}>Close</Button>
          </DialogActions>
        </Dialog>
      </Box>
    </PageLayout>
  );
};

export default GameReport;
