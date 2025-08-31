import React from 'react';
import { Box, Grid, Paper, Typography } from '@mui/material';
import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
  TimelineOppositeContent,
} from '@mui/lab';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Sample data for the dashboard
const recentActivities = [
  {
    time: '09:30 AM',
    title: 'New simulation started',
    description: 'Simulation #1234 has been initiated',
  },
  {
    time: 'Yesterday',
    title: 'Supply chain updated',
    description: 'Added new distributor node',
  },
  {
    time: 'Mar 15',
    title: 'New user registered',
    description: 'User: supply_chain_manager@example.com',
  },
];

const performanceData = [
  { name: 'Jan', value: 4000 },
  { name: 'Feb', value: 3000 },
  { name: 'Mar', value: 5000 },
  { name: 'Apr', value: 2780 },
  { name: 'May', value: 1890 },
  { name: 'Jun', value: 2390 },
];

const Dashboard = () => {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Performance Overview */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Performance Overview
            </Typography>
            <Box sx={{ height: 300, mt: 3 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="value" stroke="#8884d8" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Recent Activities */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Recent Activities
            </Typography>
            <Timeline position="right" sx={{ mt: 0, pt: 0 }}>
              {recentActivities.map((activity, index) => (
                <TimelineItem key={index}>
                  <TimelineOppositeContent color="textSecondary">
                    {activity.time}
                  </TimelineOppositeContent>
                  <TimelineSeparator>
                    <TimelineDot color="primary" />
                    {index < recentActivities.length - 1 && <TimelineConnector />}
                  </TimelineSeparator>
                  <TimelineContent>
                    <Typography variant="subtitle2">{activity.title}</Typography>
                    <Typography variant="body2" color="textSecondary">
                      {activity.description}
                    </Typography>
                  </TimelineContent>
                </TimelineItem>
              ))}
            </Timeline>
          </Paper>
        </Grid>

        {/* Quick Stats */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Supply Chain Stats
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h4">5</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Active Nodes
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={6}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h4">3</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Simulations
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={6}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h4">12</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Products
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={6}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h4">87%</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Avg. Efficiency
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Recent Simulations */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Simulations
            </Typography>
            <Box sx={{ mt: 2 }}>
              <Paper sx={{ p: 2, mb: 2 }}>
                <Grid container alignItems="center">
                  <Grid item xs={12} sm={4}>
                    <Typography variant="subtitle1">Simulation #1234</Typography>
                    <Typography variant="body2" color="textSecondary">
                      Started 2 hours ago
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Typography variant="body2">Status: Running</Typography>
                  </Grid>
                  <Grid item xs={12} sm={4} sx={{ textAlign: { sm: 'right' }, mt: { xs: 1, sm: 0 } }}>
                    <Typography variant="body2" color="primary" sx={{ cursor: 'pointer' }}>
                      View Details
                    </Typography>
                  </Grid>
                </Grid>
              </Paper>
              <Paper sx={{ p: 2 }}>
                <Grid container alignItems="center">
                  <Grid item xs={12} sm={4}>
                    <Typography variant="subtitle1">Simulation #1233</Typography>
                    <Typography variant="body2" color="textSecondary">
                      Completed yesterday
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Typography variant="body2">Status: Completed</Typography>
                  </Grid>
                  <Grid item xs={12} sm={4} sx={{ textAlign: { sm: 'right' }, mt: { xs: 1, sm: 0 } }}>
                    <Typography variant="body2" color="primary" sx={{ cursor: 'pointer' }}>
                      View Analysis
                    </Typography>
                  </Grid>
                </Grid>
              </Paper>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
