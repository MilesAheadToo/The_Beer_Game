import React, { useState } from 'react';
import {
  Box,
  Button,
  Typography,
  Grid,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Divider,
  Card,
  CardContent,
  Paper,
  Step,
  StepLabel,
  Stepper,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  SkipNext as NextIcon,
  SkipPrevious as PreviousIcon,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';

// Sample data for the simulation
const sampleDemandData = [
  { week: 'W1', demand: 4000 },
  { week: 'W2', demand: 3000 },
  { week: 'W3', demand: 5000 },
  { week: 'W4', demand: 2780 },
  { week: 'W5', demand: 1890 },
  { week: 'W6', demand: 2390 },
];

const sampleInventoryData = [
  { week: 'W1', inventory: 4000, reorder: 2000 },
  { week: 'W2', inventory: 3000, reorder: 2000 },
  { week: 'W3', inventory: 2000, reorder: 2000 },
  { week: 'W4', inventory: 2780, reorder: 2000 },
  { week: 'W5', inventory: 1890, reorder: 2000 },
  { week: 'W6', inventory: 2390, reorder: 2000 },
];

const simulationSteps = [
  'Configure Simulation',
  'Review Parameters',
  'Run Simulation',
  'Analyze Results',
];

const Simulation = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [simulationState, setSimulationState] = useState('idle'); // 'idle', 'running', 'paused', 'completed'
  const [currentWeek, setCurrentWeek] = useState(0);
  // Simulation speed state (commented out as it's not currently used)
  // const [simulationSpeed, setSimulationSpeed] = useState(1);
  const [parameters, setParameters] = useState({
    duration: 26, // weeks
    demandDistribution: 'normal',
    demandMean: 3000,
    demandStdDev: 500,
    leadTimeDistribution: 'lognormal',
    leadTimeMean: 2,
    leadTimeStdDev: 0.5,
    initialInventory: 10000,
    reorderPoint: 2000,
    orderQuantity: 5000,
  });

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleParameterChange = (param) => (event) => {
    const value = event.target ? event.target.value : event;
    setParameters((prev) => ({
      ...prev,
      [param]: value,
    }));
  };

  const handleStartSimulation = () => {
    setSimulationState('running');
    // In a real implementation, this would connect to the backend
  };

  const handlePauseSimulation = () => {
    setSimulationState('paused');
  };

  const handleStopSimulation = () => {
    setSimulationState('idle');
    setCurrentWeek(0);
  };

  const handleStepForward = () => {
    if (currentWeek < parameters.duration - 1) {
      setCurrentWeek(currentWeek + 1);
    } else {
      setSimulationState('completed');
    }
  };

  const handleStepBackward = () => {
    if (currentWeek > 0) {
      setCurrentWeek(currentWeek - 1);
    }
  };

  const renderStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Demand Parameters
              </Typography>
              <FormControl fullWidth sx={{ mb: 3 }}>
                <InputLabel>Demand Distribution</InputLabel>
                <Select
                  value={parameters.demandDistribution}
                  onChange={handleParameterChange('demandDistribution')}
                  label="Demand Distribution"
                >
                  <MenuItem value="normal">Normal</MenuItem>
                  <MenuItem value="lognormal">Log-normal</MenuItem>
                  <MenuItem value="poisson">Poisson</MenuItem>
                  <MenuItem value="constant">Constant</MenuItem>
                </Select>
              </FormControl>
              <TextField
                fullWidth
                type="number"
                label="Mean Demand (units/week)"
                value={parameters.demandMean}
                onChange={handleParameterChange('demandMean')}
                sx={{ mb: 3 }}
              />
              <TextField
                fullWidth
                type="number"
                label="Standard Deviation"
                value={parameters.demandStdDev}
                onChange={handleParameterChange('demandStdDev')}
                sx={{ mb: 3 }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Lead Time Parameters
              </Typography>
              <FormControl fullWidth sx={{ mb: 3 }}>
                <InputLabel>Lead Time Distribution</InputLabel>
                <Select
                  value={parameters.leadTimeDistribution}
                  onChange={handleParameterChange('leadTimeDistribution')}
                  label="Lead Time Distribution"
                >
                  <MenuItem value="lognormal">Log-normal</MenuItem>
                  <MenuItem value="normal">Normal</MenuItem>
                  <MenuItem value="constant">Constant</MenuItem>
                </Select>
              </FormControl>
              <TextField
                fullWidth
                type="number"
                label="Mean Lead Time (weeks)"
                value={parameters.leadTimeMean}
                onChange={handleParameterChange('leadTimeMean')}
                sx={{ mb: 3 }}
              />
              <TextField
                fullWidth
                type="number"
                label="Standard Deviation"
                value={parameters.leadTimeStdDev}
                onChange={handleParameterChange('leadTimeStdDev')}
                step={0.1}
                sx={{ mb: 3 }}
              />
            </Grid>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Inventory Policy
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Initial Inventory (units)"
                    value={parameters.initialInventory}
                    onChange={handleParameterChange('initialInventory')}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Reorder Point (units)"
                    value={parameters.reorderPoint}
                    onChange={handleParameterChange('reorderPoint')}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Order Quantity (units)"
                    value={parameters.orderQuantity}
                    onChange={handleParameterChange('orderQuantity')}
                  />
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        );
      case 1:
        return (
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Simulation Parameters Summary
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1">Demand Parameters</Typography>
                <Typography>Distribution: {parameters.demandDistribution}</Typography>
                <Typography>Mean: {parameters.demandMean} units/week</Typography>
                <Typography>Std Dev: {parameters.demandStdDev}</Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1">Lead Time Parameters</Typography>
                <Typography>Distribution: {parameters.leadTimeDistribution}</Typography>
                <Typography>Mean: {parameters.leadTimeMean} weeks</Typography>
                <Typography>Std Dev: {parameters.leadTimeStdDev}</Typography>
              </Grid>
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle1">Inventory Policy</Typography>
                <Typography>Initial Inventory: {parameters.initialInventory} units</Typography>
                <Typography>Reorder Point: {parameters.reorderPoint} units</Typography>
                <Typography>Order Quantity: {parameters.orderQuantity} units</Typography>
              </Grid>
              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle1">Simulation Duration</Typography>
                <Typography>{parameters.duration} weeks</Typography>
                <Slider
                  value={parameters.duration}
                  onChange={handleParameterChange('duration')}
                  min={4}
                  max={52}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${value} weeks`}
                  sx={{ mt: 3 }}
                />
              </Grid>
            </Grid>
          </Paper>
        );
      case 2:
        return (
          <Box>
            <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Box>
                <Typography variant="h6">Simulation Controls</Typography>
                <Typography variant="body2" color="textSecondary">
                  {simulationState === 'idle' && 'Ready to start simulation'}
                  {simulationState === 'running' && 'Simulation in progress...'}
                  {simulationState === 'paused' && 'Simulation paused'}
                  {simulationState === 'completed' && 'Simulation completed'}
                </Typography>
              </Box>
              <Box>
                <Tooltip title="Previous Step">
                  <span>
                    <IconButton
                      onClick={handleStepBackward}
                      disabled={currentWeek === 0 || simulationState === 'running'}
                      sx={{ mr: 1 }}
                    >
                      <PreviousIcon />
                    </IconButton>
                  </span>
                </Tooltip>
                {simulationState === 'running' ? (
                  <Tooltip title="Pause">
                    <IconButton onClick={handlePauseSimulation} color="primary" sx={{ mr: 1 }}>
                      <PauseIcon />
                    </IconButton>
                  </Tooltip>
                ) : (
                  <Tooltip title={simulationState === 'completed' ? 'Restart' : 'Start'}>
                    <IconButton
                      onClick={handleStartSimulation}
                      color="primary"
                      disabled={simulationState === 'completed'}
                      sx={{ mr: 1 }}
                    >
                      <PlayIcon />
                    </IconButton>
                  </Tooltip>
                )}
                <Tooltip title="Stop">
                  <IconButton
                    onClick={handleStopSimulation}
                    color="error"
                    disabled={simulationState === 'idle'}
                    sx={{ mr: 1 }}
                  >
                    <StopIcon />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Next Step">
                  <span>
                    <IconButton
                      onClick={handleStepForward}
                      disabled={currentWeek >= parameters.duration - 1 || simulationState === 'running'}
                    >
                      <NextIcon />
                    </IconButton>
                  </span>
                </Tooltip>
              </Box>
            </Box>

            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Demand Forecast
                    </Typography>
                    <Box sx={{ height: 300 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={sampleDemandData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="week" />
                          <YAxis />
                          <RechartsTooltip />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="demand"
                            stroke="#8884d8"
                            activeDot={{ r: 8 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Inventory Level
                    </Typography>
                    <Box sx={{ height: 300 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={sampleInventoryData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="week" />
                          <YAxis />
                          <RechartsTooltip />
                          <Legend />
                          <Line
                            type="monotone"
                            dataKey="inventory"
                            stroke="#82ca9d"
                            activeDot={{ r: 8 }}
                          />
                          <Line
                            type="monotone"
                            dataKey="reorder"
                            stroke="#ff7300"
                            strokeDasharray="5 5"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="h6">Simulation Progress</Typography>
                      <Typography variant="body2" color="textSecondary">
                        Week {currentWeek + 1} of {parameters.duration}
                      </Typography>
                    </Box>
                    <Box sx={{ width: '100%' }}>
                      <Slider
                        value={currentWeek}
                        min={0}
                        max={parameters.duration - 1}
                        onChange={(e, value) => setCurrentWeek(value)}
                        valueLabelDisplay="auto"
                        valueLabelFormat={(value) => `Week ${value + 1}`}
                        disabled={simulationState === 'running'}
                      />
                    </Box>
                    <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2" color="textSecondary">
                        Start
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        End
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        );
      case 3:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Simulation Results
            </Typography>
            <Typography variant="body1" paragraph>
              Simulation completed successfully. Here are the key performance indicators:
            </Typography>
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      Total Cost
                    </Typography>
                    <Typography variant="h4">$24,567</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      Average Inventory Level
                    </Typography>
                    <Typography variant="h4">2,345 units</Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={4}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      Service Level
                    </Typography>
                    <Typography variant="h4">94.5%</Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
            <Typography variant="h6" gutterBottom>
              Detailed Analysis
            </Typography>
            <Box sx={{ height: 400, mt: 3 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sampleInventoryData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="week" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <RechartsTooltip />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="inventory" stroke="#8884d8" name="Inventory Level" />
                  <Line yAxisId="right" type="monotone" dataKey="reorder" stroke="#82ca9d" name="Reorder Point" />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Box>
        );
      default:
        return 'Unknown step';
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Supply Chain Simulation
      </Typography>
      
      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {simulationSteps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <Box sx={{ mb: 4 }}>
        {renderStepContent(activeStep)}
      </Box>

      <Box sx={{ display: 'flex', justifyContent: 'space-between', pt: 2 }}>
        <Button
          disabled={activeStep === 0}
          onClick={handleBack}
          variant="outlined"
        >
          Back
        </Button>
        {activeStep === simulationSteps.length - 1 ? (
          <Button variant="contained" color="primary" onClick={handleStartSimulation}>
            Save Results
          </Button>
        ) : (
          <Button variant="contained" onClick={handleNext}>
            {activeStep === simulationSteps.length - 2 ? 'Finish' : 'Next'}
          </Button>
        )}
      </Box>
    </Box>
  );
};

export default Simulation;
