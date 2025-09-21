import React, { useEffect, useState } from 'react';
import {
  Box,
  Button,
  Typography,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Card,
  CardContent,
  Paper,
  Step,
  StepLabel,
  Stepper,
  IconButton,
  Tooltip,
  Switch,
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

const demandPatternOptions = [
  { value: 'classic_step', label: 'Classic (Step Change)' },
  { value: 'random', label: 'Random' },
  { value: 'seasonal', label: 'Seasonal' },
  { value: 'custom', label: 'Custom Scenario' },
];

const llmModelOptions = [
  { value: 'opt-66-mini', label: 'opt-66-mini' },
  { value: 'gpt-4o-mini', label: 'gpt-4o-mini' },
  { value: 'llama-3.1-8b', label: 'Llama 3.1 8B' },
];

const ParameterSlider = ({
  label,
  description,
  value,
  min,
  max,
  step = 1,
  onChange,
  valueFormatter,
  disabled = false,
  sx,
}) => (
  <Box sx={{ ...sx, opacity: disabled ? 0.6 : 1 }}>
    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
      <Box>
        <Typography variant="subtitle2" color="text.primary">
          {label}
        </Typography>
        {description && (
          <Typography variant="caption" color="text.secondary">
            {description}
          </Typography>
        )}
      </Box>
      <Typography variant="subtitle2" color="text.secondary">
        {valueFormatter ? valueFormatter(value) : value}
      </Typography>
    </Box>
    <Slider
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={onChange}
      sx={{ mt: 2 }}
      valueLabelDisplay="off"
      disabled={disabled}
    />
  </Box>
);

const ToggleControl = ({ label, description, checked, onChange, disabled = false }) => (
  <Box
    sx={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      py: 0.5,
      opacity: disabled ? 0.6 : 1,
    }}
  >
    <Box sx={{ pr: 2 }}>
      <Typography variant="subtitle2" color="text.primary">
        {label}
      </Typography>
      {description && (
        <Typography variant="caption" color="text.secondary">
          {description}
        </Typography>
      )}
    </Box>
    <Switch checked={checked} onChange={onChange} color="primary" disabled={disabled} />
  </Box>
);

const SummaryItem = ({ label, value }) => (
  <Box>
    <Typography variant="caption" color="text.secondary">
      {label}
    </Typography>
    <Typography variant="subtitle1" color="text.primary">
      {value}
    </Typography>
  </Box>
);

const SummaryRow = ({ label, value }) => (
  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', py: 0.75 }}>
    <Typography variant="body2" color="text.secondary">
      {label}
    </Typography>
    <Typography variant="body2" color="text.primary">
      {value}
    </Typography>
  </Box>
);

const formatBoolean = (value) => (value ? 'Enabled' : 'Disabled');
const formatWeeks = (value) => `${value} week${value === 1 ? '' : 's'}`;
const formatUnits = (value) => `${value} unit${value === 1 ? '' : 's'}`;
const formatCurrencyPerUnit = (value) => `$${value.toFixed(2)}/unit/week`;

const Simulation = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [simulationState, setSimulationState] = useState('idle'); // 'idle', 'running', 'paused', 'completed'
  const [currentWeek, setCurrentWeek] = useState(0);
  // Simulation speed state (commented out as it's not currently used)
  // const [simulationSpeed, setSimulationSpeed] = useState(1);
  const [config, setConfig] = useState({
    duration: 40,
    orderLeadTime: 2,
    shippingLeadTime: 1,
    productionDelay: 2,
    initialInventory: 12,
    holdingCost: 0.5,
    backorderCost: 1.0,
    infoSharing: true,
    historicalWeeks: 6,
    demandVolatility: true,
    confidenceThreshold: 60,
    pipelineInventory: true,
    centralizedForecast: false,
    manufacturerVisibility: false,
    demandPattern: 'classic_step',
    initialDemand: 4,
    newDemand: 8,
    demandChangeWeek: 6,
    llmModel: 'opt-66-mini',
    useRlModel: false,
  });

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const updateConfig = (field, value) => {
    setConfig((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSliderChange = (field) => (_, value) => {
    const resolvedValue = Array.isArray(value) ? value[0] : value;
    updateConfig(field, Number(resolvedValue));
  };

  const handleToggleChange = (field) => (event) => {
    updateConfig(field, event.target.checked);
  };

  const handleSelectChange = (field) => (event) => {
    updateConfig(field, event.target.value);
  };

  useEffect(() => {
    if (currentWeek > config.duration - 1) {
      setCurrentWeek(Math.max(0, config.duration - 1));
    }
  }, [config.duration, currentWeek]);

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
    if (currentWeek < config.duration - 1) {
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
            <Grid item xs={12} lg={7}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Simulation Parameters
                </Typography>
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <ParameterSlider
                      label="Number of weeks"
                      description="Duration of the simulation"
                      value={config.duration}
                      min={10}
                      max={52}
                      step={1}
                      onChange={handleSliderChange('duration')}
                      valueFormatter={formatWeeks}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <ParameterSlider
                      label="Order lead time (weeks)"
                      description="Time for players to receive orders"
                      value={config.orderLeadTime}
                      min={0}
                      max={8}
                      step={1}
                      onChange={handleSliderChange('orderLeadTime')}
                      valueFormatter={formatWeeks}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <ParameterSlider
                      label="Shipping lead time (weeks)"
                      description="Time for shipments to reach downstream partners"
                      value={config.shippingLeadTime}
                      min={0}
                      max={6}
                      step={1}
                      onChange={handleSliderChange('shippingLeadTime')}
                      valueFormatter={formatWeeks}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <ParameterSlider
                      label="Production delay (weeks)"
                      description="Time required to produce manufacturer orders"
                      value={config.productionDelay}
                      min={0}
                      max={6}
                      step={1}
                      onChange={handleSliderChange('productionDelay')}
                      valueFormatter={formatWeeks}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <ParameterSlider
                      label="Initial inventory"
                      description="Starting inventory for each player"
                      value={config.initialInventory}
                      min={0}
                      max={50}
                      step={1}
                      onChange={handleSliderChange('initialInventory')}
                      valueFormatter={formatUnits}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <ParameterSlider
                      label="Holding cost"
                      description="Cost per unit of inventory per week"
                      value={config.holdingCost}
                      min={0}
                      max={5}
                      step={0.1}
                      onChange={handleSliderChange('holdingCost')}
                      valueFormatter={formatCurrencyPerUnit}
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <ParameterSlider
                      label="Backorder cost"
                      description="Penalty per unit of unmet demand"
                      value={config.backorderCost}
                      min={0}
                      max={5}
                      step={0.1}
                      onChange={handleSliderChange('backorderCost')}
                      valueFormatter={formatCurrencyPerUnit}
                    />
                  </Grid>
                </Grid>
              </Paper>
            </Grid>
            <Grid item xs={12} lg={5}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Decision Support Features
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      <ToggleControl
                        label="Enable information sharing"
                        description="Share customer demand updates with all players"
                        checked={config.infoSharing}
                        onChange={handleToggleChange('infoSharing')}
                      />
                      <ParameterSlider
                        label="Historical weeks to share"
                        description="Amount of past demand data included in updates"
                        value={config.historicalWeeks}
                        min={0}
                        max={12}
                        step={1}
                        onChange={handleSliderChange('historicalWeeks')}
                        valueFormatter={formatWeeks}
                        disabled={!config.infoSharing}
                      />
                      <ToggleControl
                        label="Demand analysis with volatility insights"
                        description="Provide volatility commentary based on demand changes"
                        checked={config.demandVolatility}
                        onChange={handleToggleChange('demandVolatility')}
                      />
                      <ParameterSlider
                        label="Confidence threshold"
                        description="Minimum confidence required to surface volatility guidance"
                        value={config.confidenceThreshold}
                        min={0}
                        max={100}
                        step={5}
                        onChange={handleSliderChange('confidenceThreshold')}
                        valueFormatter={(value) => `${value}%`}
                        disabled={!config.demandVolatility}
                      />
                      <ToggleControl
                        label="Pipeline inventory sharing"
                        description="Show upstream orders and shipments to all players"
                        checked={config.pipelineInventory}
                        onChange={handleToggleChange('pipelineInventory')}
                      />
                      <ToggleControl
                        label="Centralized demand forecast"
                        description="Share a central demand forecast with each player"
                        checked={config.centralizedForecast}
                        onChange={handleToggleChange('centralizedForecast')}
                      />
                      <ToggleControl
                        label="Supplier inventory visibility"
                        description="Reveal upstream manufacturer inventory levels"
                        checked={config.manufacturerVisibility}
                        onChange={handleToggleChange('manufacturerVisibility')}
                      />
                    </Box>
                  </Paper>
                </Grid>
                <Grid item xs={12}>
                  <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Demand Pattern
                    </Typography>
                    <FormControl fullWidth margin="normal">
                      <InputLabel>Demand Pattern</InputLabel>
                      <Select
                        value={config.demandPattern}
                        onChange={handleSelectChange('demandPattern')}
                        label="Demand Pattern"
                      >
                        {demandPatternOptions.map((option) => (
                          <MenuItem key={option.value} value={option.value}>
                            {option.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
                      <ParameterSlider
                        label="Initial demand"
                        description="Customer demand before any change"
                        value={config.initialDemand}
                        min={0}
                        max={20}
                        step={1}
                        onChange={handleSliderChange('initialDemand')}
                        valueFormatter={formatUnits}
                      />
                      <ParameterSlider
                        label="New demand after change"
                        description="Customer demand after the step change"
                        value={config.newDemand}
                        min={0}
                        max={30}
                        step={1}
                        onChange={handleSliderChange('newDemand')}
                        valueFormatter={formatUnits}
                        disabled={config.demandPattern !== 'classic_step'}
                      />
                      <ParameterSlider
                        label="Weeks before new demand"
                        description="Time before the demand shift takes effect"
                        value={config.demandChangeWeek}
                        min={1}
                        max={20}
                        step={1}
                        onChange={handleSliderChange('demandChangeWeek')}
                        valueFormatter={formatWeeks}
                        disabled={config.demandPattern !== 'classic_step'}
                      />
                    </Box>
                  </Paper>
                </Grid>
                <Grid item xs={12}>
                  <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Model Selection
                    </Typography>
                    <FormControl fullWidth margin="normal">
                      <InputLabel>LLM Model</InputLabel>
                      <Select
                        value={config.llmModel}
                        onChange={handleSelectChange('llmModel')}
                        label="LLM Model"
                      >
                        {llmModelOptions.map((option) => (
                          <MenuItem key={option.value} value={option.value}>
                            {option.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <ToggleControl
                      label="Enable RL model for decision making"
                      description="Use reinforcement learning support during the game"
                      checked={config.useRlModel}
                      onChange={handleToggleChange('useRlModel')}
                    />
                  </Paper>
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        );
      case 1: {
        const demandPatternLabel =
          demandPatternOptions.find((option) => option.value === config.demandPattern)?.label || config.demandPattern;
        const llmModelLabel =
          llmModelOptions.find((option) => option.value === config.llmModel)?.label || config.llmModel;
        const newDemandSummary =
          config.demandPattern === 'classic_step' ? formatUnits(config.newDemand) : 'N/A';
        const demandChangeSummary =
          config.demandPattern === 'classic_step' ? formatWeeks(config.demandChangeWeek) : 'N/A';

        return (
          <Grid container spacing={3}>
            <Grid item xs={12} lg={7}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Simulation Parameters Summary
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <SummaryItem label="Number of weeks" value={formatWeeks(config.duration)} />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <SummaryItem label="Order lead time" value={formatWeeks(config.orderLeadTime)} />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <SummaryItem label="Shipping lead time" value={formatWeeks(config.shippingLeadTime)} />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <SummaryItem label="Production delay" value={formatWeeks(config.productionDelay)} />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <SummaryItem label="Initial inventory" value={formatUnits(config.initialInventory)} />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <SummaryItem label="Holding cost" value={formatCurrencyPerUnit(config.holdingCost)} />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <SummaryItem label="Backorder cost" value={formatCurrencyPerUnit(config.backorderCost)} />
                  </Grid>
                </Grid>
              </Paper>
            </Grid>
            <Grid item xs={12} lg={5}>
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Decision Support Features
                    </Typography>
                    <SummaryRow label="Information sharing" value={formatBoolean(config.infoSharing)} />
                    <SummaryRow
                      label="Historical weeks"
                      value={config.infoSharing ? formatWeeks(config.historicalWeeks) : 'Not shared'}
                    />
                    <SummaryRow label="Volatility analysis" value={formatBoolean(config.demandVolatility)} />
                    <SummaryRow
                      label="Confidence threshold"
                      value={config.demandVolatility ? `${config.confidenceThreshold}%` : 'N/A'}
                    />
                    <SummaryRow label="Pipeline inventory" value={formatBoolean(config.pipelineInventory)} />
                    <SummaryRow label="Centralized forecast" value={formatBoolean(config.centralizedForecast)} />
                    <SummaryRow label="Manufacturer visibility" value={formatBoolean(config.manufacturerVisibility)} />
                  </Paper>
                </Grid>
                <Grid item xs={12}>
                  <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Demand Pattern
                    </Typography>
                    <SummaryRow label="Pattern" value={demandPatternLabel} />
                    <SummaryRow label="Initial demand" value={formatUnits(config.initialDemand)} />
                    <SummaryRow label="New demand" value={newDemandSummary} />
                    <SummaryRow label="Change occurs after" value={demandChangeSummary} />
                  </Paper>
                </Grid>
                <Grid item xs={12}>
                  <Paper sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      Models
                    </Typography>
                    <SummaryRow label="LLM model" value={llmModelLabel} />
                    <SummaryRow label="RL model" value={formatBoolean(config.useRlModel)} />
                  </Paper>
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        );
      }
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
                      disabled={currentWeek >= config.duration - 1 || simulationState === 'running'}
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
                        Week {currentWeek + 1} of {config.duration}
                      </Typography>
                    </Box>
                    <Box sx={{ width: '100%' }}>
                      <Slider
                        value={currentWeek}
                        min={0}
                        max={config.duration - 1}
                        onChange={(e, value) => setCurrentWeek(Array.isArray(value) ? value[0] : value)}
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
