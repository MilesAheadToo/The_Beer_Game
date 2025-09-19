import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Divider,
  FormControl,
  FormControlLabel,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  Snackbar,
  Stack,
  Switch,
  TextField,
  Typography,
} from '@mui/material';
import PageLayout from '../../components/PageLayout';
import { mixedGameApi } from '../../services/api';

const DEFAULT_RANGES = {
  supply_leadtime: [0, 6],
  order_leadtime: [0, 6],
  init_inventory: [4, 60],
  holding_cost: [0.1, 2.0],
  backlog_cost: [0.2, 4.0],
  max_inbound_per_link: [50, 300],
  max_order: [50, 300],
};

const formatRangeLabel = (key) => key.replaceAll('_', ' ');

const TrainingPanel = () => {
  const [serverHost, setServerHost] = useState('aiserver.local');
  const [source, setSource] = useState('sim');
  const [windowSize, setWindowSize] = useState(12);
  const [horizon, setHorizon] = useState(1);
  const [epochs, setEpochs] = useState(10);
  const [device, setDevice] = useState('cpu');
  const [dataPath, setDataPath] = useState('');
  const [stepsTable, setStepsTable] = useState('beer_game_steps');
  const [dbUrl, setDbUrl] = useState('');
  const [useSimpy, setUseSimpy] = useState(true);
  const [simAlpha, setSimAlpha] = useState(0.3);
  const [simWipK, setSimWipK] = useState(1.0);
  const [ranges, setRanges] = useState(DEFAULT_RANGES);
  const [job, setJob] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
  const [working, setWorking] = useState({ generate: false, train: false, stop: false });

  const showMessage = useCallback((message, severity = 'info') => {
    setSnackbar({ open: true, message, severity });
  }, []);

  const closeSnackbar = useCallback(() => {
    setSnackbar((prev) => ({ ...prev, open: false }));
  }, []);

  const numericValue = useCallback((value, fallback = 0) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }, []);

  const handleRangeChange = useCallback((key, index, value) => {
    setRanges((prev) => {
      const next = [...(prev[key] || [0, 0])];
      next[index] = value;
      return { ...prev, [key]: next };
    });
  }, []);

  const generateDataset = useCallback(async () => {
    setWorking((prev) => ({ ...prev, generate: true }));
    try {
      const param_ranges = Object.fromEntries(
        Object.entries(ranges).map(([key, pair]) => [
          key,
          [numericValue(pair?.[0], 0), numericValue(pair?.[1], 0)],
        ]),
      );

      const payload = {
        num_runs: 64,
        T: 64,
        window: numericValue(windowSize, 12),
        horizon: numericValue(horizon, 1),
        param_ranges,
        use_simpy: useSimpy,
        sim_alpha: numericValue(simAlpha, 0.3),
        sim_wip_k: numericValue(simWipK, 1.0),
      };

      const result = await mixedGameApi.generateData(payload);
      setDataPath(result?.path || '');
      showMessage(result?.path ? `Dataset generated at ${result.path}` : 'Dataset generated', 'success');
    } catch (error) {
      setDataPath('');
      showMessage(error?.response?.data?.detail || error?.message || 'Failed to generate data', 'error');
    } finally {
      setWorking((prev) => ({ ...prev, generate: false }));
    }
  }, [ranges, windowSize, horizon, useSimpy, simAlpha, simWipK, numericValue, showMessage]);

  const launchTraining = useCallback(async () => {
    setWorking((prev) => ({ ...prev, train: true }));
    try {
      const payload = {
        server_host: serverHost,
        source,
        window: numericValue(windowSize, 12),
        horizon: numericValue(horizon, 1),
        epochs: numericValue(epochs, 10),
        device,
        steps_table: stepsTable,
        db_url: dbUrl || undefined,
      };
      const result = await mixedGameApi.trainModel(payload);
      setJob(result);
      showMessage(result?.note || 'Training started', 'success');
    } catch (error) {
      showMessage(error?.response?.data?.detail || error?.message || 'Failed to start training', 'error');
    } finally {
      setWorking((prev) => ({ ...prev, train: false }));
    }
  }, [serverHost, source, windowSize, horizon, epochs, device, stepsTable, dbUrl, numericValue, showMessage]);

  const stopTraining = useCallback(async () => {
    if (!job?.job_id) return;
    setWorking((prev) => ({ ...prev, stop: true }));
    try {
      await mixedGameApi.stopJob(job.job_id);
      showMessage('Training job stopped', 'info');
    } catch (error) {
      showMessage(error?.response?.data?.detail || error?.message || 'Failed to stop job', 'error');
    } finally {
      setWorking((prev) => ({ ...prev, stop: false }));
    }
  }, [job?.job_id, showMessage]);

  useEffect(() => {
    let timer;
    const poll = async () => {
      if (!job?.job_id) return;
      try {
        const status = await mixedGameApi.getJobStatus(job.job_id);
        setJobStatus(status);
        if (status?.running) {
          timer = setTimeout(poll, 2500);
        }
      } catch (error) {
        setJobStatus(null);
      }
    };
    poll();
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [job?.job_id]);

  const jobActive = Boolean(job?.job_id && jobStatus?.running);

  const datasetSummary = useMemo(() => {
    if (!dataPath) return null;
    return (
      <Alert severity="success" sx={{ mt: 2 }}>
        Dataset available at <strong>{dataPath}</strong>
      </Alert>
    );
  }, [dataPath]);

  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="h6" gutterBottom fontWeight={700}>
          Daybreak Agent Training
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Generate synthetic datasets and launch training jobs for the Daybreak agent. All actions run on the backend service.
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Stack spacing={2}>
              <TextField
                label="Server Host"
                value={serverHost}
                onChange={(event) => setServerHost(event.target.value)}
              />
              <FormControl>
                <InputLabel id="source-label">Source</InputLabel>
                <Select
                  labelId="source-label"
                  label="Source"
                  value={source}
                  onChange={(event) => setSource(event.target.value)}
                >
                  <MenuItem value="sim">Simulator</MenuItem>
                  <MenuItem value="db">Database</MenuItem>
                </Select>
              </FormControl>
              {source === 'db' && (
                <>
                  <TextField
                    label="Database URL"
                    value={dbUrl}
                    onChange={(event) => setDbUrl(event.target.value)}
                    placeholder="mysql+pymysql://user:pass@host/db"
                  />
                  <TextField
                    label="Steps Table"
                    value={stepsTable}
                    onChange={(event) => setStepsTable(event.target.value)}
                  />
                </>
              )}
            </Stack>
          </Grid>
          <Grid item xs={12} md={6}>
            <Stack spacing={2}>
              <TextField
                label="Window"
                type="number"
                inputProps={{ min: 1, max: 128 }}
                value={windowSize}
                onChange={(event) => setWindowSize(event.target.value)}
              />
              <TextField
                label="Horizon"
                type="number"
                inputProps={{ min: 1, max: 8 }}
                value={horizon}
                onChange={(event) => setHorizon(event.target.value)}
              />
              <TextField
                label="Epochs"
                type="number"
                inputProps={{ min: 1, max: 500 }}
                value={epochs}
                onChange={(event) => setEpochs(event.target.value)}
              />
              <FormControl>
                <InputLabel id="device-label">Device</InputLabel>
                <Select
                  labelId="device-label"
                  label="Device"
                  value={device}
                  onChange={(event) => setDevice(event.target.value)}
                >
                  <MenuItem value="cpu">CPU</MenuItem>
                  <MenuItem value="cuda">GPU</MenuItem>
                </Select>
              </FormControl>
            </Stack>
          </Grid>
        </Grid>

        <Divider sx={{ my: 4 }} />

        <Typography variant="subtitle1" fontWeight={700} gutterBottom>
          Simulation Settings
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={4}>
            <FormControlLabel
              control={(
                <Switch
                  checked={useSimpy}
                  onChange={(event) => setUseSimpy(event.target.checked)}
                />
              )}
              label="Use SimPy"
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <TextField
              label="Smoothing Alpha"
              type="number"
              inputProps={{ step: 0.05, min: 0, max: 1 }}
              value={simAlpha}
              onChange={(event) => setSimAlpha(event.target.value)}
            />
          </Grid>
          <Grid item xs={12} sm={4}>
            <TextField
              label="WIP Gain"
              type="number"
              inputProps={{ step: 0.1, min: 0, max: 5 }}
              value={simWipK}
              onChange={(event) => setSimWipK(event.target.value)}
            />
          </Grid>
        </Grid>

        <Typography variant="subtitle1" fontWeight={700} sx={{ mt: 4 }} gutterBottom>
          Parameter Ranges
        </Typography>
        <Grid container spacing={2}>
          {Object.entries(ranges).map(([key, pair]) => (
            <Grid item xs={12} md={6} key={key}>
              <Stack direction="row" alignItems="center" spacing={2}>
                <Box sx={{ minWidth: 160, textTransform: 'capitalize' }}>{formatRangeLabel(key)}</Box>
                <TextField
                  label="Min"
                  type="number"
                  value={pair?.[0] ?? ''}
                  onChange={(event) => handleRangeChange(key, 0, event.target.value)}
                  size="small"
                />
                <TextField
                  label="Max"
                  type="number"
                  value={pair?.[1] ?? ''}
                  onChange={(event) => handleRangeChange(key, 1, event.target.value)}
                  size="small"
                />
              </Stack>
            </Grid>
          ))}
        </Grid>

        {datasetSummary}

        <Stack direction="row" spacing={2} sx={{ mt: 4 }} justifyContent="flex-end">
          <Button
            variant="outlined"
            onClick={generateDataset}
            disabled={working.generate}
          >
            {working.generate ? 'Generating…' : 'Generate Data'}
          </Button>
          <Button
            variant="contained"
            onClick={launchTraining}
            disabled={working.train || !dataPath}
          >
            {working.train ? 'Starting…' : 'Launch Training'}
          </Button>
          {jobActive && (
            <Button
              variant="outlined"
              color="error"
              onClick={stopTraining}
              disabled={working.stop}
            >
              {working.stop ? 'Stopping…' : 'Stop'}
            </Button>
          )}
        </Stack>

        {job && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="subtitle1" fontWeight={700}>Job Details</Typography>
            <Typography variant="body2">Job ID: {job.job_id || '—'}</Typography>
            {job.cmd && (
              <Typography variant="body2" sx={{ mt: 1 }}>
                Command: <code>{job.cmd}</code>
              </Typography>
            )}
            {job.note && (
              <Alert severity="info" sx={{ mt: 1 }}>{job.note}</Alert>
            )}
            {jobStatus && (
              <Card variant="outlined" sx={{ mt: 2 }}>
                <CardContent>
                  <Typography variant="subtitle2" fontWeight={700}>Status</Typography>
                  <Typography variant="body2">Running: {String(jobStatus.running)}</Typography>
                  <Typography variant="body2">PID: {jobStatus.pid || '—'}</Typography>
                  {jobStatus.log_tail && (
                    <TextField
                      label="Log Tail"
                      value={jobStatus.log_tail}
                      multiline
                      minRows={6}
                      InputProps={{ readOnly: true }}
                      fullWidth
                      sx={{ mt: 2 }}
                    />
                  )}
                </CardContent>
              </Card>
            )}
          </Box>
        )}
      </CardContent>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={4000}
        onClose={closeSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert severity={snackbar.severity} onClose={closeSnackbar} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Card>
  );
};

const Training = () => (
  <PageLayout title="Daybreak Agent Training">
    <TrainingPanel />
  </PageLayout>
);

export { TrainingPanel };
export default Training;
