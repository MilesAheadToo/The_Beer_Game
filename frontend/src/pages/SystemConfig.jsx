import React, { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from '@mui/material';
import PageLayout from '../components/PageLayout';
import { mixedGameApi, api } from '../services/api';
import { getItems, getNodes, getLanes } from '../services/supplyChainConfigService';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { isGroupAdmin as isGroupAdminUser } from '../utils/authUtils';

const DEFAULTS = {
  supply_leadtime: { min: 0, max: 8 },
  order_leadtime: { min: 0, max: 8 },
  init_inventory: { min: 0, max: 1000 },
  holding_cost: { min: 0, max: 100 },
  backlog_cost: { min: 0, max: 200 },
  max_inbound_per_link: { min: 10, max: 2000 },
  max_order: { min: 10, max: 2000 },
  price: { min: 0, max: 10000 },
  standard_cost: { min: 0, max: 10000 },
  min_order_qty: { min: 0, max: 1000 },
};

const LABELS = {
  order_leadtime: 'Order Leadtime',
  supply_leadtime: 'Supply Leadtime',
  max_inbound_per_link: 'Inbound Lane Capacity',
  min_order_qty: 'MOQ',
};

const SystemConfig = () => {
  const [ranges, setRanges] = useState(DEFAULTS);
  const [saved, setSaved] = useState(false);
  const [name, setName] = useState('Undefined');
  const [configs, setConfigs] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [counts, setCounts] = useState({ items: 0, nodes: 0, lanes: 0 });
  const [loadingConfigs, setLoadingConfigs] = useState(true);
  const navigate = useNavigate();
  const { user } = useAuth();
  const isGroupAdmin = isGroupAdminUser(user);
  const scConfigBasePath = isGroupAdmin ? '/admin/group/supply-chain-configs' : '/supply-chain-config';

  useEffect(() => {
    let active = true;
    const init = async () => {
      let configuredName;
      try {
        const data = await mixedGameApi.getSystemConfig();
        const { name: cfgName, variable_cost, ...rest } = data || {};
        configuredName = cfgName;
        if (!active) return;
        setName(cfgName || 'Undefined');
        setRanges({ ...DEFAULTS, ...rest });
      } catch (error) {
        const fallback = localStorage.getItem('systemConfigRanges');
        if (fallback) {
          try {
            const parsed = JSON.parse(fallback);
            const { name: cfgName, variable_cost, ...rest } = parsed || {};
            configuredName = cfgName;
            if (!active) return;
            setName(cfgName || 'Undefined');
            setRanges({ ...DEFAULTS, ...rest });
          } catch (_) {
            // ignore invalid cache
          }
        }
      }

      try {
        const response = await api.get('/supply-chain-config/');
        if (!active) return;
        const list = Array.isArray(response.data) ? response.data : [];
        setConfigs(list);
        const matching = list.find((cfg) => cfg.name === configuredName);
        if (matching) {
          setSelectedId(String(matching.id));
        }
      } catch (error) {
        if (active) {
          setConfigs([]);
        }
      } finally {
        if (active) setLoadingConfigs(false);
      }
    };

    init();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setCounts({ items: 0, nodes: 0, lanes: 0 });
      return;
    }

    const fetchCounts = async (configId) => {
      try {
        const [items, nodes, lanes] = await Promise.all([
          getItems(configId),
          getNodes(configId),
          getLanes(configId),
        ]);
        setCounts({
          items: items.length,
          nodes: nodes.length,
          lanes: lanes.length,
        });
      } catch (error) {
        setCounts({ items: 0, nodes: 0, lanes: 0 });
      }
    };

    fetchCounts(Number(selectedId));
  }, [selectedId]);

  const handleRangeChange = (key, field, value) => {
    setRanges((prev) => ({
      ...prev,
      [key]: {
        ...prev[key],
        [field]: Number(value),
      },
    }));
  };

  const handleSelectChange = (event) => {
    const id = event.target.value;
    setSelectedId(id);
    const cfg = configs.find((config) => String(config.id) === String(id));
    setName(cfg?.name || 'Undefined');
  };

  const handleSave = async () => {
    try {
      await mixedGameApi.saveSystemConfig({ name, ...ranges });
      showSaved();
    } catch (error) {
      showSaved();
    }
  };

  const showSaved = () => {
    setSaved(true);
    localStorage.setItem('systemConfigRanges', JSON.stringify({ name, ...ranges }));
    setTimeout(() => setSaved(false), 1500);
  };

  const navigateToConfig = () => {
    if (!selectedId) return;
    navigate(`${scConfigBasePath}/edit/${selectedId}`);
  };

  const systemSummary = useMemo(() => (
    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
      Define allowable ranges for configuration variables. These ranges seed the Mixed Game definition workflow.
    </Typography>
  ), []);

  return (
    <PageLayout title="System Configuration">
      <Card variant="outlined">
        <CardContent>
          <Typography variant="h6" fontWeight={700} gutterBottom>System Configuration</Typography>
          {systemSummary}

          <Box display="flex" gap={2} alignItems="flex-end" flexWrap="wrap" sx={{ mb: 3 }}>
            <FormControl sx={{ minWidth: 260 }}>
              <InputLabel id="config-select-label">Configuration Name ({configs.length})</InputLabel>
              <Select
                labelId="config-select-label"
                label={`Configuration Name (${configs.length})`}
                value={selectedId}
                onChange={handleSelectChange}
                displayEmpty
              >
                <MenuItem value="">
                  <em>Select configuration</em>
                </MenuItem>
                {configs.map((config) => (
                  <MenuItem key={config.id} value={String(config.id)}>
                    {config.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              label="Active Name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              sx={{ minWidth: 220 }}
            />
            <Button variant="contained" onClick={() => navigate(`${scConfigBasePath}/new`)}>
              New Configuration
            </Button>
          </Box>

          {loadingConfigs && (
            <Alert severity="info" sx={{ mb: 3 }}>Loading configurations…</Alert>
          )}

          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Parameter</TableCell>
                <TableCell align="right">Min Value</TableCell>
                <TableCell align="right">Max Value</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(ranges)
                .filter(([key]) => key !== 'name' && key !== 'variable_cost')
                .map(([key, value]) => (
                  <TableRow key={key}>
                    <TableCell sx={{ textTransform: 'capitalize' }}>
                      {LABELS[key] || key.replaceAll('_', ' ')}
                    </TableCell>
                    <TableCell align="right">
                      <TextField
                        type="number"
                        size="small"
                        value={value.min}
                        onChange={(event) => handleRangeChange(key, 'min', event.target.value)}
                        inputProps={{ step: 1 }}
                      />
                    </TableCell>
                    <TableCell align="right">
                      <TextField
                        type="number"
                        size="small"
                        value={value.max}
                        onChange={(event) => handleRangeChange(key, 'max', event.target.value)}
                        inputProps={{ step: 1 }}
                      />
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>

          <Box sx={{ mt: 4 }}>
            <Typography variant="subtitle1" fontWeight={700}>Definitions</Typography>
            <Typography variant="body2">Items: {counts.items}</Typography>
            <Typography variant="body2">Nodes: {counts.nodes}</Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>Lanes: {counts.lanes}</Typography>
            <Button
              variant="outlined"
              onClick={navigateToConfig}
              disabled={!selectedId || name === 'Undefined'}
            >
              Define Items, Nodes, Lanes
            </Button>
          </Box>

          <Box textAlign="right" sx={{ mt: 4 }}>
            <Button variant="contained" onClick={handleSave}>
              Save Ranges
            </Button>
            {saved && (
              <Typography component="span" color="success.main" sx={{ ml: 2 }}>
                Saved
              </Typography>
            )}
          </Box>
        </CardContent>
      </Card>
    </PageLayout>
  );
};

export default SystemConfig;
