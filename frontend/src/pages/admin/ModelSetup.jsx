import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  TextField,
  MenuItem,
  IconButton,
  Select,
  Checkbox,
  ListItemText,
  FormControl,
  InputLabel,
  OutlinedInput,
  Tooltip,
  CircularProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemIcon,
  Typography,
} from '@mui/material';
import {
  Delete as DeleteIcon,
  Add as AddIcon,
  AutoFixHigh as AutoFixIcon,
  CheckCircle,
  ErrorOutline,
} from '@mui/icons-material';
import PageLayout from '../../components/PageLayout';
import { mixedGameApi } from '../../services/api';
import { useSystemConfig } from '../../contexts/SystemConfigContext.jsx';

const siteTypes = ['manufacturer','distributor','wholesaler','retailer'];

const classicPreset = () => ({
  version: 1,
  items: [{ id: 'item_1', name: 'Item 1' }],
  sites: [
    { id: 'manufacturer_1', type: 'manufacturer', name: 'Manufacturer 1', items_sold: ['item_1'] },
    { id: 'distributor_1', type: 'distributor', name: 'Distributor 1', items_sold: ['item_1'] },
    { id: 'wholesaler_1', type: 'wholesaler', name: 'Wholesaler 1', items_sold: ['item_1'] },
    { id: 'retailer_1', type: 'retailer', name: 'Retailer 1', items_sold: ['item_1'] },
  ],
  site_item_settings: {
    manufacturer_1: { item_1: { inventory_target: 20, holding_cost: 0.5, backorder_cost: 1.0, avg_selling_price: 7.0, standard_cost: 5.0, moq: 0 } },
    distributor_1: { item_1: { inventory_target: 20, holding_cost: 0.5, backorder_cost: 1.0, avg_selling_price: 7.0, standard_cost: 5.0, moq: 0 } },
    wholesaler_1: { item_1: { inventory_target: 20, holding_cost: 0.5, backorder_cost: 1.0, avg_selling_price: 7.0, standard_cost: 5.0, moq: 0 } },
    retailer_1: { item_1: { inventory_target: 20, holding_cost: 0.5, backorder_cost: 1.0, avg_selling_price: 7.0, standard_cost: 5.0, moq: 0 } },
  },
  lanes: [
    { from_site_id: 'manufacturer_1', to_site_id: 'distributor_1', item_id: 'item_1', lead_time: 2, capacity: null, otif_target: 0.95 },
    { from_site_id: 'distributor_1', to_site_id: 'wholesaler_1', item_id: 'item_1', lead_time: 2, capacity: null, otif_target: 0.95 },
    { from_site_id: 'wholesaler_1', to_site_id: 'retailer_1', item_id: 'item_1', lead_time: 2, capacity: null, otif_target: 0.95 },
  ],
  retailer_demand: { distribution: 'profile', params: { week1_4: 4, week5_plus: 8 }, expected_delivery_offset: 1 },
  manufacturer_lead_times: { item_1: 2 },
});

function numberIn(range, v) {
  if (!range) return true;
  if (typeof v !== 'number' || Number.isNaN(v)) return false;
  return v >= range.min && v <= range.max;
}

export default function ModelSetup() {
  const navigate = useNavigate();
  const { ranges } = useSystemConfig();
  const [cfg, setCfg] = useState(classicPreset());
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [processDialogOpen, setProcessDialogOpen] = useState(false);
  const [processSteps, setProcessSteps] = useState([]);
  const autoCloseRef = useRef(null);

  useEffect(() => {
    (async () => {
      try {
        const data = await mixedGameApi.getModelConfig();
        if (data) setCfg(data);
      } catch (e) {
        // keep preset
      } finally { setLoading(false); }
    })();
  }, []);

  const items = cfg.items || [];
  const sites = cfg.sites || [];

  const setSiteItem = (siteId, itemId, field, value) => {
    setCfg((prev) => ({
      ...prev,
      site_item_settings: {
        ...(prev.site_item_settings || {}),
        [siteId]: {
          ...(prev.site_item_settings?.[siteId] || {}),
          [itemId]: { ...(prev.site_item_settings?.[siteId]?.[itemId] || {}), [field]: value }
        }
      }
    }));
  };

  const setLane = (idx, field, value) => {
    setCfg((prev) => {
      const lanes = [...(prev.lanes || [])];
      lanes[idx] = { ...lanes[idx], [field]: value };
      return { ...prev, lanes };
    });
  };

  const addItem = () => {
    const n = (items?.length || 0) + 1;
    const newItem = { id: `item_${n}`, name: `Item ${n}` };
    setCfg((prev) => ({ ...prev, items: [...(prev.items || []), newItem] }));
  };

  const removeItem = (itemId) => {
    setCfg((prev) => {
      const items = (prev.items || []).filter((it) => it.id !== itemId);
      const sites = (prev.sites || []).map((s) => ({ ...s, items_sold: (s.items_sold || []).filter((id) => id !== itemId) }));
      // remove site_item_settings entries for this item
      const sis = Object.fromEntries(Object.entries(prev.site_item_settings || {}).map(([sid, m]) => [sid, Object.fromEntries(Object.entries(m).filter(([iid]) => iid !== itemId))]));
      // remove lanes for this item
      const lanes = (prev.lanes || []).filter((ln) => ln.item_id !== itemId);
      return { ...prev, items, sites, site_item_settings: sis, lanes };
    });
  };

  const addSite = (type) => {
    // next index for type
    const idx = (sites.filter((s) => s.type === type).length || 0) + 1;
    const id = `${type}_${idx}`;
    const name = `${type.charAt(0).toUpperCase() + type.slice(1)} ${idx}`;
    setCfg((prev) => ({ ...prev, sites: [...(prev.sites || []), { id, type, name, items_sold: prev.items?.map((i) => i.id) || [] }] }));
  };

  const removeSite = (siteId) => {
    setCfg((prev) => {
      const sites = (prev.sites || []).filter((s) => s.id !== siteId);
      // remove settings for siteId
      const sis = Object.fromEntries(Object.entries(prev.site_item_settings || {}).filter(([sid]) => sid !== siteId));
      // remove lanes touching siteId
      const lanes = (prev.lanes || []).filter((ln) => ln.from_site_id !== siteId && ln.to_site_id !== siteId);
      return { ...prev, sites, site_item_settings: sis, lanes };
    });
  };

  const setItemsSold = (siteId, itemIds) => {
    setCfg((prev) => ({
      ...prev,
      sites: (prev.sites || []).map((s) => (s.id === siteId ? { ...s, items_sold: itemIds } : s)),
    }));
  };

  const addLane = () => {
    const firstItem = items[0]?.id || 'item_1';
    const manufacturer = sites.find((s) => s.type === 'manufacturer')?.id || '';
    const retailer = sites.find((s) => s.type === 'retailer')?.id || '';
    setCfg((prev) => ({ ...prev, lanes: [...(prev.lanes || []), { from_site_id: manufacturer, to_site_id: retailer, item_id: firstItem, lead_time: 1, capacity: null, otif_target: 0.95 }] }));
  };

  const generateChainLanesAllToAll = () => {
    const typesOrder = ['manufacturer','distributor','wholesaler','retailer'];
    const byType = Object.fromEntries(typesOrder.map((t) => [t, sites.filter((s) => s.type === t)]));
    const newLanes = [];
    for (let i = 0; i < typesOrder.length - 1; i++) {
      const froms = byType[typesOrder[i]];
      const tos = byType[typesOrder[i+1]];
      for (const it of items) {
        for (const f of froms) {
          for (const t of tos) {
            newLanes.push({ from_site_id: f.id, to_site_id: t.id, item_id: it.id, lead_time: 2, capacity: null, otif_target: 0.95 });
          }
        }
      }
    }
    setCfg((prev) => ({ ...prev, lanes: newLanes }));
  };

  const violations = useMemo(() => {
    const errs = [];
    // site-item checks
    for (const site of sites) {
      for (const item of items) {
        const s = cfg.site_item_settings?.[site.id]?.[item.id];
        if (!s) continue;
        if (!numberIn(ranges?.init_inventory, s.inventory_target)) errs.push(`${site.name}/${item.name}: inventory_target`);
        if (!numberIn(ranges?.holding_cost, s.holding_cost)) errs.push(`${site.name}/${item.name}: holding_cost`);
        if (!numberIn(ranges?.backlog_cost, s.backorder_cost)) errs.push(`${site.name}/${item.name}: backorder_cost`);
        if (!numberIn(ranges?.price, s.avg_selling_price)) errs.push(`${site.name}/${item.name}: avg_selling_price`);
        if (!numberIn(ranges?.standard_cost, s.standard_cost)) errs.push(`${site.name}/${item.name}: standard_cost`);
        if (!numberIn(ranges?.min_order_qty, s.moq)) errs.push(`${site.name}/${item.name}: moq`);
      }
    }
    // lanes
    for (const [i, lane] of (cfg.lanes || []).entries()) {
      if (!numberIn(ranges?.order_leadtime, lane.lead_time)) errs.push(`Lane ${i+1}: lead_time`);
      if (lane.capacity != null && !numberIn(ranges?.max_inbound_per_link, Number(lane.capacity))) errs.push(`Lane ${i+1}: capacity`);
    }
    return errs;
  }, [cfg, ranges, items, sites]);

  const closeProcessDialog = () => {
    if (autoCloseRef.current) {
      clearTimeout(autoCloseRef.current);
      autoCloseRef.current = null;
    }
    setProcessDialogOpen(false);
    setProcessSteps([]);
  };

  useEffect(() => () => {
    if (autoCloseRef.current) {
      clearTimeout(autoCloseRef.current);
    }
  }, []);

  const save = async () => {
    setSaving(true);
    setError(null);
    setProcessSteps([
      {
        id: 'save-model-config',
        label: 'Saving model configuration',
        status: 'running',
        message: 'Submitting your latest changes to the server.',
      },
    ]);
    setProcessDialogOpen(true);
    try {
      const saved = await mixedGameApi.saveModelConfig(cfg);
      setCfg(saved);
      setProcessSteps((prev) =>
        prev.map((step) =>
          step.id === 'save-model-config'
            ? { ...step, status: 'success', message: 'Configuration saved successfully.' }
            : step
        )
      );
      if (autoCloseRef.current) {
        clearTimeout(autoCloseRef.current);
      }
      autoCloseRef.current = setTimeout(() => {
        closeProcessDialog();
      }, 1200);
    } catch (e) {
      const detail = e?.response?.data?.detail;
      const message = typeof detail === 'string' ? detail : detail?.message || 'Failed to save configuration.';
      setError(message);
      setProcessSteps((prev) =>
        prev.map((step) =>
          step.id === 'save-model-config'
            ? { ...step, status: 'error', message }
            : step
        )
      );
    } finally {
      setSaving(false);
    }
  };

  if (loading) return <PageLayout title="Model Setup"><div className="pad-6">Loading...</div></PageLayout>;

  const hasActiveProcess = processSteps.some((step) => step.status === 'running');
  const canCloseProcessDialog = processSteps.length > 0 && !hasActiveProcess;

  return (
    <PageLayout title="Model Setup">
      <Box className="card-surface pad-6 space-y-6">
        <Box display="flex" gap={1}>
          <Button variant="contained" onClick={() => setCfg(classicPreset())}>Classic Beer Game Preset</Button>
          <Button variant="outlined" onClick={() => navigate('/admin')}>Back to Admin</Button>
        </Box>

        {error && <Box className="bg-red-50 text-red-700 p-3 rounded">{error}</Box>}

        {/* Items */}
        <Box>
          <h3 className="text-lg font-semibold mb-2">Items</h3>
          <Box mb={1}>
            <Button size="small" startIcon={<AddIcon />} onClick={addItem}>Add Item</Button>
          </Box>
          {items.map((it, idx) => (
            <Box key={it.id} display="flex" gap={2} mb={1} alignItems="center">
              <TextField size="small" label="Item ID" value={it.id} onChange={(e) => {
                const items2 = [...items]; items2[idx] = { ...it, id: e.target.value }; setCfg({ ...cfg, items: items2 });
              }} />
              <TextField size="small" label="Name" value={it.name} onChange={(e) => {
                const items2 = [...items]; items2[idx] = { ...it, name: e.target.value }; setCfg({ ...cfg, items: items2 });
              }} />
              <Tooltip title="Remove Item">
                <IconButton color="error" onClick={() => removeItem(it.id)}><DeleteIcon /></IconButton>
              </Tooltip>
            </Box>
          ))}
        </Box>

        {/* Sites */}
        <Box>
          <h3 className="text-lg font-semibold mb-2">Sites</h3>
          <Box display="flex" gap={1} mb={1}>
            {siteTypes.map((t) => (
              <Button key={t} size="small" startIcon={<AddIcon />} onClick={() => addSite(t)}>
                Add {t}
              </Button>
            ))}
          </Box>
          {sites.map((s, idx) => (
            <Box key={s.id} display="flex" gap={2} mb={1} alignItems="center">
              <TextField size="small" label="Site ID" value={s.id} onChange={(e) => {
                const sites2 = [...sites]; sites2[idx] = { ...s, id: e.target.value }; setCfg({ ...cfg, sites: sites2 });
              }} />
              <TextField select size="small" label="Type" value={s.type} onChange={(e) => {
                const sites2 = [...sites]; sites2[idx] = { ...s, type: e.target.value }; setCfg({ ...cfg, sites: sites2 });
              }}>
                {siteTypes.map((t) => <MenuItem key={t} value={t}>{t}</MenuItem>)}
              </TextField>
              <TextField size="small" label="Name" value={s.name} onChange={(e) => {
                const sites2 = [...sites]; sites2[idx] = { ...s, name: e.target.value }; setCfg({ ...cfg, sites: sites2 });
              }} />
              <FormControl size="small" sx={{ minWidth: 220 }}>
                <InputLabel id={`items-sold-${s.id}`}>Items Sold</InputLabel>
                <Select
                  labelId={`items-sold-${s.id}`}
                  multiple
                  value={s.items_sold || []}
                  onChange={(e) => setItemsSold(s.id, e.target.value)}
                  input={<OutlinedInput label="Items Sold" />}
                  renderValue={(selected) => items.filter((it) => selected.includes(it.id)).map((it) => it.name).join(', ')}
                >
                  {items.map((it) => (
                    <MenuItem key={it.id} value={it.id}>
                      <Checkbox checked={(s.items_sold || []).indexOf(it.id) > -1} />
                      <ListItemText primary={it.name} />
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Tooltip title="Remove Site"><IconButton color="error" onClick={() => removeSite(s.id)}><DeleteIcon /></IconButton></Tooltip>
            </Box>
          ))}
        </Box>

        {/* Site-Item Settings table for first item (classic) */}
        <Box>
          <h3 className="text-lg font-semibold mb-2">Site-Item Settings</h3>
          {sites.map((s) => (
            <Box key={s.id} className="rounded border p-3 mb-2">
              <div className="font-medium mb-2">{s.name}</div>
              {items.map((it) => {
                const st = cfg.site_item_settings?.[s.id]?.[it.id] || {};
                const field = (name, label, range, step=1) => (
                  <TextField
                    type="number" size="small" sx={{ mr: 1, mb: 1, width: 180 }}
                    label={`${label}${range ? ` [${range.min}-${range.max}]` : ''}`}
                    value={st[name] ?? ''}
                    error={range && !numberIn(range, Number(st[name]))}
                    onChange={(e) => setSiteItem(s.id, it.id, name, e.target.valueAsNumber)}
                    inputProps={{ step }}
                  />
                );
                return (
                  <Box key={it.id}>
                    <div className="text-sm text-gray-600 mb-1">{it.name}</div>
                    {field('inventory_target','Inventory Target', ranges?.init_inventory)}
                    {field('holding_cost','Holding Cost', ranges?.holding_cost, 0.1)}
                    {field('backorder_cost','Backorder Cost', ranges?.backlog_cost, 0.1)}
                    {field('avg_selling_price','Avg Selling Price', ranges?.price, 0.1)}
                    {field('standard_cost','Standard Cost', ranges?.standard_cost, 0.1)}
                    {field('moq','MOQ', ranges?.min_order_qty)}
                  </Box>
                );
              })}
            </Box>
          ))}
        </Box>

        {/* Lanes */}
        <Box>
          <h3 className="text-lg font-semibold mb-2">Lanes</h3>
          <Box display="flex" gap={1} mb={1}>
            <Button size="small" startIcon={<AddIcon />} onClick={addLane}>Add Lane</Button>
            <Button size="small" startIcon={<AutoFixIcon />} onClick={generateChainLanesAllToAll}>Generate Chain Lanes (all-to-all)</Button>
          </Box>
          {(cfg.lanes || []).map((ln, i) => (
            <Box key={i} display="flex" gap={2} mb={1}>
              <TextField size="small" label="From" value={ln.from_site_id} onChange={(e) => setLane(i,'from_site_id', e.target.value)} />
              <TextField size="small" label="To" value={ln.to_site_id} onChange={(e) => setLane(i,'to_site_id', e.target.value)} />
              <TextField size="small" label="Item" value={ln.item_id} onChange={(e) => setLane(i,'item_id', e.target.value)} />
              <TextField size="small" type="number" label={`Lead Time ${ranges?.order_leadtime ? `[${ranges.ship_order_leadtimedelay.min}-${ranges.order_leadtime.max}]` : ''}`} value={ln.lead_time}
                error={ranges?.order_leadtime && !numberIn(ranges.order_leadtime, Number(ln.lead_time))}
                onChange={(e) => setLane(i,'lead_time', e.target.valueAsNumber)} />
              <TextField size="small" type="number" label={`Capacity ${ranges?.max_inbound_per_link ? `[${ranges.max_inbound_per_link.min}-${ranges.max_inbound_per_link.max}]` : ''}`}
                value={ln.capacity ?? ''}
                error={ln.capacity != null && ranges?.max_inbound_per_link && !numberIn(ranges.max_inbound_per_link, Number(ln.capacity))}
                onChange={(e) => setLane(i,'capacity', e.target.value === '' ? null : e.target.valueAsNumber)} />
              <TextField size="small" type="number" label="OTIF Target (0-1 or %)" value={ln.otif_target ?? ''}
                onChange={(e) => setLane(i,'otif_target', e.target.value === '' ? null : e.target.valueAsNumber)} />
              <Tooltip title="Remove Lane"><IconButton color="error" onClick={() => setCfg((prev) => ({ ...prev, lanes: (prev.lanes || []).filter((_, j) => j !== i) }))}><DeleteIcon /></IconButton></Tooltip>
            </Box>
          ))}
        </Box>

        {/* Save */}
        <Box display="flex" gap={1}>
          <Button variant="contained" color="primary" disabled={saving || violations.length > 0} onClick={save}>Save</Button>
          {violations.length > 0 && (
            <Box className="text-sm text-red-600" ml={2}>Out of range: {violations.join(', ')}</Box>
          )}
        </Box>
      </Box>
      <Dialog
        open={processDialogOpen}
        onClose={(_, reason) => {
          if (hasActiveProcess && (reason === 'backdropClick' || reason === 'escapeKeyDown')) {
            return;
          }
          if (canCloseProcessDialog) {
            closeProcessDialog();
          }
        }}
        fullWidth
        maxWidth="xs"
      >
        <DialogTitle>Processing updates</DialogTitle>
        <DialogContent dividers>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            We'll keep this window open while we finish saving your model configuration.
          </Typography>
          <List disablePadding>
            {processSteps.map((step) => (
              <ListItem key={step.id} disableGutters>
                <ListItemIcon sx={{ minWidth: 36 }}>
                  {step.status === 'success' ? (
                    <CheckCircle color="success" />
                  ) : step.status === 'error' ? (
                    <ErrorOutline color="error" />
                  ) : (
                    <CircularProgress size={20} />
                  )}
                </ListItemIcon>
                <ListItemText
                  primary={step.label}
                  secondary={step.message}
                  primaryTypographyProps={{ fontWeight: 500 }}
                />
              </ListItem>
            ))}
          </List>
        </DialogContent>
        {canCloseProcessDialog && (
          <DialogActions>
            <Button onClick={closeProcessDialog}>Close</Button>
          </DialogActions>
        )}
      </Dialog>
    </PageLayout>
  );
}
