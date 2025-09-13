import React, { useEffect, useState } from 'react';
import PageLayout from '../components/PageLayout';
import { mixedGameApi, api } from '../services/api';
import { getItems, getNodes, getLanes } from '../services/supplyChainConfigService';
import {
  Box,
  Grid,
  FormControl,
  FormLabel,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  NumberIncrementStepper,
  NumberDecrementStepper,
  Button,
  Text,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Select,
} from '@chakra-ui/react';
import { useNavigate } from 'react-router-dom';

const DEFAULTS = {
  order_leadtime: { min: 0, max: 8 },
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

export default function SystemConfig() {
  const [ranges, setRanges] = useState(DEFAULTS);
  const [saved, setSaved] = useState(false);
  const [name, setName] = useState('Undefined');
  const [configs, setConfigs] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [counts, setCounts] = useState({ items: 0, nodes: 0, lanes: 0 });
  const navigate = useNavigate();
  useEffect(() => {
    (async () => {
      let cfgName;
      try {
        const data = await mixedGameApi.getSystemConfig();
        const { variable_cost, name: cfgNameLocal, ...rest } = data || {};
        cfgName = cfgNameLocal;
        setName(cfgNameLocal || 'Undefined');
        setRanges({ ...DEFAULTS, ...rest });
      } catch {
        const raw = localStorage.getItem('systemConfigRanges');
        if (raw) {
          try {
            const parsed = JSON.parse(raw);
            cfgName = parsed?.name;
            setName(parsed?.name || 'Undefined');
            const { variable_cost, name: _n, ...rest } = parsed || {};
            setRanges({ ...DEFAULTS, ...rest });
          } catch {}
        }
      }
  
      try {
        const res = await api.get('/api/v1/supply-chain-config');
        const list = res.data || [];
        setConfigs(list);
        const match = list.find((cfg) => cfg.name === cfgName);
        if (match) {
          setSelectedId(match.id);
        }
      } catch {
        setConfigs([]);
      }
    })();
  }, []);

  useEffect(() => {
    if (selectedId) {
      fetchCounts(selectedId);
    } else {
      setCounts({ items: 0, nodes: 0, lanes: 0 });
    }
  }, [selectedId]);
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
    } catch {
      setCounts({ items: 0, nodes: 0, lanes: 0 });
    }
  };

  const update = (key, field, val) => {
    setRanges((prev) => ({ ...prev, [key]: { ...prev[key], [field]: Number(val) || 0 } }));
  };

  const save = async () => {
    try {
      await mixedGameApi.saveSystemConfig({ name, ...ranges });
    } finally {
      localStorage.setItem('systemConfigRanges', JSON.stringify({ name, ...ranges }));
      setSaved(true);
      setTimeout(() => setSaved(false), 1500);
    }
  };

  return (
    <PageLayout title="System Configuration">
      <Box className="card-surface pad-6">
        <Text color="gray.600" mb={4}>Set allowable ranges for configuration variables. These will prefill the Mixed Game Definition page.</Text>

        <Grid templateColumns="1fr auto" gap={4} mb={4} maxW="lg" alignItems="end">
          <FormControl>
            <FormLabel>Configuration Name ({configs.length})</FormLabel>
            <Select
              value={selectedId ?? ''}
              onChange={(e) => {
                const id = Number(e.target.value);
                setSelectedId(id);
                const cfg = configs.find((c) => c.id === id);
                setName(cfg?.name || 'Undefined');
              }}
              placeholder="Select configuration"
            >
              {configs.map((cfg) => (
                <option key={cfg.id} value={cfg.id}>
                  {cfg.name}
                </option>
              ))}
            </Select>
          </FormControl>
          <Button colorScheme="blue" onClick={() => navigate('/supply-chain-config/new')}>New</Button>
        </Grid>
        <Table variant='simple' size='sm'>
          <Thead>
            <Tr>
              <Th>Parameter</Th>
              <Th isNumeric>Min Value</Th>
              <Th isNumeric>Max Value</Th>
            </Tr>
          </Thead>
          <Tbody>
            {Object.entries(ranges)
              .filter(([key]) => key !== 'name' && key !== 'variable_cost')
              .map(([key, rng]) => (
              <Tr key={key}>
                <Td textTransform="capitalize">{LABELS[key] || key.replaceAll('_',' ')}</Td>
                <Td isNumeric>
                  <NumberInput value={rng.min} onChange={(v) => update(key, 'min', v)} size='sm'>
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </Td>
                <Td isNumeric>
                  <NumberInput value={rng.max} onChange={(v) => update(key, 'max', v)} size='sm'>
                    <NumberInputField />
                    <NumberInputStepper>
                      <NumberIncrementStepper />
                      <NumberDecrementStepper />
                    </NumberInputStepper>
                  </NumberInput>
                </Td>
              </Tr>
            ))}
          </Tbody>
        </Table>
        <Box mt={6}>
          <Text fontWeight="semibold" mb={2}>Definitions</Text>
          <Text fontSize="sm">Items: {counts.items}</Text>
          <Text fontSize="sm">Nodes: {counts.nodes}</Text>
          <Text fontSize="sm">Lanes: {counts.lanes}</Text>
          <Button
            mt={2}
            colorScheme="teal"
            onClick={() => navigate(`/supply-chain-config/edit/${selectedId}`)}
            isDisabled={!selectedId || name === 'Undefined'}
          >
            Define Items, Nodes, Lanes
          </Button>
        </Box>
        <Box mt={6} textAlign="right">
          <Button colorScheme="blue" onClick={save}>Save Ranges</Button>
          {saved && <Text as="span" ml={3} color="green.600">Saved</Text>}
        </Box>
      </Box>
    </PageLayout>
  );
}
