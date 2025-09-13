import React, { useEffect, useState } from 'react';
import PageLayout from '../components/PageLayout';
import { mixedGameApi } from '../services/api';
import { Box, Grid, FormControl, FormLabel, NumberInput, NumberInputField, NumberInputStepper, NumberIncrementStepper, NumberDecrementStepper, Button, Text, Table, Thead, Tbody, Tr, Th, Td, Input, Select, Flex } from '@chakra-ui/react';

const DEFAULTS = {
  info_delay: { min: 0, max: 8 },
  ship_delay: { min: 0, max: 8 },
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
  info_delay: 'Supply Leadtime',
  ship_delay: 'Order Leadtime',
  max_inbound_per_link: 'Inbound Lane Capacity',
  min_order_qty: 'MOQ',
};

export default function SystemConfig() {
  const [ranges, setRanges] = useState(DEFAULTS);
  const [saved, setSaved] = useState(false);
  const [name, setName] = useState('Undefined');
  const [configs, setConfigs] = useState([]);

  useEffect(() => {
    (async () => {
      try {
        const data = await mixedGameApi.getSystemConfig();
        const { variable_cost, name: cfgName, ...rest } = data || {};
        setName(cfgName || 'Undefined');
        setRanges({ ...DEFAULTS, ...rest });
      } catch {
        const raw = localStorage.getItem('systemConfigRanges');
        if (raw) {
          try {
            const parsed = JSON.parse(raw);
            setName(parsed?.name || 'Undefined');
            const { variable_cost, name: _n, ...rest } = parsed || {};
            setRanges({ ...DEFAULTS, ...rest });
          } catch {}
        }
      }

      try {
        const cfgs = await mixedGameApi.getSupplyChainConfigs();
        setConfigs(cfgs || []);
        if (cfgs && cfgs.length > 0) {
          setName(cfgs[0].name);
        }
      } catch {
        setConfigs([]);
      }    })();
  }, []);

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

        <FormControl mb={4} maxW="lg">
        <FormLabel>Configuration</FormLabel>
          <Flex align="center">
            <Select value={name} onChange={(e)=> setName(e.target.value)} maxW="sm">
              {configs.map((c) => (
                <option key={c.id} value={c.name}>{c.name}</option>
              ))}
            </Select>
            <Text ml={4}>{configs.length} configurations defined</Text>
          </Flex>
        </FormControl>
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
        <Box mt={6} textAlign="right">
          <Button colorScheme="blue" onClick={save}>Save Ranges</Button>
          {saved && <Text as="span" ml={3} color="green.600">Saved</Text>}
        </Box>
      </Box>
    </PageLayout>
  );
}
