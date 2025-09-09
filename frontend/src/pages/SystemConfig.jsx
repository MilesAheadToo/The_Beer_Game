import React, { useEffect, useState } from 'react';
import PageLayout from '../components/PageLayout';
import { mixedGameApi } from '../services/api';
import { Box, Grid, FormControl, FormLabel, NumberInput, NumberInputField, NumberInputStepper, NumberIncrementStepper, NumberDecrementStepper, Button, Text } from '@chakra-ui/react';

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
  variable_cost: { min: 0, max: 10000 },
  min_order_qty: { min: 0, max: 1000 },
};

export default function SystemConfig() {
  const [ranges, setRanges] = useState(DEFAULTS);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const data = await mixedGameApi.getSystemConfig();
        setRanges({ ...DEFAULTS, ...data });
      } catch {
        const raw = localStorage.getItem('systemConfigRanges');
        if (raw) {
          try { setRanges({ ...DEFAULTS, ...JSON.parse(raw) }); } catch {}
        }
      }
    })();
  }, []);

  const update = (key, field, val) => {
    setRanges((prev) => ({ ...prev, [key]: { ...prev[key], [field]: Number(val) || 0 } }));
  };

  const save = async () => {
    try {
      await mixedGameApi.saveSystemConfig(ranges);
    } finally {
      localStorage.setItem('systemConfigRanges', JSON.stringify(ranges));
      setSaved(true);
      setTimeout(() => setSaved(false), 1500);
    }
  };

  return (
    <PageLayout title="System Configuration">
      <Box className="card-surface pad-6">
        <Text color="gray.600" mb={4}>Set allowable ranges for configuration variables. These will prefill the Mixed Game Definition page.</Text>
        <Grid templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)' }} gap={6}>
          {Object.entries(ranges).map(([key, rng]) => (
            <Box key={key}>
              <FormControl mb={3}>
                <FormLabel textTransform="capitalize">{key.replaceAll('_',' ')} Min</FormLabel>
                <NumberInput value={rng.min} onChange={(v) => update(key, 'min', v)}>
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>
              <FormControl>
                <FormLabel textTransform="capitalize">{key.replaceAll('_',' ')} Max</FormLabel>
                <NumberInput value={rng.max} onChange={(v) => update(key, 'max', v)}>
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>
            </Box>
          ))}
        </Grid>
        <Box mt={6} textAlign="right">
          <Button colorScheme="blue" onClick={save}>Save Ranges</Button>
          {saved && <Text as="span" ml={3} color="green.600">Saved</Text>}
        </Box>
      </Box>
    </PageLayout>
  );
}
